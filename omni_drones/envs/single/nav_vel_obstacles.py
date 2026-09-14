# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
[M2, 2026-09-04] Static 3D-sphere obstacle logic for NavVel (pure torch, no isaac).

Design (kaiwu-inspired sphere model, 3D; see navvel_obstacle_migration_guide.md §4-§6):
  - Obstacle i = sphere at ``p_oi`` (env frame) with *logical* radius ``r_oi`` sampled
    per obstacle from configurable tiers (e.g. [0.20, 0.30, 0.40]).
  - Drone is treated as a sphere ``drone_radius`` (+ ``inflation`` cushion), giving a
    per-obstacle *decision* radius   r_si = drone_radius + r_oi + inflation.
  - Surface clearance (net distance) d_i = ||drone - p_oi|| - r_si  (>0 outside,
    <0 inside). Collision when min_i d_i < collision_margin; danger zone < danger_radius.

Reachability / spawn (NavRL-inspired, 2026-09-04; NavRL pins start/goal to an
obstacle-free border band, we instead enforce per-sample explicit clearance so that a
randomly placed start never *spawns inside* an obstacle and the goal holding zone is
never swallowed by an obstacle):
    dist(obs_center, init) >= r_si + init_clearance
    dist(obs_center, goal) >= r_si + goal_clearance
    dist(oi, oj)           >= r_oi + r_oj + min_gap_between
  Batch layout is done with a jittered grid candidate pool + vectorized greedy
  selection (L rounds, all envs in parallel); on failure the clearances/gap are
  relaxed by ``spawn_relax`` per retry round (init hard clearance kept >= a floor).

All methods are pure tensor ops over the env-frame positions so this file is
unit-testable on CPU and is the only place that needs replacing when obstacle
perception (radar/camera estimates) replaces ground truth later.
"""

import torch


def scene_slot_count(oc) -> int:
    """Number of physical obstacle slots (M) implied by an ``obstacle.*`` config.

    SINGLE SOURCE OF TRUTH. Both the env (which allocates the RigidPrimView with M prims)
    and the ObstacleManager (which allocates M slot buffers) must call this. When the two
    sides disagreed, reset() raised a shape mismatch inside set_world_poses and Isaac's
    shutdown path turned it into a SIGSEGV - that is how the first A2 run died twice
    (2026-09-12): the manager reserved n_max*L_max = 24 slots while the env still computed
    n_pillars*pillar_layers = 16.

    Randomised layouts (A2/A3) reserve the UPPER bound, because the number of active slots
    varies per env/episode; unused slots are marked inactive.
    """
    K = int(oc.get("max_slots", 8))
    npr = oc.get("n_pillars_range", None)
    plr = oc.get("pillar_layers_range", None)
    pzr = oc.get("pillar_z_range", None)
    if int(oc.get("n_pillars", 0)) > 0 or any(x is not None for x in (npr, plr, pzr)):
        nP = int(npr[1]) if npr is not None else int(oc.get("n_pillars", 0))
        L = int(plr[1]) if plr is not None else max(1, int(oc.get("pillar_layers", 4)))
        return nP * L + int(oc.get("n_free_obstacles", 0))
    return int(max(int(oc.get("num_scene") or K), K))


class ObstacleManager:
    """Per-env static obstacle bookkeeping + layout sampling + geometry helpers.

    The manager owns full-batch buffers (all ``num_envs``):
        self.pos     (N, K, 3)  obstacle centers, env frame (inactive slots = 0)
        self.radius  (N, K)     logical radius per slot (inactive = 0)
        self.active  (N, K)     bool, slot index < current level active count
    """

    def __init__(self, cfg, num_envs, device="cuda:0"):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        # [M2 2026-09-05 obs-window] obs 窗口槽位 K（=max_slots, obs 维度 4K）与
        #   场景物理障碍数 M（=num_scene, 缓冲区/prim 数）解耦：M>K 时 build_obs 每步
        #   取最近的 K 个（滑动窗口）；M<=K（默认, num_scene 未设）保持固定槽旧语义。
        self.K = int(cfg.get("max_slots", 8))
        # [2026-09-08 pillar C] 柱组(方案A): 每柱 pillar_layers 个外接圆球沿 z 叠覆盖全高,
        #   柱球半径固定 pillar_radius(=0.5m 柱外接 0.354), 自由球从 radius_choices 抽。
        #   n_pillars=0 -> 原纯球语义(M 由 num_scene 定)。n_pillars>0 时 M = 柱层球 + 自由球。
        self.n_pillars = int(cfg.get("n_pillars", 0))
        self.pillar_layers = max(1, int(cfg.get("pillar_layers", 4)))
        self.pillar_radius = float(cfg.get("pillar_radius", 0.354))
        self.pillar_z_lo = float(cfg.get("pillar_z_lo", 0.4))
        self.pillar_z_hi = float(cfg.get("pillar_z_hi", 2.6))
        self.n_free_balls = int(cfg.get("n_free_obstacles", 0))
        self.pillar_ball_count = self.n_pillars * self.pillar_layers
        # [P1 A2 2026-09-12] parking spot for INACTIVE slots (see world_pose_tensor):
        #   ABOVE the arena (no ground-plane penetration) and spread in xy so parked
        #   bodies never coincide. Overridable, but these defaults are the safe ones -
        #   the old (0,0,-100) made PhysX corrupt its interaction registry and segfault.
        self.park_x = float(cfg.get("park_x", 100.0))
        self.park_dx = float(cfg.get("park_dx", 2.7))
        self.park_z = float(cfg.get("park_z", 20.0))
        self.radius_choices = [float(x) for x in cfg.get("radius_choices", [0.30])]
        if self.n_pillars > 0:
            self.M = self.pillar_ball_count + self.n_free_balls
        else:
            self.M = int(max(int(cfg.get("num_scene") or cfg.get("max_slots", 8)), self.K))
        _ow = cfg.get("obs_window")
        self.obs_window = bool(self.M > self.K) if _ow is None else bool(_ow)

        # ================= [P1 A2/A3 2026-09-12] 随机化柱布局（可选，默认全关） =================
        #   设计纪律：**新增独立路径**，uniform 路径（_sample_pillar_mixed）一行不改
        #   ⇒ A0/A/A1a/A1b 已冻结 profile 的采样分布逐位不变，单变量对比成立。
        #     n_pillars_range     [lo,hi] 每 env 柱数随机（A3；默认 null = 用 n_pillars）
        #     pillar_layers_range [lo,hi] 每柱层数随机（A2；默认 null = 用 pillar_layers）
        #     pillar_z_range      [[zlo_lo,zlo_hi],[zhi_lo,zhi_hi]] 每柱 z 跨度随机（A2）
        #     min_corridor        柱间最小**表面净宽**下界（A3；覆盖 min_gap_between 的约束）
        #     layout_tries        采样重试次数，超限则逐次放松 gap（默认 6）
        _npr = cfg.get("n_pillars_range", None)
        _plr = cfg.get("pillar_layers_range", None)
        _pzr = cfg.get("pillar_z_range", None)
        self.n_pillars_range = list(_npr) if _npr is not None else None
        self.pillar_layers_range = list(_plr) if _plr is not None else None
        self.pillar_z_range = ([list(v) for v in _pzr] if _pzr is not None else None)
        self.min_corridor = (float(cfg["min_corridor"])
                             if cfg.get("min_corridor", None) is not None else None)
        # [P1 A3 2026-09-14] Bottleneck-connectivity gate - IMPLEMENTED (was fail-fast).
        #   Semantics: W is NOT "a pillar pair must be >= W apart".  Eight pillars of
        #   r_o = 0.42 in a 6 m arena cannot satisfy that at all (it would need a centre
        #   spacing of 2r + W = 2.19 m).  W means: "there must exist a path from start to
        #   goal whose bottleneck clear width is >= W".  That is a property of the WHOLE
        #   layout, so it is a thresholded connectivity test, not a spacing floor:
        #   block every grid cell within `pillar_radius + W/2` of a pillar that overlaps
        #   the flight altitude band, then require start and goal to be in the same free
        #   component.
        #   Deliberately the SAME criterion as `scripts/pillar_layout_check.py
        #   --connectivity --corridor-clearance W`, so the CPU gate and the sampler can
        #   never disagree.  It is a conservative 2-D projection (a pillar that spans the
        #   band blocks horizontal passage even if a 3-D path might squeeze past), which is
        #   the safe direction to be wrong in.
        self.corridor_cell = float(cfg.get("corridor_cell", 0.05))
        self.corridor_extent = float(cfg.get("corridor_extent", 3.0))
        self.corridor_band_pad = float(cfg.get("corridor_band_pad", 0.15))
        # How many per-env redraws to attempt for the connectivity gate.  Higher than
        # layout_tries because connectivity is a per-env property whose failure rate can be
        # tens of percent (measured: 39.6% of A3's layouts at W=1.34), so a full batch needs
        # enough rounds for 0.4^k * num_envs to fall below 1 (~8 rounds at 1024 envs).
        self.corridor_tries = int(cfg.get("corridor_tries", 15))
        self.corridor_reject_events = 0
        self.corridor_reject_envs = 0
        self.layout_tries = int(cfg.get("layout_tries", 6))
        # [P1 A3 2026-09-14] Minimum number of ACTIVE slots per env.  Rationale: when
        #   M > K the obs window keeps the nearest K slots and `clearances()` masks unused
        #   slots to +inf.  An env that activates fewer than K slots therefore hands the
        #   policy FREE window entries (it sees padding where obstacles would be), and the
        #   dropped_relevant statistics stop meaning what they say.  A3 samples n_pillars
        #   and layers independently, so this is reachable there (2 pillars x 2 layers = 4
        #   < K = 8).  0 (default) = no constraint, which keeps the frozen A2/A2L2 worlds
        #   bit-identical.  See plan 3.2 / section 0.6.5.
        self.min_active_slots = int(cfg.get("min_active_slots", 0) or 0)
        self.pillar_random = any(x is not None for x in
                                 (self.n_pillars_range, self.pillar_layers_range,
                                  self.pillar_z_range))
        if self.min_corridor is not None and not self.pillar_random:
            # The corridor check needs the per-env drawn layout, so it only exists on the
            # randomized path.  Refuse loudly rather than silently sampling a world whose
            # connectivity was never verified - "train on an unverified world" is exactly
            # what the old fail-fast was protecting against.
            raise NotImplementedError(
                "obstacle.min_corridor is implemented only for the RANDOMIZED pillar path "
                "(set n_pillars_range / pillar_layers_range / pillar_z_range). The uniform "
                "path (_sample_pillar_mixed) has no connectivity gate; see plan 3.2.")
        self.layout_relax_events = 0
        if self.pillar_random:
            self.n_pillars_max = (int(self.n_pillars_range[1]) if self.n_pillars_range
                                  else self.n_pillars)
            self.pillar_layers_max = (int(self.pillar_layers_range[1])
                                      if self.pillar_layers_range else self.pillar_layers)
            if self.min_active_slots > self.n_pillars_max * self.pillar_layers_max:
                raise ValueError(
                    f"obstacle.min_active_slots={self.min_active_slots} is unreachable: "
                    f"n_pillars_max * pillar_layers_max = {self.n_pillars_max} * "
                    f"{self.pillar_layers_max} = "
                    f"{self.n_pillars_max * self.pillar_layers_max}. Raise a range bound "
                    f"or lower min_active_slots.")
            # 槽位 = n_max * L_max（每柱预留固定块），未用槽 active=False
            self.M = self.n_pillars_max * self.pillar_layers_max + self.n_free_balls
            if _ow is None:
                self.obs_window = bool(self.M > self.K)
            print(f"[ObstacleManager] randomized pillars ON: n_pillars<={self.n_pillars_max} "
                  f"layers<={self.pillar_layers_max} -> M={self.M} "
                  f"min_active_slots={self.min_active_slots} "
                  f"min_corridor={self.min_corridor} tries={self.layout_tries}")
        self.drone_radius = float(cfg.get("drone_radius", 0.15))
        self.inflation = float(cfg.get("inflation", 0.05))
        self.collision_margin = float(cfg.get("collision_margin", 0.05))
        self.danger_radius = float(cfg.get("danger_radius", 0.6))
        self.max_collisions = int(cfg.get("max_collisions", 2))
        self.init_clearance = float(cfg.get("init_clearance", 0.15))
        self.goal_clearance = float(cfg.get("goal_clearance", 0.35))
        self.min_gap_between = float(cfg.get("min_gap_between", 0.25))
        self.relax = float(cfg.get("spawn_relax", 0.85))
        self.relax_rounds = int(cfg.get("spawn_relax_rounds", 6))
        self.obs_dist_norm = float(cfg.get("obs_dist_norm", 5.0))
        # [P1 C 2026-09-14] per-pillar observation = one obs slot per OBSTACLE (a whole
        #   pillar, collapsed to its nearest layer) instead of one slot per LAYER.
        #   WHY: the window topk's over SLOTS = spheres = layers, so a single 6-layer
        #   pillar can eat 6 of the 8 window slots.  That is what pushed A2's
        #   dropped_relevant_step_frac to 18.86% and made the K budget look exhausted
        #   even though the world holds only 4-8 pillars.  Collapsing to one slot per
        #   pillar makes the obs requirement depend on the PILLAR count alone, so K=8
        #   covers A3's 2-8 pillars exactly, with no obs-dimension change (still 4*K) and
        #   no red line.  The CBF/physics/reward paths are untouched: the CBF ball channel
        #   is fed ALL M slots independently of this window (see nav_vel.py).
        #   Default False -> the frozen A0/A/A1a/A1b/A2/A2L2/A2L3 worlds are bit-identical.
        self.obs_per_pillar = bool(cfg.get("obs_per_pillar", False))
        self._pillar_id = None
        self._n_groups = 0
        self._obs_win_gid = None
        self.obs_radius_norm = float(cfg.get("obs_radius_norm", 0.5))

        # spawn box (env frame)
        xy = torch.as_tensor(cfg["spawn_xy_range"], dtype=torch.float32, device=self.device)
        zz = torch.as_tensor(cfg["spawn_z_range"], dtype=torch.float32, device=self.device)
        if xy.shape != (2, 2) or zz.shape != (2,):
            raise ValueError("obstacle.spawn_xy_range / spawn_z_range malformed")
        self.spawn_lo = torch.cat([xy[0], zz[:1]])
        self.spawn_hi = torch.cat([xy[1], zz[1:]])

        # [2026-09-08] 出生走廊净空(仅 x): 障碍逻辑球面不越过 |x|<=keepout_x (inf=不限,
        # 向后兼容). 球心随半径收窄 |x| <= keepout_x - r_o; 施加于 _sample_one_pass/_fallback.
        self.keepout_x = float(cfg.get("keepout_x", float("inf")))

        self.radius_max = max(self.radius_choices)
        self._tiers = torch.tensor(self.radius_choices, dtype=torch.float32, device=self.device)

        # full-batch buffers (M = 场景障碍数; M>=K)
        self.pos = torch.zeros(self.num_envs, self.M, 3, dtype=torch.float32, device=self.device)
        self.radius = torch.zeros(self.num_envs, self.M, dtype=torch.float32, device=self.device)
        self.active = torch.zeros(self.num_envs, self.M, dtype=torch.bool, device=self.device)
        # per-env episode-level running min clearance (m), for stats/eval
        self.ep_min_clearance = torch.full(
            (self.num_envs, 1), float("inf"), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ geometry
    @property
    def r_safe(self):
        """Decision radius per slot (inactive slots -> 0)."""
        return self.radius + self.drone_radius + self.inflation

    def clearances(self, drone_pos):
        """Surface clearance d_i = ||drone - p_oi|| - r_si for every slot.

        Args: drone_pos (N,1,3) env frame (full batch). Returns (N, K); inactive
        slots -> +inf.
        """
        center_dist = torch.norm(self.pos - drone_pos, dim=-1)          # (N,K)
        clr = center_dist - self.r_safe
        clr = torch.where(self.active, clr, torch.full_like(clr, float("inf")))
        return clr

    def clearances_for(self, env_ids, drone_pos):
        """Subset version used by mid-episode respawns: only the obstacles of the
        ``env_ids`` being respawned are considered against their (n,1,3) poses."""
        pos = self.pos[env_ids]
        act = self.active[env_ids]
        center_dist = torch.norm(pos - drone_pos, dim=-1)               # (n,K)
        r_s = self.radius[env_ids] + self.drone_radius + self.inflation
        clr = center_dist - r_s
        clr = torch.where(act, clr, torch.full_like(clr, float("inf")))
        return clr

    def min_clearance(self, drone_pos):
        """(N,1) min over active slots (inf when no active obstacle)."""
        clr = self.clearances(drone_pos)
        dmin = clr.min(dim=-1, keepdim=True).values
        return dmin

    # ---------------------------------------------------------------- obs block
    def pillar_counts(self):
        """(num_envs,) how many pillars currently have at least one ACTIVE slot.

        DIAGNOSTIC for the A3 randomization (`n_pillars_range=[2, 8]`).  Why it is worth
        reporting rather than policing: with route C (per-pillar obs) the number of
        *relevant* window entries equals this count, and the remaining `K - count` entries
        are genuinely empty space, not padding that misleads the policy.  So a low count is
        not an error - but if the whole distribution collapses onto one value the
        randomization has stopped varying difficulty, which IS an error.  Also the natural
        replacement for `min_active_slots >= K` in per-pillar mode, where that bound would
        mean "every env must have 8 pillars" and would destroy the range entirely.

        Returns a float tensor so it can be mean()/min()'d uniformly.
        """
        if not self.pillar_random:
            return torch.full((self.num_envs,), float(self.n_pillars),
                              dtype=torch.float32, device=self.device)
        L = self.pillar_layers_max
        nb = self.n_pillars_max * L
        return self.active[:, :nb].reshape(self.num_envs, self.n_pillars_max, L) \
            .any(dim=-1).sum(dim=-1).float()

    def _pillar_groups(self):
        """(M,) slot -> group id, or None unless obstacle.obs_per_pillar is on.

        A group is one obstacle AS A WHOLE: all the layers of one pillar, or a single free
        ball.  The mapping is static, so it is built once and cached.
        """
        if not self.obs_per_pillar or self.M <= self.K:
            return None
        if self._pillar_id is None:
            dev = self.device
            L = max(int(self.pillar_layers_max if self.pillar_random
                        else self.pillar_layers), 1)
            ids = torch.arange(self.M, device=dev) // L
            if self.pillar_random:
                # randomised path: slot p owns the fixed block [p*L_max, (p+1)*L_max)
                n_g = int(self.n_pillars_max)
            else:
                if self.n_free_balls > 0:
                    ids[self.pillar_ball_count:] = self.n_pillars + torch.arange(
                        self.M - self.pillar_ball_count, device=dev)
                n_g = int(self.n_pillars) + int(self.n_free_balls)
            self._pillar_id, self._n_groups = ids, n_g
            print(f"[ObstacleManager] obs_per_pillar ON: M={self.M} slots -> "
                  f"{n_g} groups (K={self.K})")
        return self._pillar_id

    def _window_per_pillar(self, d):
        """Nearest-K window over PILLARS instead of over slots (route C variant (a)).

        Steps: collapse each pillar to its nearest layer -> take the nearest K pillars ->
        report that representative layer's 4 features.  The feature meaning and the
        4*K dimension are unchanged; only *which* slot gets selected differs, so the
        policy stops wasting window slots on extra layers of a pillar it already sees.

        Sets self._obs_win_gid (the chosen group per window slot) so the eval-side
        dropped_relevant diagnostic can be computed on the same per-pillar basis.
        """
        gid, n_g = self._pillar_id, self._n_groups
        n = d.shape[0]
        d_g = torch.full((n, n_g), float("inf"), device=d.device)
        d_g.scatter_reduce_(1, gid.expand(n, -1), d, reduce="amin")
        if n_g < self.K:                      # pad so topk always yields K columns
            d_g = torch.cat([d_g, torch.full((n, self.K - n_g), float("inf"),
                                             device=d.device)], dim=-1)
        vals, idx_g = torch.topk(d_g, k=self.K, dim=-1, largest=False)
        self._obs_win_gid = idx_g
        # Map each chosen group back to the slot that realises its minimum.  Both masks
        # are needed: `belongs` keeps the slot in the chosen group (a padded column
        # matches nothing, so its `valid` stays False via the inf in `vals`), and
        # `realises` picks the nearest layer of that pillar.
        belongs = gid[None, None, :] == idx_g.unsqueeze(-1)      # (N,K,M)
        realises = d.unsqueeze(1) == vals.unsqueeze(-1)          # (N,K,M)
        idx = (belongs & realises).float().argmax(dim=-1)        # (N,K) representative
        return vals, idx

    def build_obs(self, drone_pos):
        """(N, 1, K*4) normalized obstacle block for the policy.

        固定槽模式（默认, M<=K）: 槽位 = 布局顺序的激活障碍, 未激活槽补零（与 M2 完全兼容）。
        滑动窗口模式（M>K, obs_window）: 每步从全部激活障碍中取距 drone 最近的 K 个并按
        距离升序填入 -> 场景可有远多于 K 个障碍而 obs 恒为 8 槽（62 维）。
        Per slot: (rpos_xyz / obs_dist_norm clamped [-1,1], radius / obs_radius_norm
        clamped [0,1]); invalid slots are exactly zero (mask). Pure geometry radius,
        NOT the CBF radius (kept out of obs on purpose).
        """
        if self.obs_window:
            # 选最近 K 个激活障碍（升序）；不足 K 个时剩余槽补零
            d = torch.norm(self.pos - drone_pos, dim=-1)              # (N,M)
            d = torch.where(self.active, d, torch.full_like(d, float("inf")))
            self._obs_win_gid = None
            if self._pillar_groups() is not None:
                # route C: one window slot per PILLAR (see _window_per_pillar)
                vals, idx = self._window_per_pillar(d)
            else:
                vals, idx = torch.topk(d, k=self.K, dim=-1, largest=False)  # (N,K)
            valid = torch.isfinite(vals)                              # (N,K)
            # [K5 2026-09-12] 暴露本步的窗口选择, 供 eval 统计 dropped_relevant
            #   (危险区障碍被挤出 obs 窗口 => 策略"看不见"却会被判撞). 纯诊断, 训练不使用.
            self._obs_win_idx = idx
            self._obs_win_valid = valid
            p_sel = self.pos.gather(
                1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3))  # (N,K,3)
            r_sel = self.radius.gather(1, idx.clamp(min=0))           # (N,K)
            dvec = p_sel - drone_pos                                  # (N,K,3) 障碍相对 drone
            block = torch.cat([
                (dvec / self.obs_dist_norm).clamp(-1.0, 1.0),         # (N,K,3)
                (r_sel / self.obs_radius_norm).clamp(0.0, 1.0).unsqueeze(-1),
            ], dim=-1)                                                # (N,K,4)
            block = block * valid.unsqueeze(-1)
        else:
            dvec = self.pos - drone_pos                               # (N,M,3), M=K
            block = torch.cat([
                (dvec / self.obs_dist_norm).clamp(-1.0, 1.0),         # (N,K,3)
                (self.radius / self.obs_radius_norm).clamp(0.0, 1.0).unsqueeze(-1),
            ], dim=-1)                                                # (N,K,4)
            block = block * self.active.unsqueeze(-1)
            # [K5] 固定槽模式: 窗口 = 全部激活槽, 不存在"被挤出"的障碍
            self._obs_win_idx = None
            self._obs_win_valid = None
        return block.reshape(self.num_envs, 1, self.K * 4)

    # ------------------------------------------------------------------- layout
    def sample_layout(self, init_pos, goal_pos, n_active):
        """Sample a fresh per-env obstacle layout (pure torch).

        Args:
            init_pos: (n,1,3) drone init positions (env frame).
            goal_pos: (n,1,3) target positions (env frame).
            n_active: number of obstacles to activate for ALL these envs (global level).
        Returns:
            pos (n,K,3), radius (n,K), active (n,K)  -- full K-slot tensors.
        """
        n = init_pos.shape[0]
        # [pillar C 2026-09-08] 混合布局: 固定 n_pillars 柱(每柱 pillar_layers 层外接球) +
        #   n_free_balls 个自由球, 全部激活(忽略 n_active/level)。obs 滑动窗口(M>K)保持 62 维。
        if self.pillar_random:
            # [P1 A2/A3 2026-09-12] 随机化路径（柱数/层数/z 跨度），见 __init__ 说明
            return self._sample_pillar_random(init_pos, goal_pos)
        if self.n_pillars > 0:
            return self._sample_pillar_mixed(init_pos, goal_pos)
        if n_active <= 0 or n == 0:
            return (torch.zeros(n, self.M, 3, device=self.device),
                    torch.zeros(n, self.M, device=self.device),
                    torch.zeros(n, self.M, dtype=torch.bool, device=self.device))
        L = int(n_active)
        init_pos = init_pos.reshape(n, 1, 3)
        goal_pos = goal_pos.reshape(n, 1, 3)

        fail_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        pos, rad = None, None
        for rnd in range(self.relax_rounds + 1):
            relax_f = self.relax ** rnd
            # keep init hard clearance >= drone+infl+0.02 even at max relaxation
            init_clr = max(self.init_clearance * relax_f, 0.02)
            goal_clr = max(self.goal_clearance * relax_f, 0.0)
            gap = max(self.min_gap_between * relax_f, 0.05)
            # only re-sample the still-failing subset
            sub = fail_mask.nonzero(as_tuple=False).squeeze(-1)
            if sub.numel() == 0:
                break
            p_s, r_s, ok_s = self._sample_one_pass(
                sub, init_pos[sub], goal_pos[sub], L, init_clr, goal_clr, gap)
            # merge into full result (active slots are the first L columns)
            if pos is None:
                pos = torch.zeros(n, self.M, 3, device=self.device)
                rad = torch.zeros(n, self.M, device=self.device)
            pos[sub, :L] = p_s
            rad[sub, :L] = r_s
            new_fail = torch.zeros(n, dtype=torch.bool, device=self.device)
            new_fail[sub] = ~ok_s
            fail_mask = new_fail
        if fail_mask.any():
            # last-resort fallback: place distinct points in-box keeping a minimal
            # init clearance so the drone never spawns inside a physical ball.
            sub = fail_mask.nonzero(as_tuple=False).squeeze(-1)
            pos[sub], rad[sub] = self._fallback_layout(sub, init_pos[sub], goal_pos[sub], L)

        active = torch.arange(self.M, device=self.device).unsqueeze(0).expand(n, -1) < L
        # inactive radius/pos already zero; enforce zero for safety
        rad = rad * active
        pos = pos * active.unsqueeze(-1)
        return pos, rad, active

    def _sample_one_pass(self, idx, init_pos, goal_pos, L, init_clr, goal_clr, gap):
        """Vectorized greedy: L rounds of candidate selection, all envs in parallel.

        Returns (pos (n,L,3), rad (n,L), ok (n,) bool).
        """
        n = idx.numel()
        rad = self._tiers[torch.randint(0, len(self._tiers), (n, L), device=self.device)]

        cand = self._candidate_pool(n)                                 # (n,C,3)
        d_init = torch.norm(cand - init_pos, dim=-1)                   # (n,C)
        d_goal = torch.norm(cand - goal_pos, dim=-1)
        r_s = rad + self.drone_radius + self.inflation                 # (n,L)
        init_min = r_s + init_clr                                      # (n,L)
        goal_min = r_s + goal_clr

        placed_pos = []                                                # each (n,3)
        placed_rad = []
        ok = torch.ones(n, dtype=torch.bool, device=self.device)
        for j in range(L):
            # endpoint margins for this slot's radius
            valid = (d_init >= init_min[:, j:j + 1]) & (d_goal >= goal_min[:, j:j + 1])
            # [2026-09-08 keepout_x] 球面不出出生走廊: 球心 |x| <= keepout_x - r_o (per slot);
            #   y/z 不受限 (口径: 逻辑球 r_o; NavVel.yaml keepout_x=2.5)
            x_lim = self.keepout_x - rad[:, j:j + 1]                    # (n,1)
            valid &= cand[..., 0].abs() <= x_lim
            for pk, pr in zip(placed_pos, placed_rad):
                need = (rad[:, j] + pr + gap)[:, None]                 # (n,1)
                valid &= torch.norm(cand - pk[:, None, :], dim=-1) >= need
            # choose a uniformly-random valid candidate per env
            score = torch.rand(n, cand.shape[1], device=self.device)
            score = torch.where(valid, score, torch.full_like(score, float("-inf")))
            best = score.argmax(dim=-1)                                # (n,)
            chosen = cand.gather(1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
            any_ok = valid.any(dim=-1)
            ok &= any_ok
            placed_pos.append(chosen)
            placed_rad.append(rad[:, j])
        pos = torch.stack(placed_pos, dim=1)                           # (n,L,3)
        return pos, rad, ok

    # ------------------------------------------------------------ [pillar C]
    # 混合布局: 柱 = 沿 z 叠 pillar_layers 个外接圆球(pillar_radius), 覆盖 [z_lo,z_hi] 全高;
    #   柱心在 xy 平面采样(净空 init/goal + 柱间 gap + keepout_x), 自由球 3D 避开柱层球。
    def _pillar_layer_zs(self, n):
        """(L,) z centers of one pillar's stacked balls covering [z_lo, z_hi]."""
        L = self.pillar_layers
        lo = self.pillar_z_lo + self.pillar_radius
        hi = max(lo, self.pillar_z_hi - self.pillar_radius)
        if L <= 1:
            return torch.tensor([(lo + hi) / 2], device=self.device)
        return torch.linspace(lo, hi, L, device=self.device)

    def _pillar_centers_pool(self, n):
        """(n, C, 2) candidate pillar centers (xy) in the usable box (keepout applied)."""
        xlo = self.spawn_lo[0] + 0.35
        xhi = self.spawn_hi[0] - 0.35
        if self.keepout_x < float("inf"):
            k = self.keepout_x - self.pillar_radius
            xlo = max(xlo, -k)
            xhi = min(xhi, k)
        ylo = self.spawn_lo[1] + 0.35
        yhi = self.spawn_hi[1] - 0.35
        xs = torch.arange(xlo, xhi + 1e-3, 0.35, device=self.device)
        ys = torch.arange(ylo, yhi + 1e-3, 0.35, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (C,2)
        jitter = (torch.rand(n, 1, 2, device=self.device) - 0.5) * 0.35
        pool = grid.unsqueeze(0) + jitter
        lo = torch.tensor([xlo, ylo], device=self.device)
        hi = torch.tensor([xhi, yhi], device=self.device)
        return pool.clamp(lo, hi)

    def _sample_pillar_mixed(self, init_pos, goal_pos):
        """Sample the fixed composition: n_pillars pillars + n_free_balls free balls.

        Returns (pos (n,M,3), rad (n,M), active (n,M)) with ALL M slots active.
        Slot layout: [:nP*L] = pillar stacked balls, [nP*L:] = free balls.
        """
        n = init_pos.shape[0]
        nP, L = self.n_pillars, self.pillar_layers
        zs = self._pillar_layer_zs(n)                                 # (L,)
        init_xy = init_pos.reshape(n, 1, 3)[..., :2]
        goal_xy = goal_pos.reshape(n, 1, 3)[..., :2]
        rP = self.pillar_radius + self.drone_radius + self.inflation  # 柱对无人机的有效(判定)半径

        ok_p = torch.ones(n, dtype=torch.bool, device=self.device)
        pillar_xy = torch.zeros(n, nP, 2, device=self.device)
        if nP > 0:
            pool2 = self._pillar_centers_pool(n)                      # (n,C,2)
            placed_xy = []
            for _ in range(nP):
                d_init = torch.norm(pool2 - init_xy, dim=-1)
                d_goal = torch.norm(pool2 - goal_xy, dim=-1)
                valid = (d_init >= rP + self.init_clearance) & \
                        (d_goal >= rP + self.goal_clearance)
                for pk in placed_xy:
                    need = 2 * self.pillar_radius + self.min_gap_between
                    valid &= torch.norm(pool2 - pk[:, None, :], dim=-1) >= need
                score = torch.where(
                    valid, torch.rand(n, pool2.shape[1], device=self.device),
                    torch.full((n, pool2.shape[1]), float("-inf"), device=self.device))
                best = score.argmax(dim=-1)
                chosen = pool2.gather(1, best[:, None, None].expand(-1, 1, 2)).squeeze(1)
                any_ok = valid.any(dim=-1)
                ok_p &= any_ok
                placed_xy.append(chosen)
            pillar_xy = torch.stack(placed_xy, dim=1)                 # (n,nP,2)

        # pillar stacked balls: (n, nP*L, 3)
        ppos = torch.zeros(n, nP, L, 3, device=self.device)
        ppos[..., 0] = pillar_xy[..., 0:1]
        ppos[..., 1] = pillar_xy[..., 1:2]
        ppos[..., 2] = zs.view(1, 1, L)
        pillar_balls = ppos.reshape(n, nP * L, 3)                     # (n, nP*L, 3)

        # free balls (n, Mf, 3)
        Mf = self.n_free_balls
        radF = torch.zeros(n, Mf, device=self.device)
        posF = torch.zeros(n, Mf, 3, device=self.device)
        ok_f = torch.ones(n, dtype=torch.bool, device=self.device)
        if Mf > 0:
            radF = self._tiers[torch.randint(0, len(self._tiers), (n, Mf), device=self.device)]
            pool3 = self._candidate_pool(n)                           # (n,C,3)
            init3 = init_pos.reshape(n, 1, 3)
            goal3 = goal_pos.reshape(n, 1, 3)
            d_init = torch.norm(pool3 - init3, dim=-1)
            d_goal = torch.norm(pool3 - goal3, dim=-1)
            # per-free-ball distance to all pillar balls
            d_pil = torch.norm(pool3[:, :, None, :] - pillar_balls[:, None, :, :], dim=-1)  # (n,C,nP*L)
            min_dp = d_pil.min(dim=-1).values if nP * L > 0 else None
            placed3 = []
            placed_r = []
            for j in range(Mf):
                r_s = radF[:, j] + self.drone_radius + self.inflation  # 端点判定(对无人机)
                valid = (d_init >= r_s[:, None] + self.init_clearance) & \
                        (d_goal >= r_s[:, None] + self.goal_clearance)
                if self.keepout_x < float("inf"):
                    valid &= pool3[..., 0].abs() <= (self.keepout_x - radF[:, j:j + 1])
                if min_dp is not None:
                    # 球-球净空: 中心距 >= radF_j + pillar_radius + gap
                    valid &= min_dp >= (radF[:, j][:, None] + self.pillar_radius + self.min_gap_between)
                for pk, pr in zip(placed3, placed_r):
                    need = radF[:, j] + pr + self.min_gap_between
                    valid &= torch.norm(pool3 - pk[:, None, :], dim=-1) >= need[:, None]
                score = torch.where(
                    valid, torch.rand(n, pool3.shape[1], device=self.device),
                    torch.full((n, pool3.shape[1]), float("-inf"), device=self.device))
                best = score.argmax(dim=-1)
                chosen = pool3.gather(1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
                any_ok = valid.any(dim=-1)
                ok_f &= any_ok
                placed3.append(chosen)
                placed_r.append(radF[:, j])
            posF = torch.stack(placed3, dim=1)

        if not (ok_p.all() and ok_f.all()):
            # 兜底: rejection 补失败 env
            pos, rad = self._fallback_pillar_mixed(init_pos, goal_pos)
        else:
            pos = torch.zeros(n, self.M, 3, device=self.device)
            rad = torch.zeros(n, self.M, device=self.device)
            pos[:, :nP * L] = pillar_balls
            rad[:, :nP * L] = self.pillar_radius
            pos[:, nP * L:] = posF
            rad[:, nP * L:] = radF
        active = torch.ones(n, self.M, dtype=torch.bool, device=self.device)
        return pos, rad, active

    # ------------------------------------------------- [P1 A2/A3] 随机化柱布局
    def _corridor_connected(self, pos, rad, act, init_pos, goal_pos):
        """Per-env bottleneck-connectivity test for the A3 corridor gate (plan 3.2).

        `min_corridor = W` means "there must exist a start -> goal path whose bottleneck
        clear width is >= W".  Implemented as: rasterise the flight band onto a 2-D grid,
        block every cell within `pillar_radius + W/2` of a sphere that overlaps the band,
        then require start and goal to lie in the same free connected component.

        Same criterion as `scripts/pillar_layout_check.py --connectivity
        --corridor-clearance W` (cell 0.05, extent [-3, 3], band pad 0.15) ON PURPOSE, so
        the CPU gate and the sampler cannot disagree about what W means.

        Returns a bool tensor (n,): True where a W-wide corridor exists.
        """
        import numpy as _np
        try:
            from scipy import ndimage as _ndi
        except Exception:                       # pragma: no cover - scipy is a hard dep
            raise RuntimeError("min_corridor needs scipy.ndimage for the connectivity test")

        n = pos.shape[0]
        W = float(self.min_corridor)
        cell = self.corridor_cell
        ext = self.corridor_extent
        nc = int(round((2.0 * ext) / cell))

        init = init_pos.reshape(n, 3)
        goal = goal_pos.reshape(n, 3)
        # Flight band: the drone only needs a horizontal corridor between the start and
        # goal ALTITUDES (+ pad).  A sphere that never enters the band cannot block it -
        # the drone flies under/over it.  This is what makes the gate 3-D-aware.
        b_lo = torch.minimum(init[:, 2], goal[:, 2]) - self.corridor_band_pad
        b_hi = torch.maximum(init[:, 2], goal[:, 2]) + self.corridor_band_pad

        # Offsets of the grid-cell centres, centred on the ORIGIN, so the same grid works
        # for every env (pillar centres are already expressed in the env frame).
        half = 0.5 * (nc - 1) * cell
        ax = (torch.arange(nc, device=pos.device, dtype=pos.dtype) * cell) - half

        # Which slots overlap the band and are active?  (n, M) -> only those can block.
        zc = pos[..., 2]                                    # (n,M)
        in_band = act & (zc + rad > b_lo[:, None]) & (zc - rad < b_hi[:, None])

        thr = self.pillar_radius + 0.5 * W
        thr2 = thr * thr

        out = torch.zeros(n, dtype=torch.bool, device=pos.device)
        for e in range(n):
            sel = torch.nonzero(in_band[e], as_tuple=False).reshape(-1)
            blocked = _np.zeros((nc, nc), dtype=bool)
            if sel.numel() > 0:
                cx = pos[e, sel, 0]
                cy = pos[e, sel, 1]
                # Vectorised: (S, nc, nc) distance^2, reduced with OR over slots.
                dx = ax[None, None, :] - cx[:, None, None]
                dy = ax[None, :, None] - cy[:, None, None]
                blocked = (dx * dx + dy * dy < thr2).any(dim=0).cpu().numpy()

            def _rc(p):
                i = int(round((float(p[1]) + half) / cell))
                j = int(round((float(p[0]) + half) / cell))
                return (0 <= i < nc and 0 <= j < nc), i, j

            si_ok, si, sj = _rc(init[e])
            gi_ok, gi, gj = _rc(goal[e])
            if not (si_ok and gi_ok):
                continue                        # start/goal outside the grid: cannot certify
            if blocked[si, sj] or blocked[gi, gj]:
                continue                        # start or goal itself is inside a blocked cell
            # Label the FREE space, not the blocked space.  scipy.ndimage.label marks the
            # True cells as components and everything False as background 0, so feeding it
            # `blocked` makes every free cell (start and goal included) label 0 and the
            # test would answer False for EVERY layout - which is exactly the bug this
            # comment is here to prevent: an always-rejecting gate looks like a very
            # strict gate rather than a broken one.
            lab, _ = _ndi.label(~blocked, structure=_np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
            out[e] = bool(lab[si, sj] != 0 and lab[si, sj] == lab[gi, gj])
        return out

    def _sample_pillar_random(self, init_pos, goal_pos):
        """Randomised pillar layout + the A3 bottleneck-connectivity gate.

        WHY A WRAPPER.  Inactive/geometric constraints (spawn & goal clearance, pillar
        spacing) are satisfied by almost every draw, so `_sample_pillar_random_once`'s
        "retry until ok.all()" loop works for them.  Connectivity is NOT like that: its
        per-env failure rate is tens of percent (measured 39.6% for A3 at W=1.34), so
        demanding that all 1024 envs succeed in the SAME attempt is hopeless
        (0.60^1024 ~ 0) and the loop would always exhaust its tries and return
        disconnected worlds.  Rejection therefore has to be PER-ENV and STICKY: freeze the
        envs that already have a corridor and redraw only the ones that do not.

        When `min_corridor is None` this calls `_sample_pillar_random_once` exactly once and
        returns, so the frozen profiles (A0/A/A1a/A1b/A2/A2L2/A2L3/A2P) consume the RNG in
        exactly the same order as before - verified bit-exact against git HEAD.
        """
        pos, rad, act = self._sample_pillar_random_once(init_pos, goal_pos)
        if self.min_corridor is None:
            return pos, rad, act

        conn = self._corridor_connected(pos, rad, act, init_pos, goal_pos)
        for _ in range(max(self.corridor_tries, 0)):
            bad = ~conn
            if not bool(bad.any()):
                break
            self.corridor_reject_events += 1
            self.corridor_reject_envs += int(bad.sum())
            pos[bad], rad[bad], act[bad] = self._sample_pillar_random_once(
                init_pos[bad], goal_pos[bad])
            conn[bad] = self._corridor_connected(
                pos[bad], rad[bad], act[bad], init_pos[bad], goal_pos[bad])
        if not bool(conn.all()):
            print(f"[ObstacleManager] WARN corridor gate: {int((~conn).sum())}/"
                  f"{pos.shape[0]} envs still have no W={self.min_corridor} m corridor "
                  f"after {self.corridor_tries} redraws; these envs need the CPU gate "
                  f"(scripts/pillar_layout_check.py --connectivity) to be checked")
        return pos, rad, act

    def _sample_pillar_random_once(self, init_pos, goal_pos):
        """Randomised pillar layout: per-env pillar count / per-pillar layer count and
        z-span (A2), optional pillar-to-pillar corridor lower bound and spawn/goal
        clearance relaxation with retries (A3).

        Slot numbering is **fixed-block**: pillar p always owns `[p*L_max, (p+1)*L_max)`,
        so an env with fewer pillars/layers simply leaves the tail slots inactive
        (`active=False`). `clearances()` masks inactive slots to +inf and `build_obs`
        picks the nearest K among the finite ones, so the existing obs/window machinery
        needs no change.

        Returns (pos (n,M,3), rad (n,M), active (n,M)). The uniform path
        (`_sample_pillar_mixed`) is untouched: this method is only reachable when one of
        `n_pillars_range` / `pillar_layers_range` / `pillar_z_range` is set.
        """
        n = init_pos.shape[0]
        dev = self.device
        nP_max, L_max = self.n_pillars_max, self.pillar_layers_max
        if self.n_free_balls > 0:
            raise NotImplementedError(
                "randomized pillar layout with free balls is not implemented "
                "(A2/A3 profiles keep n_free_obstacles=0)")
        r = self.pillar_radius
        rP = r + self.drone_radius + self.inflation
        init_xy = init_pos.reshape(n, 1, 3)[..., :2]
        goal_xy = goal_pos.reshape(n, 1, 3)[..., :2]
        pool2 = self._pillar_centers_pool(n)                          # (n,C,2)
        k_lay = torch.arange(L_max, device=dev).float()[None, None, :]  # (1,1,L)
        k_idx = torch.arange(L_max, device=dev)[None, :]                # (1,L)

        np_lo, np_hi = (self.n_pillars_range if self.n_pillars_range
                        else [self.n_pillars, self.n_pillars])
        lay_lo, lay_hi = (self.pillar_layers_range if self.pillar_layers_range
                          else [self.pillar_layers, self.pillar_layers])
        if self.pillar_z_range:
            (zl0, zl1), (zh0, zh1) = self.pillar_z_range
        else:
            (zl0, zl1), (zh0, zh1) = ((self.pillar_z_lo, self.pillar_z_lo),
                                      (self.pillar_z_hi, self.pillar_z_hi))

        last = None
        for attempt in range(max(self.layout_tries, 1)):
            relax = 0.9 ** attempt            # 0.9^0 = 1.0 -> 首次完全不放松
            # NOTE: min_corridor is NOT a spacing floor applied here.  It is a bottleneck
            # connectivity property of the WHOLE layout, so it is tested after the layout
            # is built, near the end of this loop (`_corridor_connected`), and folded into
            # the same `ok` mask so `layout_tries` retries it.  Using it as a pairwise
            # pillar-to-pillar gap here would be wrong: 8 pillars of r_o=0.42 in a 6 m
            # arena would need a 2.19 m centre spacing and no layout could ever satisfy it.
            need_gap = 2.0 * r + self.min_gap_between
            v_init = self.init_clearance * relax
            v_goal = self.goal_clearance * relax

            nP_e = torch.randint(int(np_lo), int(np_hi) + 1, (n,), device=dev).clamp(min=1)
            L_e = torch.randint(int(lay_lo), int(lay_hi) + 1, (n, nP_max), device=dev)
            if self.min_active_slots > 0:
                # [P1 A3 2026-09-14] Satisfy the bound by CONSTRUCTION, not by rejection.
                #   Rejecting through `ok` (the first thing I tried) does not work: the retry
                #   loop re-draws from the SAME distribution, so an env that keeps drawing
                #   2 pillars x 2 layers stays short forever, the loop exhausts layout_tries
                #   and then returns the invalid layout with only a WARN.  Measured:
                #   min_active_slots=8 with n_pillars_range=[2,8]/L_max=2 left 74/256 envs
                #   unsatisfied and the realised minimum was 4.  Instead, promote any env
                #   below the bound to the full allocation.  __init__ guarantees
                #   nP_max * L_max >= min_active_slots, so this always succeeds.
                k_p = torch.arange(nP_max, device=dev)[None, :]                 # (1,P)
                total = (L_e * (k_p < nP_e[:, None])).sum(dim=-1)               # (n,)
                short = total < self.min_active_slots
                if bool(short.any()):
                    nP_e = torch.where(short, torch.full_like(nP_e, nP_max), nP_e)
                    L_e = torch.where(short[:, None], torch.full_like(L_e, L_max), L_e)
            z_lo = torch.empty(n, nP_max, device=dev).uniform_(float(zl0), float(zl1))
            z_hi = torch.empty(n, nP_max, device=dev).uniform_(float(zh0), float(zh1))
            z_hi = torch.maximum(z_hi, z_lo + 2.0 * r)      # 至少容得下 1 层

            # --- greedy xy placement (same scheme as the uniform path) ---
            chosen = torch.zeros(n, nP_max, 2, device=dev)
            ok = torch.ones(n, dtype=torch.bool, device=dev)
            d_init = (pool2 - init_xy).norm(dim=-1)
            d_goal = (pool2 - goal_xy).norm(dim=-1)
            base = (d_init >= rP + v_init) & (d_goal >= rP + v_goal)
            for p in range(nP_max):
                valid = base.clone()
                for q in range(p):
                    valid &= (pool2 - chosen[:, q][:, None, :]).norm(dim=-1) >= need_gap
                score = torch.where(
                    valid, torch.rand(n, pool2.shape[1], device=dev),
                    torch.full((n, pool2.shape[1]), float("-inf"), device=dev))
                b = score.argmax(dim=-1)
                chosen[:, p] = pool2.gather(
                    1, b[:, None, None].expand(-1, 1, 2)).squeeze(1)
                ok &= valid.any(dim=-1)

            # --- fill fixed blocks; tail slots stay inactive ---
            pos = torch.zeros(n, self.M, 3, device=dev)
            rad = torch.zeros(n, self.M, device=dev)
            act = torch.zeros(n, self.M, dtype=torch.bool, device=dev)
            span = torch.clamp(z_hi - z_lo - 2.0 * r, min=0.0)                  # (n,P)
            denom = (L_e - 1).clamp(min=1).float()[..., None]                   # (n,P,1)
            frac = torch.where(L_e[..., None] > 1, k_lay / denom,
                               torch.full_like(L_e[..., None], 0.5))            # (n,P,L)
            zc = (z_lo + r)[..., None] + span[..., None] * frac                 # (n,P,L)
            for p in range(nP_max):
                sl = slice(p * L_max, (p + 1) * L_max)
                m = (k_idx < L_e[:, p:p + 1]) & (nP_e[:, None] > p)             # (n,L)
                pos[:, sl, 0] = torch.where(m, chosen[:, p, 0:1], pos[:, sl, 0])
                pos[:, sl, 1] = torch.where(m, chosen[:, p, 1:2], pos[:, sl, 1])
                pos[:, sl, 2] = torch.where(m, zc[:, p, :], pos[:, sl, 2])
                rad[:, sl] = torch.where(m, torch.full((n, L_max), r, device=dev),
                                         rad[:, sl])
                act[:, sl] = m

            # Guard rail for the by-construction promotion above: unreachable, but if it
            # ever fires the layout would silently hand the policy free obs-window slots,
            # so fail loudly instead.
            if self.min_active_slots > 0:
                if int(act.sum(dim=-1).min()) < self.min_active_slots:
                    raise RuntimeError(
                        f"min_active_slots={self.min_active_slots} not satisfied: "
                        f"minimum realised active slots = {int(act.sum(dim=-1).min())}")

            # NOTE: the A3 corridor gate is NOT applied here - see the wrapper
            # `_sample_pillar_random`, which enforces it per-env after this returns.
            if bool(ok.all()):
                return pos, rad, act
            self.layout_relax_events += 1
            last = (pos, rad, act, ok)
        print(f"[ObstacleManager] WARN randomized layout: "
              f"{int((~last[3]).sum())}/{n} envs unsatisfied after "
              f"{self.layout_tries} tries (constraints already relaxed)")
        return last[0], last[1], last[2]

    def _fallback_pillar_mixed(self, init_pos, goal_pos):
        """Rejection fallback guaranteeing a valid composition (looser gaps)."""
        n = init_pos.shape[0]
        nP, L = self.n_pillars, self.pillar_layers
        zs = self._pillar_layer_zs(n)
        init3 = init_pos.reshape(n, 1, 3)
        goal3 = goal_pos.reshape(n, 1, 3)
        rP = self.pillar_radius + self.drone_radius + self.inflation
        x_hi = self.spawn_hi[0] - 0.2
        x_lo = self.spawn_lo[0] + 0.2
        if self.keepout_x < float("inf"):
            k = self.keepout_x - self.pillar_radius
            x_hi = min(x_hi, k)
            x_lo = max(x_lo, -k)
        y_lo = self.spawn_lo[1] + 0.2
        y_hi = self.spawn_hi[1] - 0.2
        px = torch.zeros(n, nP, device=self.device)
        py = torch.zeros(n, nP, device=self.device)
        for i in range(nP):
            for _ in range(60):
                cx = x_lo + (x_hi - x_lo) * torch.rand(n, device=self.device)
                cy = y_lo + (y_hi - y_lo) * torch.rand(n, device=self.device)
                ok = torch.ones(n, dtype=torch.bool, device=self.device)
                if i > 0:
                    dx = cx - px[:, :i]
                    dy = cy - py[:, :i]
                    ok &= (dx * dx + dy * dy).min(dim=-1).values >= (2 * self.pillar_radius) ** 2
                ok &= torch.sqrt((cx - init3[..., 0, 0]) ** 2 + (cy - init3[..., 0, 1]) ** 2) >= rP + self.init_clearance
                ok &= torch.sqrt((cx - goal3[..., 0, 0]) ** 2 + (cy - goal3[..., 0, 1]) ** 2) >= rP + self.goal_clearance
                if ok.all():
                    break
            px[:, i] = cx
            py[:, i] = cy
        ppos = torch.zeros(n, nP, L, 3, device=self.device)
        ppos[..., 0] = px.unsqueeze(-1)
        ppos[..., 1] = py.unsqueeze(-1)
        ppos[..., 2] = zs.view(1, 1, L)
        pillar_balls = ppos.reshape(n, nP * L, 3)

        Mf = self.n_free_balls
        radF = self._tiers[torch.randint(0, len(self._tiers), (n, Mf), device=self.device)]
        posF = torch.zeros(n, Mf, 3, device=self.device)
        for j in range(Mf):
            for _ in range(200):
                c = (self.spawn_lo + 0.2) + (self.spawn_hi - self.spawn_lo - 0.4) * torch.rand(n, 1, 3, device=self.device)
                if self.keepout_x < float("inf"):
                    lim = (self.keepout_x - radF[:, j:j + 1]).clamp(min=0.0)
                    c[..., 0] = c[..., 0].clamp(-lim, lim)
                d_pil = torch.norm(c - pillar_balls, dim=-1).min(dim=-1).values
                ok = torch.ones(n, dtype=torch.bool, device=self.device)
                ok &= d_pil >= (radF[:, j] + self.pillar_radius + self.min_gap_between * 0.5)
                ok &= torch.norm(c - init3, dim=-1).squeeze(-1) >= radF[:, j] + self.drone_radius + self.inflation + 0.02
                ok &= torch.norm(c - goal3, dim=-1).squeeze(-1) >= radF[:, j] + self.goal_clearance * 0.5
                if j > 0:
                    dd = c - posF[:, :j]
                    need = radF[:, j] + radF[:, :j] + self.min_gap_between * 0.5
                    ok &= torch.norm(dd, dim=-1).min(dim=-1).values >= need.min(dim=-1).values
                if ok.all():
                    break
            posF[:, j:j + 1] = c
        pos = torch.zeros(n, self.M, 3, device=self.device)
        rad = torch.zeros(n, self.M, device=self.device)
        pos[:, :nP * L] = pillar_balls
        rad[:, :nP * L] = self.pillar_radius
        pos[:, nP * L:] = posF
        rad[:, nP * L:] = radF
        return pos, rad

    def _candidate_pool(self, n):
        """Jittered grid of candidate centers inside the spawn box (n, C, 3)."""
        # grid spacing ~0.35 gives dense-but-tractable pool
        xs = torch.arange(self.spawn_lo[0], self.spawn_hi[0] + 1e-3, 0.35, device=self.device)
        ys = torch.arange(self.spawn_lo[1], self.spawn_hi[1] + 1e-3, 0.35, device=self.device)
        zs = torch.arange(self.spawn_lo[2], self.spawn_hi[2] + 1e-3, 0.35, device=self.device)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (C,3)
        C = grid.shape[0]
        # per-env jitter within one cell (breaks identical grids across envs)
        jitter = (torch.rand(n, 1, 3, device=self.device) - 0.5) * 0.35
        pool = grid.unsqueeze(0) + jitter
        pool = pool.clamp(self.spawn_lo + 0.02, self.spawn_hi - 0.02)
        return pool.expand(n, C, 3)

    def _fallback_layout(self, idx, init_pos, goal_pos, L):
        """Guarantee L distinct in-box positions without overlapping the drone spawn."""
        n = idx.numel()
        rad = self._tiers[torch.randint(0, len(self._tiers), (n, L), device=self.device)]
        hard = (rad + self.drone_radius + self.inflation + 0.02).unsqueeze(-1)  # (n,L,1)
        pos = torch.zeros(n, L, 3, device=self.device)
        # sequential offset fallback: place along a per-env diagonal inside the box
        # until outside the init hard ball
        base = self.spawn_lo + 0.3
        span = self.spawn_hi - self.spawn_lo - 0.6
        for j in range(L):
            # [2026-09-08 keepout_x] fallback 同样保证球面不出走廊(球心 clamp 到 ±(keepout-r_o))
            x_lim = (self.keepout_x - rad[:, j:j + 1]).clamp(min=0.0)   # (n,1)
            cand = base + span * torch.rand(n, 1, 3, device=self.device)
            cand[..., 0] = cand[..., 0].clamp(-x_lim, x_lim)
            # bump candidates that are too close to init
            d = torch.norm(cand - init_pos.reshape(n, 1, 3), dim=-1)
            too_close = d < hard[:, j]
            for _ in range(8):
                repl = base + span * torch.rand(n, 1, 3, device=self.device)
                repl[..., 0] = repl[..., 0].clamp(-x_lim, x_lim)
                cand = torch.where(too_close, repl, cand)
                d = torch.norm(cand - init_pos.reshape(n, 1, 3), dim=-1)
                too_close = d < hard[:, j]
                if not too_close.any():
                    break
            pos[:, j:j + 1] = cand
        return pos, rad

    # --------------------------------------------------------------- bookkeeping
    def commit_layout(self, env_ids, pos, radius, active):
        """Write a sampled layout into the full-batch buffers for ``env_ids``."""
        self.pos[env_ids] = pos
        self.radius[env_ids] = radius
        self.active[env_ids] = active
        self.ep_min_clearance[env_ids] = float("inf")

    def update_min_clearance(self, drone_pos):
        """Rolling per-env episode minimum clearance (N,1) used by stats / eval."""
        clr = self.clearances(drone_pos)
        dmin = clr.min(dim=-1, keepdim=True).values
        self.ep_min_clearance = torch.minimum(self.ep_min_clearance, dmin)

    def world_pose_tensor(self, env_ids, envs_positions):
        """(n, M, 3) world-frame positions for the view write at reset.

        Active slots -> env-frame pos + env offset. Inactive slots are parked far above
        the arena, spread out in xy so they never coincide.

        [P1 A2 2026-09-12] Two separate things, recorded because I got them mixed up:

        1. The old spot (0, 0, -100) put the bodies 100 m *inside* the infinite default
           ground plane, which is nonsense even for kinematic bodies. Parking above the
           arena (park_z = 20, well clear of the drone's z <= 3 envelope and of every
           collision test) removes that.
        2. **It was NOT the cause of the A2 segfaults**, although it looked like it: the
           crashes came with the PhysX "Unexpectedly unregistered an interaction that
           does not have a valid interaction ID." error, and the only worlds that park
           anything are the ones with M > active count, i.e. exactly the crashing ones.
           A 2x2 bisect disproved it - `thin6` (M=24, 4 pillars x 6 layers, all 24 slots
           active, ZERO parked) and `p6_L4` (M=24, 6 x 4, zero parked) both crashed,
           while `deep16` (M=16, 4 coincident layers, zero parked) was stable. Parking
           was a red herring that merely co-occurred with large M; the actual cause was
           the PhysX GPU contact/patch pool overflowing (see cfg/base/sim_base.yaml and
           the commit that fixed it). Verified after that fix: profiles/A2 (M=24, with
           parked slots) trains with zero physx errors.
        """
        pos = self.pos[env_ids].clone()
        park = (~self.active[env_ids]).unsqueeze(-1)                     # (n,M,1)
        idx = torch.arange(self.M, device=self.device).float()
        park_xyz = torch.stack([self.park_x + self.park_dx * idx,
                                torch.zeros_like(idx),
                                torch.full_like(idx, self.park_z)], dim=-1)   # (M,3)
        pos = torch.where(park, park_xyz.unsqueeze(0), pos)
        pos = pos + envs_positions[env_ids].unsqueeze(1)
        return pos
