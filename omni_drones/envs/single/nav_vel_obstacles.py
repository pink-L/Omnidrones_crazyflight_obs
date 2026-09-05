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
        self.M = int(max(int(cfg.get("num_scene") or cfg.get("max_slots", 8)), self.K))
        _ow = cfg.get("obs_window")
        self.obs_window = bool(self.M > self.K) if _ow is None else bool(_ow)
        self.radius_choices = [float(x) for x in cfg.get("radius_choices", [0.30])]
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
        self.obs_radius_norm = float(cfg.get("obs_radius_norm", 0.5))

        # spawn box (env frame)
        xy = torch.as_tensor(cfg["spawn_xy_range"], dtype=torch.float32, device=self.device)
        zz = torch.as_tensor(cfg["spawn_z_range"], dtype=torch.float32, device=self.device)
        if xy.shape != (2, 2) or zz.shape != (2,):
            raise ValueError("obstacle.spawn_xy_range / spawn_z_range malformed")
        self.spawn_lo = torch.cat([xy[0], zz[:1]])
        self.spawn_hi = torch.cat([xy[1], zz[1:]])

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
            vals, idx = torch.topk(d, k=self.K, dim=-1, largest=False)  # (N,K)
            valid = torch.isfinite(vals)                              # (N,K)
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
            cand = base + span * torch.rand(n, 1, 3, device=self.device)
            # bump candidates that are too close to init
            d = torch.norm(cand - init_pos.reshape(n, 1, 3), dim=-1)
            too_close = d < hard[:, j]
            for _ in range(8):
                repl = base + span * torch.rand(n, 1, 3, device=self.device)
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
        """(n, K, 3) world-frame positions for the view write at reset.

        Active slots -> env-frame pos + env offset. Inactive slots are parked far
        below the ground (z=-100) so they never collide or occlude.
        """
        pos = self.pos[env_ids].clone()
        park = (~self.active[env_ids]).unsqueeze(-1)
        pos = torch.where(park, torch.tensor([0.0, 0.0, -100.0], device=self.device), pos)
        pos = pos + envs_positions[env_ids].unsqueeze(1)
        return pos
