# One-off deterministic evaluation for a saved policy checkpoint (headless).
# Usage (from OmniDrones/scripts):
#   python eval_ckpt.py task=HoverCrazyflie algo=ppo headless=true wandb.mode=disabled \
#       +checkpoint=/path/to/checkpoint_XXXX.pt rollout_steps=400
#
# ===================== 标准验收模板（默认严格单命口径, 2026-09-07） =====================
# 目标: 一次飞行 = 一次考核（坠毁/出界/碰撞超限 ⇒ 该 env 判 terminated 且不复活）。
# [env_design 2026-09-07] 6×6×3 NavRL 式迁移后: 训练 yaml 默认已是严格单命
#   (soft_respawn=false + obstacle.max_collisions=1 = 坠地/OOB/首碰都硬终止, 与验收同构),
#   故 eval 不必再覆盖 soft_respawn(保留显式给以自文档化)。eval 用固定起终点(训练用
#   episode_sampler=edge 随机对侧), 环境侧与训练同构, 只需配置覆盖、无需改代码:
#
#   python eval_ckpt.py task=NavVel algo=ppo headless=true wandb.mode=disabled \
#       task.soft_respawn=false task.obstacle.max_collisions=1 \     # 严格单命(显式自文档化)
#       task.fixed_init=[-2.8,0.0,0.5] task.fixed_target=[2.8,0.0,1.0] \  # eval 固定点(6×6×3 对侧)
#       task.reward_scheme=f1 task.reward_fly_weight=6.0 \
#       task.arrive_bonus=30 task.arrive_time_bonus=30 task.reward_timeout_penalty=40 \
#       task.reward_action_smoothness_weight=0.2 \
#       task.obstacle.num_scene=16 'task.obstacle.spawn_xy_range=[[-3.0,-3.0],[3.0,3.0]]' \
#       'task.curriculum.levels=[16]' task.curriculum.enabled=false \
#       task.cbf.mode=hybrid task.cbf.use_brake_term=false task.cbf.penalty_src=dual \
#       task.cbf.reward_weight=0.1 task.cbf.correction_weight=0.1 task.cbf.correction_sigma=0.5 \
#       +checkpoint=<ckpt> +rollout_steps=600 +runtime_filter=true      # ON（带 CBF filter 部署）
#   # OFF（internalize/撤 filter 判据）换成 +runtime_filter=false；低密度档换 num_scene=8
#   # + levels=[8]。obs_safety 键须与训练一致（无则不加）。障碍/密度键须与训练 ckpt 一致。
#
# 指标口径（eval_ckpt 直接读 env 内部计数器, 单窗口 600 步无 reset）:
#   arr    = 窗口内曾 ‖rpos‖<arrive_radius(0.5m) 连续保持 ≥arrive_hold_steps(50步) 的 env 比例
#   coll   = 窗口内曾发生几何接触边沿(new_edge=dmin<collision_margin(0.05) 的进入瞬间)的 env 比例
#   joint  = arr ∧ 整窗 0 碰撞边沿 的 env 比例（同课程 success 定义）
# 注: 训练配置里 reward 键（arrive_bonus 等）不影响判定, 仅为保持 obs/几何与训练同构而带上。
# ==============================================================================
import collections
import logging
import math
import hydra
import torch

from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


class _PerturbWrapper(torch.nn.Module):
    """Wrap a policy to inject command-space perturbation on the raw velocity action
    BEFORE the CBF velocity filter runs (Compose applies the appended cbf filter first
    into the env). This tests whether the CBF filter absorbs unsafe commands under
    uncertainty (CBF-RL robustness claim): modes
      cmd_gauss  = per-step Gaussian noise on the 3D velocity command,
      cmd_pulse  = intermittent gust pulses (random direction, a few steps).
    """
    def __init__(self, policy, mode, strength):
        super().__init__()
        self.policy = policy
        self.mode = mode
        self.strength = float(strength)
        self._t = 0
        self._pulse_until = 0
        self._pulse = None

    @torch.no_grad()
    def forward(self, tensordict):
        tensordict = self.policy(tensordict)
        a = tensordict[("agents", "action")]
        if self.mode == "cmd_gauss":
            a[..., :3] = a[..., :3] + torch.randn_like(a[..., :3]) * self.strength
        elif self.mode == "cmd_pulse":
            if self._t >= self._pulse_until:
                dur = int(torch.randint(5, 9, (1,)).item())
                self._pulse_until = self._t + dur
                d = torch.randn(1, 1, 3)
                d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                self._pulse = d * self.strength
            if self._t < self._pulse_until and self._pulse is not None:
                a[..., :3] = a[..., :3] + self._pulse.to(a.device)
        self._t += 1
        tensordict[("agents", "action")] = a
        return tensordict


@hydra.main(config_path=".", config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        OmegaConf.set_struct(cfg.task, False)
    except Exception:
        pass
    # [K5 2026-09-12] 可选确定性种子。plan §4.3 的 filter 依赖度要求 ON/OFF 两次评估
    #   在**同一套障碍布局 + 同一套 DR 采样**下比较（"同 seed、同分布"）。此处显式播种,
    #   并在报告里输出布局指纹(layout_fp)以**验证**两次跑确实同分布。
    _sd = cfg.get("set_seed", None)
    if _sd is not None:
        import random as _rand
        _sd = int(_sd)
        torch.manual_seed(_sd)
        _rand.seed(_sd)
        try:
            import numpy as _np
            _np.random.seed(_sd)
        except Exception:
            pass
        print(f"[eval_ckpt] set_seed={_sd} -> deterministic layout/DR (ON vs OFF comparable)")
    # [2026-09-08] eval_points=fixed|train
    #   fixed (默认) = 固定起终点(single-point acceptance, CLI 传 fixed_init/fixed_target)
    #   train = 训练同分布(episode_sampler=edge 随机对侧起终点, 不设 fixed, 每 env 一个随机起点)
    #           -> 验证"任意起终点泛化" policy (用户目标: 策略应应对任意 start/goal)。
    #   确定性/探索由 set_exploration_type(MODE=mean) 决定; filter ON/OFF 由 +runtime_filter 决定
    #   (后期 dual/filter_only 的 internalize 对照即用 fixed|train × ON|OFF)。
    _ep = str(cfg.get("eval_points", "fixed"))
    if _ep == "train":
        cfg.task.fixed_init = None
        cfg.task.fixed_target = None
        cfg.task.episode_sampler = "edge"
        print("[eval_ckpt] eval_points=train -> 训练同分布(edge 随机起终点), 验证任意起终点泛化")
    else:
        print("[eval_ckpt] eval_points=fixed -> 固定起终点(single-point acceptance)")
    # [2026-09-07] 打印 eval 口径(日志自解释): 严格单命 vs 窗口多命 + runtime filter。
    #   验收默认 = 严格单命(task.soft_respawn=false); 若为 true 是训练同构的 soft-respawn
    #   窗口口径(多命, 稀释失败, 数字偏乐观), 结果应标注口径再比较。
    _sr = bool(cfg.task.get("soft_respawn", True))
    _rf = bool(cfg.get("runtime_filter", True))
    print("[eval_ckpt] semantics: soft_respawn={} -> {}"
          .format(_sr, "STRICT single-life (acceptance default, keep task.soft_respawn=false)"
                  if not _sr else "window multi-life (soft-respawn, optimistic)"))
    print(f"[eval_ckpt] runtime_filter={_rf} -> "
          + ("CBF velocity filter ON (deploy)" if _rf else "CBF filter DISABLED (internalize check)"))
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    cbf_filter = None            # [K5 2026-09-12] 由 action_transform=="velocity" 分支赋值
    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            # [B1 2026-09-09] mount the low-level controller on the env so a reset with
            #   task.controller_sync_dr=true re-syncs it to the per-env randomized mass/KF.
            #   No effect unless controller_sync_dr is on (env sync is gated on the flag).
            base_env.low_level_controller = controller
            vl = cfg.task.get("vel_limit", {})
            transforms.append(VelController(
                controller,
                max_vel=vl.get("max_vel", None),
                max_yaw_rate=vl.get("max_yaw_rate", None),
            ))
            # [M2-3] optional CBF velocity filter (appended AFTER VelController so Compose
            # applies it FIRST into the env; same as train.py).
            # [2026-09-06] +runtime_filter=false disables the CBF filter transform during
            # eval (policy raw action goes straight to the controller) -> quantifies the
            # runtime-filter contribution to collision safety under perturbation.
            # [K5 2026-09-12] runtime_filter=false 现在改为挂一个 **shadow** 滤波器：照常
            #   计算 a_cbf 与 ‖Δa‖/h_min 做诊断，但 **不写回动作** -> 策略行为与"完全撤掉
            #   filter"逐位一致，同时仍能量出"滤波器本会介入多少"（plan §4.3 零介入率 /
            #   E‖Δa‖ p50,p95 / h_min^train）。O2 实测：shadow 与无滤波器 rollout 的动作
            #   序列完全一致（见 acceptance.md 记录）。
            from omni_drones.utils.cbf import build_cbf_filter
            cbf_filter = build_cbf_filter(
                cfg,
                shadow=not bool(cfg.get("runtime_filter", True)),
                record_diag=bool(cfg.get("cbf_diag", True)),
            )
            if cbf_filter is not None:
                transforms.append(cbf_filter)
                if not bool(cfg.get("runtime_filter", True)):
                    print("[eval_ckpt] runtime_filter=false -> CBF filter in SHADOW mode "
                          "(a_cbf computed for diagnostics only, action passed through)")
            elif not bool(cfg.get("runtime_filter", True)):
                print("[eval_ckpt] runtime_filter=false -> CBF velocity filter DISABLED "
                      "(cbf.mode=none: nothing to shadow)")
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transforms.append(PIDRateController(controller))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec,
        device=base_env.device,
    )
    policy.load_state_dict(torch.load(cfg.checkpoint, map_location=cfg.sim.device))
    print(f"[eval_ckpt] loaded checkpoint from {cfg.checkpoint}")

    # [2026-09-05 CBF-sensitivity] optional command-space perturbation (none default)
    perturb = str(cfg.get("perturb", "none"))
    pstrength = float(cfg.get("perturb_strength", 0.3))
    if perturb in ("cmd_gauss", "cmd_pulse"):
        policy = _PerturbWrapper(policy, perturb, pstrength)
        print(f"[eval_ckpt] perturbation mode={perturb} strength={pstrength:.3f} "
              f"(injected on raw action, before CBF filter)")
    elif perturb != "none":
        raise ValueError(f"Unknown perturb mode: {perturb}")

    env.eval()
    base_env.eval()
    base_env.enable_render(False)

    steps = int(cfg.get("rollout_steps", 400))
    # [2026-09-08 reach 口径] +record_min_rpos=true: rollout callback 累积每 env 距目标的
    #   最小距离 -> 打印 "曾到过一次(r<arrive_radius, 不需保持50步)" 比例, 用于判断 DR 崩在
    #   "到不了目标" 还是 "到了但保持不住"(对照 arrival_rate = 曾到∧保持50)。
    _rec = bool(cfg.get("record_min_rpos", False))
    class _MinRposCb:
        def __init__(self):
            self.m = None
        def __call__(self, env, *args):
            r = torch.norm(env.rpos.float(), dim=-1).squeeze(-1)   # (N,)
            self.m = r if self.m is None else torch.minimum(self.m, r)
    _cb = _MinRposCb() if _rec else None

    # ==================================================================================
    # [K5 2026-09-12] plan §4.2 / §4.3 验收指标累加器（只读 env 内部状态，不改 env 行为）
    #   arrival@r  = 距目标 < r 且连续保持 hold_02 步（0.2 s @100 Hz = 20 步）的 env 比例
    #                （**锁存**、不看 done：与该 env 自身 arrival_triggered 的"曾到达∧保持"
    #                 语义一致；arrival_at[0.5] 应与脚本原有的 arrival_rate 同量级互证）
    #   stall      = 速度 < stall_vel 的步占比，**仅在未到达的 env** 上统计（防"等待"策略）
    #   dropped_relevant = 几何净空 < danger_radius 但**未进 obs 窗口**(最近 K 个) 的障碍
    #                      —— 策略"看不见"却会被判撞的隐患（plan §3.2/§4.2）
    #   ⚠ 早期版本用 td["next","done"] 做单命门控，实测与 env 语义不符（env 在 rollout 中
    #     会重置：episode_len mean≈1212<1500），会把已到达 env 误判为"死"而漏记到达 —— 已移除。
    # ==================================================================================
    ARR_RADII = tuple(float(x) for x in cfg.get("arr_r", [0.5, 0.3, 0.2]))
    HOLD_02 = max(1, int(round(0.2 / float(cfg.sim.dt))))
    STALL_V = float(cfg.get("stall_vel", 0.05))

    class _AcceptanceCb:
        def __init__(self, b):
            n, dev = b.num_envs, b.device
            self.n = n
            self.inside = {r: torch.zeros(n, 1, dtype=torch.long, device=dev) for r in ARR_RADII}
            self.arrived = {r: torch.zeros(n, 1, dtype=torch.bool, device=dev) for r in ARR_RADII}
            self.stall_steps = torch.zeros(n, 1, device=dev)
            self.notarrived_steps = torch.zeros(n, 1, device=dev)
            self.oob = torch.zeros(n, 1, dtype=torch.bool, device=dev)   # "曾经" OOB
            self.nan = torch.zeros(n, 1, dtype=torch.bool, device=dev)   # "曾经" NaN(坠毁)
            self.dr_steps = 0.0
            self.dr_relevant = 0.0
            self.dr_dropped = 0.0
            self.steps = 0
            self.layout_fp = None            # 布局指纹（首步采样一次），验证 ON/OFF 同分布
            # [2026-09-16] plan 4.2 trajectory metrics (guide-defined).  `rpos` is
            # `target_pos - drone_state[..., :3]`, so it gives us the height error and the
            # straight-line distance for free, with no extra env plumbing:
            #   z_err_rmse        = RMSE of |rpos_z| over the whole rollout
            #   path_length_ratio = sum |d rpos| / |rpos| at the first sample
            #   terminal_speed_xy = mean |v_xy| over the LAST 0.5 s of the rollout
            self.dt = float(getattr(b, "dt", 0.01) or 0.01)
            self.term_win = max(1, int(round(0.5 / self.dt)))
            self.term_buf = collections.deque(maxlen=self.term_win)
            # [2026-09-16] `terminal_z_err` is a DIAGNOSTIC, not a plan row, added because
            #   a whole-rollout `z_err` RMSE is dominated by the initial climb (start z 0.5
            #   -> goal z 1.0) and therefore cannot be read against the guide's 0.10 m bar,
            #   which belongs to the hardware regime (the real drone starts near the goal
            #   altitude).  Splitting "did it end up at the goal height" from "how fast did
            #   it climb" makes the RMSE interpretable instead of mysterious.
            self.term_z_buf = collections.deque(maxlen=self.term_win)
            self.z_sq_sum = 0.0
            self.z_n = 0
            self.path_len = torch.zeros(n, 1, device=dev)
            self.straight = None             # (N,1) straight-line distance, set on step 1
            self.prev_rpos = None

        @torch.no_grad()
        def __call__(self, _env, *args, **kwargs):
            b = base_env
            self.steps += 1
            # 0 OOB / 0 crash = "曾经发生过"口径（逐步骤计，不受终局状态影响）
            pos_ = b.drone_state[..., :3].float()
            self.nan |= (~torch.isfinite(pos_).all(-1))
            oob_ = (pos_[..., 2] < b.z_min) | (pos_[..., 2] > b.z_max)
            ax = getattr(b, "arena_bound_x", None)
            if ax is not None:
                oob_ = oob_ | (pos_[..., 0].abs() > ax) | (pos_[..., 1].abs() > b.arena_bound_y)
            self.oob |= oob_
            r = torch.norm(b.rpos.float(), dim=-1)                        # (N,1)
            # ---- [2026-09-16] trajectory metrics (see __init__ for the definitions) --
            # The callback runs AFTER each env.step, so the first sample here is the state
            # at t=dt, not t=0.  The straight-line distance is therefore short by at most
            # one step of travel (~1 m/s * 0.01 s = 1 cm), which is far below the 0.01
            # resolution we report the ratio at; noted rather than corrected.
            rp = b.rpos.float()                                           # (N,1,3)
            if self.straight is None:
                self.straight = torch.norm(rp, dim=-1)                    # (N,1)
            else:
                self.path_len += torch.norm(rp - self.prev_rpos, dim=-1)
            self.prev_rpos = rp
            ze = rp[..., 2].abs()
            fz = torch.isfinite(ze)
            if bool(fz.any()):
                self.z_sq_sum += float((ze[fz] ** 2).sum())
                self.z_n += int(fz.sum())
            # xy speed uses the same velocity slice as `sp` below, but drops z
            self.term_buf.append(torch.norm(b.drone_state[..., 7:9].float(), dim=-1))
            self.term_z_buf.append(ze)
            for rad in ARR_RADII:
                self.inside[rad] = torch.where(r < rad, self.inside[rad] + 1,
                                               torch.zeros_like(self.inside[rad]))
                self.arrived[rad] |= (self.inside[rad] >= HOLD_02)
            sp = torch.norm(b.drone_state[..., 7:10].float(), dim=-1)     # (N,1) 线速度
            arrived_any = b.arrival_triggered.bool().reshape(-1, 1)
            na = (~arrived_any) & torch.isfinite(sp)
            self.notarrived_steps += na.float()
            self.stall_steps += (na & (sp < STALL_V)).float()
            # ---- dropped_relevant（仅滑动窗口模式；固定槽模式无"被挤出"） ----
            obs = getattr(b, "obstacles", None)
            idx = getattr(obs, "_obs_win_idx", None) if obs is not None else None
            if obs is not None and self.layout_fp is None:
                # 首步的布局指纹：位置/半径/激活数的和 + 首步最小净空（对置换不敏感，
                # 不同布局几乎必不同）-> 用于确认 ON 与 OFF 跑的是**同一套**障碍布置。
                try:
                    cl0 = obs.clearances(b.drone_state[..., :3])
                    mn = cl0[torch.isfinite(cl0)].min()
                    self.layout_fp = "{:.3f}/{:.3f}/{}/{}".format(
                        float(obs.pos.sum()), float(obs.radius.sum()),
                        int(obs.active.sum()),
                        ("%.3f" % float(mn)) if torch.isfinite(mn) else "inf")
                except Exception:
                    pass
            if idx is not None:
                clr = obs.clearances(b.drone_state[..., :3])               # (N,M) inf=inactive
                rel = torch.isfinite(clr) & (clr < b.obstacle_danger_radius)
                gid = getattr(obs, "_obs_win_gid", None)
                gid_full = getattr(obs, "_pillar_id", None)
                if gid is not None and gid_full is not None:
                    # [P1 C 2026-09-14] Per-pillar window: a pillar counts as relevant when
                    # ANY of its layers is inside the danger radius, and the window holds
                    # whole pillars - so the comparison has to be per pillar as well.
                    # Reusing the per-layer formula here would report the *other* layers of
                    # an already-visible pillar as "dropped", which is not what this metric
                    # means.  Consequence: per-pillar and per-layer numbers are NOT
                    # comparable, so the report must say which mode produced them.
                    n_g = int(getattr(obs, "_n_groups", 0))
                    # [2026-09-14] `obs._obs_win_gid` is the GROUP ID of each WINDOW slot,
                    #   i.e. shape (N, K), while `rel` has one column per SLOT, shape
                    #   (N, M) with M = nP_max * L_max = 48 for A3.  Scattering `rel` with
                    #   that (N, K) index does NOT raise: torch accepts index.size(1) <=
                    #   src.size(1) and silently uses only the FIRST K slot columns, with
                    #   group ids used as column numbers.  That produced a meaningless
                    #   "dropped_relevant_step_frac = 0.90" for A3 which looks exactly like
                    #   "route C does not work" - the opposite of the truth (with one slot
                    #   per pillar and K = n_groups nothing CAN be dropped).  Use the full
                    #   (M,) slot -> group map for `rel_g`; only `win_g` comes from the
                    #   window's group ids.
                    # scatter_reduce_ with a numeric reduce op is NOT implemented for bool
                    # on CUDA ("cuda_scatter_gather_base_kernel_func not implemented for
                    # 'Bool'"), so reduce on a float view and threshold back.
                    rel_g = torch.zeros(rel.shape[0], n_g, dtype=torch.float32,
                                        device=rel.device)
                    rel_g.scatter_reduce_(1, gid_full.expand(rel.shape[0], -1),
                                          rel.float(), reduce="amax")
                    rel_g = rel_g > 0.5
                    win_gid = getattr(obs, "_obs_win_gid")
                    win_val = getattr(obs, "_obs_win_valid")
                    if win_val is None:                     # defensive: no window info
                        win_val = torch.ones_like(win_gid, dtype=torch.bool)
                    # max-combine so an invalid (padded) column can never erase a valid
                    # one, and clamp because padded columns carry arbitrary group ids
                    wf = torch.zeros_like(rel_g, dtype=torch.float32)
                    wf.scatter_reduce_(1, win_gid.clamp(min=0, max=n_g - 1),
                                       win_val.float(), reduce="amax")
                    win_g = wf > 0.5
                    dropped = rel_g & ~win_g
                    self.dr_relevant += float(rel_g.sum())
                    self.dr_dropped += float(dropped.sum())
                    self.dr_steps += float(dropped.any())
                elif rel.any():
                    win = torch.zeros_like(rel)
                    win.scatter_(1, idx.clamp(min=0),
                                 torch.ones_like(idx, dtype=torch.bool))
                    win &= rel
                    dropped = rel & (~win)
                    self.dr_relevant += float(rel.sum())
                    self.dr_dropped += float(dropped.sum())
                    self.dr_steps += float(dropped.any())

    class _ChainCb:
        def __init__(self, fns):
            self.fns = [f for f in fns if f is not None]

        def __call__(self, env, *a, **kw):
            for f in self.fns:
                f(env, *a, **kw)

    _acc = _AcceptanceCb(base_env)
    if cbf_filter is not None:
        cbf_filter.reset_diag()
    _chain = _ChainCb([_cb, _acc])

    td = env.reset()
    # [K5 2026-09-12] 显存约束：torchrl 的 non-stop rollout 会把全部 T 步的 tensordict 累积
    #   后再 stack。实测 T=1500 × num_envs=1024 会把 32 GiB 显存打爆（CUDA OOM）；
    #   T=1500 × num_envs=512 峰值约 8–9 GiB，可 3 进程并行。评估协议因此固定为
    #   **512 envs × 1500 步（= T，完整一局）**，并在 acceptance.md 里留档。
    with set_exploration_type(ExplorationType.MODE):
        env.rollout(
            max_steps=steps,
            policy=policy,
            tensordict=td,
            auto_reset=False,
            break_when_any_done=False,
            callback=_chain,
        )
    if _rec:
        ar = float(getattr(base_env, "arrive_radius", 0.5))
        reach = (_cb.m < ar).float()
        print(f"[eval_ckpt] REACH(到过一次 r<{ar}, 不需保持): {reach.mean().item():.3f} "
              f"({reach.sum().item()}/{reach.numel()})  min_rpos mean={_cb.m.mean().item():.3f}", flush=True)

    stats = base_env.stats
    def r(k):
        v = stats[k].float()
        return v.mean().item(), v.std().item(), v.min().item(), v.max().item()

    print("\n========== [eval_ckpt] rollout stats over %d steps ==========" % steps)
    for k in ("pos_error", "heading_alignment", "uprightness", "return", "episode_len", "action_smoothness"):
        if k in stats.keys():
            m, s, lo, hi = r(k)
            print(f"  {k:20s} mean={m:+.3f}  std={s:.3f}  min={lo:+.3f}  max={hi:+.3f}")

    # direct position error to the hover target
    if hasattr(base_env, "rpos"):
        d = torch.norm(base_env.rpos.float(), dim=-1)  # (num_envs, 1)
        print(f"  {'|rpos| (dist to target)':20s} mean={d.mean().item():.3f}  max={d.max().item():.3f}  <0.1 count={(d<0.1).sum().item()}/{d.numel()}")
    if hasattr(base_env, "drone"):
        pos = base_env.drone.pos[..., :2].float() if base_env.drone.pos.ndim >= 2 else None
        if pos is not None:
            drift = torch.norm(pos, dim=-1)
            print(f"  {'xy drift from (0,0)':20s} mean={drift.mean().item():.3f}  max={drift.max().item():.3f}")

    # [M2 2026-09-04] obstacle-aware evaluation (naive nav, no CBF): read the env's
    # *internal* counters directly after a no-reset rollout so the numbers are honest.
    #   arrival rate  = fraction of envs that reached AND held the target >= once
    #   collision rate= collision-edge events / total steps  (edge = new contact)
    #   min_clearance = per-env min surface clearance over the rollout (distribution)
    if hasattr(base_env, "_has_obstacles") and base_env._has_obstacles:
        n_env = base_env.num_envs
        if hasattr(base_env, "arrival_triggered"):
            arr = base_env.arrival_triggered.float()
            print(f"\n========== [eval_ckpt] obstacle metrics (level={base_env.level_idx}, "
                  f"{base_env.curriculum_levels[base_env.level_idx] if base_env.curriculum_levels else 0} obstacles) ==========")
            print(f"  {'arrival_rate (arrived&held >=once)':28s} = {arr.mean().item():.3f}  ({arr.sum().item()}/{n_env})")
            if hasattr(base_env, "ep_collision_edges"):
                edges = base_env.ep_collision_edges.float()           # per-env edge count
                n_edges = edges.sum().item()
                print(f"  {'collision edges (total)':28s} = {n_edges:.0f}  rate={n_edges/(n_env*steps):.4f}")
                print(f"  {'envs with >=1 collision edge':28s} = {edges.gt(0).sum().item()}/{n_env}  "
                      f"({edges.gt(0).float().mean().item():.3f})")
                # joint window success used by the curriculum (arrival & zero edges)
                if hasattr(base_env, "episode_any_arrival"):
                    suc = (base_env.episode_any_arrival.squeeze(-1) & (edges.squeeze(-1) == 0)).float()
                    print(f"  {'joint success (arr & 0 edges)':28s} = {suc.mean().item():.3f}  ({suc.sum().item()}/{n_env})")
            if hasattr(base_env, "obstacles") and base_env.obstacles is not None:
                mc = base_env.obstacles.ep_min_clearance.float().squeeze(-1)
                finite = torch.isfinite(mc)
                if finite.any():
                    fmc = mc[finite]
                    print(f"  {'min_clearance (rollout min, m)':28s} mean={fmc.mean().item():.3f}  "
                          f"std={fmc.std().item():.3f}  min={fmc.min().item():.3f}  max={fmc.max().item():.3f}  "
                          f"<0.1 count={(fmc<0.1).sum().item()}/{finite.sum().item()}")
                else:
                    print(f"  {'min_clearance':28s} = inf (no active obstacles in eval)")
            # [env_design 2026-09-07] 未到达 env 的结束归类(按 rollout 末状态, 单命 soft_respawn=false 有效):
            #   类别 = crash(坠地/NaN) | oob(出界/z>z_max) | collide(碰撞边沿) | timeout(600 未到且未坠/出界)
            #   arrived = 本窗曾到达保持(与 arrival_rate 一致, 不因其后坠/出界而撤销)
            if hasattr(base_env, "drone_state") and hasattr(base_env, "arrival_triggered"):
                pos = base_env.drone_state[..., :3].float()                 # (N,1,3) env frame
                z = pos[..., 2]
                nan = torch.isnan(pos).any(-1)
                crash = (z < base_env.z_min) | nan
                oob = z > base_env.z_max
                if getattr(base_env, "arena_bound_x", None) is not None:
                    oob = oob | (pos[..., 0].abs() > base_env.arena_bound_x) \
                            | (pos[..., 1].abs() > base_env.arena_bound_y)
                elif hasattr(base_env, "bound_xy"):
                    oob = oob | (torch.norm(pos[..., :2], dim=-1) > base_env.bound_xy)
                coll = torch.zeros_like(crash)
                if hasattr(base_env, "ep_collision_edges"):
                    coll = base_env.ep_collision_edges.squeeze(-1) > 0
                arr = base_env.arrival_triggered.squeeze(-1).bool()
                cr = crash.squeeze(-1); ob = oob.squeeze(-1); cl = coll
                timeout = ~(cr | ob | cl | arr)          # 跑到 end 未到、没坠/出界/碰
                print("  end-cause (per-env, not-arrived categorized):")
                for name, m in [("crash", cr), ("oob", ob & ~cr), ("collide", cl & ~cr & ~ob),
                                ("timeout", timeout)]:
                    print(f"    {name:8s} = {m.sum().item():4d}   (arrived {arr[m].sum().item():4d})")
                print(f"    {'arrived':8s} = {arr.sum().item():4d}")
            for k in ("collision", "collision_episodes", "min_clearance", "success_rate"):
                if k in stats.keys():
                    m, s, lo, hi = r(k)
                    print(f"  {'stats.'+k:26s} mean={m:+.3f}  std={s:.3f}  min={lo:+.3f}  max={hi:+.3f}")
            # [2026-09-05 CBF-sensitivity] mean per-step CBF reward-core violation
            # (EMA, >=0) meaningful only when the env has a CBF reward core / filter.
            if hasattr(base_env, "cbf_extra") and base_env.cbf_extra is not None and \
                    "cbf_violation" in stats.keys():
                m, s, lo, hi = r("cbf_violation")
                print(f"  {'stats.cbf_violation (per-step, >=0)':26s} mean={m:+.4f}  std={s:.4f}  "
                      f"max={hi:+.4f}")

    # ==================================================================================
    # [K5 2026-09-12] 验收报告（plan §4.2 阶段 1 / §4.3 阶段 2）
    #   一次跑同时给出「到达率(多半径) / stall / 碰撞 / OOB / h_min / 零介入率 / ‖Δa‖」，
    #   并以 [eval_metrics] JSON 单行输出 -> 便于多 seed × ON/OFF 批量汇总。
    #   注：runtime_filter=false 时 CBF 数字来自 **shadow** 滤波器 = "滤波器本会介入多少"，
    #       是反事实量；ON 时才是真实介入。
    # ==================================================================================
    import json

    rep = {
        "model_id": str(cfg.get("model_id", "unknown")),
        "checkpoint": str(cfg.checkpoint),
        "seed": int(cfg.get("seed", -1)),
        "runtime_filter": bool(cfg.get("runtime_filter", True)),
        "eval_points": str(_ep),
        "rollout_steps": int(steps),
        "num_envs": int(base_env.num_envs),
        "task": str(cfg.task.name),
        "device": str(cfg.sim.device),
        "layout_fp": _acc.layout_fp,
    }
    rep["arrival_at"] = {f"{r:g}": round(float(_acc.arrived[r].float().mean()), 4)
                         for r in ARR_RADII}
    # [2026-09-14] HORIZON-INDEPENDENT speed metric.  arrival@0.2 saturates as soon as the
    #   rollout covers a whole episode - measured on ONE checkpoint with everything else
    #   fixed: 0.7943 at 600 steps, 0.9635 at 900, 0.9974 at 1500 - so a time-boxed rate
    #   cannot discriminate rungs at a long horizon (and corr_mean/corr_p95 drift down with
    #   the horizon too).  The env already records, per env, the progress_buf step of the
    #   first arrival inside a window (nav_vel `stats["first_arrival_step"]`, 0 = never
    #   arrived).  Being episode-relative, it does not depend on the rollout length, so its
    #   median/p90 over the envs that DID arrive is the quantity that stays comparable.
    try:
        _fas = base_env.stats["first_arrival_step"].reshape(-1)
        _ok = _fas > 0
        rep["arrival_steps_n"] = int(_ok.sum())
        if bool(_ok.any()):
            _v = _fas[_ok].float()
            rep["arrival_steps_median"] = round(float(_v.median()), 1)
            rep["arrival_steps_mean"] = round(float(_v.mean()), 1)
            rep["arrival_steps_p90"] = round(float(_v.quantile(0.9)), 1)
    except Exception as _e:                       # diagnostic only; never break the eval
        rep["arrival_steps_note"] = f"unavailable: {type(_e).__name__}"

    # [2026-09-14] PILLAR-COUNT distribution.  A3 randomizes pillars 2-8, and `min_active_
    #   slots >= K` is deliberately NOT applied there (in per-pillar obs mode that bound
    #   would mean "every env must have 8 pillars", destroying the range).  So the risk it
    #   used to cover - the world silently collapsing onto one difficulty - has to be
    #   watched instead of policed, and this is where it is watched.  Reported per env over
    #   the sampled layouts; a healthy A3 run shows a spread, not a single value.
    try:
        _pc = base_env.obstacles.pillar_counts().float()
        rep["active_pillars_min"] = int(_pc.min())
        rep["active_pillars_mean"] = round(float(_pc.mean()), 3)
        rep["active_pillars_max"] = int(_pc.max())
        rep["active_pillars_distinct"] = int(torch.unique(_pc).numel())
    except Exception as _e:                       # diagnostic only
        rep["active_pillars_note"] = f"unavailable: {type(_e).__name__}"
    rep["notarrived_step_frac"] = round(
        float(_acc.notarrived_steps.sum() / max(_acc.steps * base_env.num_envs, 1)), 4)
    rep["stall_frac"] = round(
        float(_acc.stall_steps.sum() / _acc.notarrived_steps.sum().clamp(min=1)), 4)
    rep["dropped_relevant_frac"] = round(
        _acc.dr_dropped / max(_acc.dr_relevant, 1.0), 4)
    rep["oob_envs_ever"] = int(_acc.oob.sum())
    rep["crash_envs_ever"] = int(_acc.nan.sum())
    rep["dropped_relevant_step_frac"] = round(
        _acc.dr_steps / max(_acc.steps, 1), 4)
    rep["relevant_obstacles_total"] = int(_acc.dr_relevant)
    if hasattr(base_env, "ep_collision_edges"):
        e = base_env.ep_collision_edges.float().squeeze(-1)
        rep["collision_envs"] = int(e.gt(0).sum())
        rep["collision_edges"] = int(e.sum())
        if hasattr(base_env, "episode_any_arrival"):
            arr = base_env.arrival_triggered.squeeze(-1).bool()
            rep["joint_success_envs"] = int((arr & (e == 0)).sum())
    if hasattr(base_env, "obstacles") and base_env.obstacles is not None:
        mc = base_env.obstacles.ep_min_clearance.float().squeeze(-1)
        f = mc[torch.isfinite(mc)]
        if f.numel():
            rep["min_clearance_global_min"] = round(float(f.min()), 4)
            rep["min_clearance_env_mean"] = round(float(f.mean()), 4)
            # [2026-09-16] THREE CONVENTIONS, ONE RAW NUMBER.
            #   `ObstacleManager.clearances()` returns
            #       |p - p_oi| - (r_oi + drone_radius + inflation)
            #   i.e. the CENTRE distance minus the *whole* decision radius.  The env header
            #   (nav_vel_obstacles.py L27-28) says drone_radius is the drone's physical
            #   sphere and inflation is an extra cushion on top, so this raw value is
            #   drone-surface-to-obstacle-surface MINUS the cushion.  Three different
            #   documents then call the result "d_min" while meaning three different
            #   quantities, and the profile-A pad is small enough (0.12 m) that the choice
            #   flips `min d_min >= 0.10` from PASS to FAIL.  So report all three, derived
            #   from the live config rather than hardcoded, and let the plan pick:
            #     cbf     = |p-p_oi| - (r_oi+dr+inf)   <- what the CBF acts on; == raw
            #     surface = |p-p_oi| - (r_oi+dr)       <- drone body surface <-> obstacle
            #                                             surface  (plan 4.2 "表面净空")
            #     center  = |p-p_oi| - r_oi            <- drone centre <-> obstacle surface
            #                                             (guide B.3 notation `d_i`)
            # Cross-check available at runtime: h_min_train should equal
            # (min_clearance_global_min - cbf.r_safety_margin).
            _ob = base_env.obstacles
            pad_inf = float(getattr(_ob, "inflation", 0.0) or 0.0)
            pad_cen = pad_inf + float(getattr(_ob, "drone_radius", 0.0) or 0.0)
            rep["clearance_pad_inflation"] = round(pad_inf, 4)
            rep["clearance_pad_center"] = round(pad_cen, 4)
            rep["clearance_conv_note"] = ("min_clearance_global_min is the CBF/raw "
                                          "convention; +pad_inflation = surface; "
                                          "+pad_center = centre")
            rep["min_clearance_surface_global_min"] = round(float(f.min()) + pad_inf, 4)
            rep["min_clearance_center_global_min"] = round(float(f.min()) + pad_cen, 4)
            rep["min_clearance_surface_env_mean"] = round(float(f.mean()) + pad_inf, 4)
    # ---- [2026-09-16] plan 4.2 trajectory metrics that the aggregator was MISSING ----
    #   These were listed in the plan 4.2 table but never measured, so the "all gates
    #   pass" line for A3/A4 was missing three of its own rows.  Definitions come from
    #   NAVVEL_RETRAIN_GUIDE.md: z_err RMSE = RMSE of |z - z_goal| over the rollout;
    #   terminal speed = mean |v_xy| over the LAST 0.5 s; path length ratio = arc length
    #   / straight-line distance.
    if _acc.z_n:
        rep["z_err_rmse"] = round(math.sqrt(_acc.z_sq_sum / _acc.z_n), 4)
        rep["z_err_n"] = int(_acc.z_n)
    if _acc.term_buf:
        tw = torch.stack(list(_acc.term_buf), 0)                # (<=W, N, 1)
        tv = tw[torch.isfinite(tw)]
        if tv.numel():
            rep["terminal_speed_xy"] = round(float(tv.mean()), 4)
        # audit the window: it must be ~0.5 s, otherwise the gate is measuring a
        # different horizon than the guide's definition says it does
        rep["terminal_speed_win_steps"] = int(tw.shape[0])
        rep["terminal_speed_win_sec"] = round(tw.shape[0] * _acc.dt, 3)
    if _acc.term_z_buf:
        zw = torch.stack(list(_acc.term_z_buf), 0)
        zv = zw[torch.isfinite(zw)]
        if zv.numel():
            rep["terminal_z_err"] = round(float(zv.mean()), 4)
    if _acc.straight is not None:
        d0 = _acc.straight.squeeze(-1)
        plr = _acc.path_len.squeeze(-1) / d0.clamp(min=1e-6)
        fin = torch.isfinite(plr) & (d0 > 1e-6)
        if bool(fin.any()):
            rep["path_length_ratio"] = round(float(plr[fin].mean()), 4)
    rep["cbf_extra"] = round(float(getattr(base_env, "cbf_extra", 0.0) or 0.0), 4)
    if cbf_filter is not None and cbf_filter.diag_tensor() is not None:
        dg = cbf_filter.diag_tensor().reshape(-1, 4)
        corr, inter, hmin = dg[:, 0], dg[:, 1], dg[:, 2]
        fin = torch.isfinite(corr)
        cf = corr[fin]
        rep["cbf_diag_steps"] = int(fin.sum())
        rep["zero_intervention_rate"] = round(float((cf == 0).float().mean()), 4)
        rep["intervened_step_frac"] = round(float((inter[fin] > 0).float().mean()), 4)
        rep["corr_mean"] = round(float(cf.mean()), 4)
        rep["corr_p50"] = round(float(torch.quantile(cf, 0.50)), 4)
        rep["corr_p95"] = round(float(torch.quantile(cf, 0.95)), 4)
        hf = hmin[torch.isfinite(hmin)]
        rep["h_min_train"] = round(float(hf.min()), 4) if hf.numel() else None
        rep["h_below0_frac"] = (round(float((hf < 0).float().mean()), 4)
                                if hf.numel() else None)
        # ---- [P3 2026-09-16] 可选 diag 落盘（+dump_diag=<dir>） -------------------
        #   把逐步 diag (T, N, 4) = [corr, intervened, h_min, fix_norm] 存成 npz，
        #   供 CPU 离线分析"干预发生在什么时候/离边界多近"（plan §7.1 收尾、F3 定标）。
        #   默认不写（不改变任何既有行为）。
        _dd = cfg.get("dump_diag", None)
        if _dd:
            try:
                import os as _os
                import numpy as _np
                _dg = cbf_filter.diag_tensor().detach().cpu()
                _os.makedirs(str(_dd), exist_ok=True)
                _fn = _os.path.join(
                    str(_dd),
                    f"diag_s{rep['seed']}_{'on' if rep['runtime_filter'] else 'off'}.npz")
                _np.savez_compressed(_fn, diag=_dg.numpy(),
                                     cbf_extra=float(rep.get("cbf_extra", 0.0)),
                                     seed=int(rep["seed"]),
                                     runtime_filter=bool(rep["runtime_filter"]))
                rep["diag_dump"] = _fn
                print(f"[eval_ckpt] diag dumped -> {_fn}  shape={tuple(_dg.shape)}",
                      flush=True)
            except Exception as _e:                    # diagnostic only; never break the eval
                rep["diag_dump_error"] = f"{type(_e).__name__}: {_e}"
    if cbf_filter is None:
        rep["zero_intervention_rate"] = None
        rep["h_min_train"] = None

    print("\n========== [eval_ckpt] ACCEPTANCE METRICS (plan 4.2 / 4.3) ==========")
    print(f"  {'runtime_filter':28s} = {rep['runtime_filter']}"
          f"{'' if rep['runtime_filter'] else '   (CBF 数值 = shadow 反事实)'}")
    for k in ("arrival_at", "notarrived_step_frac", "stall_frac", "collision_envs",
              "collision_edges", "oob_envs_ever", "crash_envs_ever",
              "joint_success_envs", "min_clearance_global_min",
              "min_clearance_env_mean",
              # [2026-09-16] the two extra clearance conventions (see the comment where
              # they are computed) - printed so the plan's ambiguous `min d_min >= 0.10`
              # can be adjudicated from one run instead of three.
              "clearance_pad_inflation", "clearance_pad_center",
              "min_clearance_surface_global_min", "min_clearance_center_global_min",
              "min_clearance_surface_env_mean",
              # [2026-09-16] plan 4.2 rows that were never measured before
              "z_err_rmse", "z_err_n", "terminal_z_err",
              "terminal_speed_xy", "terminal_speed_win_steps", "terminal_speed_win_sec",
              "path_length_ratio",
              "h_min_train", "h_below0_frac",
              "zero_intervention_rate", "intervened_step_frac", "corr_mean",
              "corr_p50", "corr_p95", "dropped_relevant_frac",
              "dropped_relevant_step_frac", "relevant_obstacles_total",
              "arrival_steps_median", "arrival_steps_mean", "arrival_steps_p90",
              "arrival_steps_n",
              "active_pillars_min", "active_pillars_mean", "active_pillars_max",
              "active_pillars_distinct"):
        if k in rep:
            print(f"  {k:28s} = {rep[k]}")
    print("[eval_metrics] " + json.dumps(rep, sort_keys=True), flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
