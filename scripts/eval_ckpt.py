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
import logging
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
            from omni_drones.utils.cbf import build_cbf_filter
            if bool(cfg.get("runtime_filter", True)):
                cbf_filter = build_cbf_filter(cfg)
                if cbf_filter is not None:
                    transforms.append(cbf_filter)
            else:
                print("[eval_ckpt] runtime_filter=false -> CBF velocity filter DISABLED")
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
    td = env.reset()
    with set_exploration_type(ExplorationType.MODE):
        env.rollout(
            max_steps=steps,
            policy=policy,
            tensordict=td,
            auto_reset=False,
            break_when_any_done=False,
            callback=_cb,
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

    simulation_app.close()


if __name__ == "__main__":
    main()
