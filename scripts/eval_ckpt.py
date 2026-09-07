# One-off deterministic evaluation for a saved policy checkpoint (headless).
# Usage (from OmniDrones/scripts):
#   python eval_ckpt.py task=HoverCrazyflie algo=ppo headless=true wandb.mode=disabled \
#       +checkpoint=/path/to/checkpoint_XXXX.pt rollout_steps=400
#
# ===================== 标准验收模板（默认严格单命口径, 2026-09-07） =====================
# 目标: 一次飞行 = 一次考核（坠毁/出界/碰撞超限 ⇒ 该 env 判 terminated 且不复活）。
# NavVel 训练用 soft_respawn=true（软重生给多条命），但验收评估必须覆盖为 false 才是
# 「一次飞行失败即失败」的严格口径；窗口+多命会稀释失败、数字偏乐观。环境侧与训练同构，
# 只需配置覆盖、无需改代码（配合下方 auto_reset=False = 单命 600 步窗口）。
#
#   python eval_ckpt.py task=NavVel algo=ppo headless=true wandb.mode=disabled \
#       task.soft_respawn=false          # ← 验收默认：严格单命（必须显式给, 否则=多命乐观口径）
#       task.reward_scheme=f1 task.reward_fly_weight=6.0 \
#       task.arrive_bonus=30 task.arrive_time_bonus=30 task.reward_timeout_penalty=40 \
#       task.reward_action_smoothness_weight=0.2 \
#       task.obstacle.num_scene=16 'task.obstacle.spawn_xy_range=[[-2.2,-2.2],[2.2,2.2]]' \
#       'task.curriculum.levels=[16]' task.curriculum.enabled=false \
#       task.obstacle.reward_collision_edge=4 task.obstacle.reward_near_slowdown_weight=1.0 \
#       task.cbf.mode=hybrid task.cbf.use_brake_term=false task.cbf.penalty_src=dual \
#       task.cbf.reward_weight=0.1 task.cbf.correction_weight=0.1 task.cbf.correction_sigma=0.5 \
#       +checkpoint=<ckpt> +rollout_steps=600 +runtime_filter=true      # ON（带 CBF filter 部署）
#   # OFF（internalize/撤 filter 判据）换成 +runtime_filter=false；8obs 密度换 num_scene=8
#   # + levels=[8]。obs_safety 键须与训练一致（无则不加）。
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
    td = env.reset()
    with set_exploration_type(ExplorationType.MODE):
        env.rollout(
            max_steps=steps,
            policy=policy,
            tensordict=td,
            auto_reset=False,
            break_when_any_done=False,
        )

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
