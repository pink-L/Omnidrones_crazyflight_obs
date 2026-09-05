# One-off deterministic evaluation for a saved policy checkpoint (headless).
# Usage (from OmniDrones/scripts):
#   python eval_ckpt.py task=HoverCrazyflie algo=ppo headless=true wandb.mode=disabled \
#       +checkpoint=/path/to/checkpoint_XXXX.pt rollout_steps=400
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
