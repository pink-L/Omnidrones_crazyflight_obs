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

    simulation_app.close()


if __name__ == "__main__":
    main()
