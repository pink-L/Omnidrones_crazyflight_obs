# One-off SCRIPTED-EXPERT probe (no learned policy, no RL).
# Purpose: isolate the CONTROL chain for NavVel (action -> VelController -> Lee ->
#         rotor -> motion). Every step we inject a target-seeking world velocity
#         `v = speed(d) * (target - pos)/||target - pos||` (same transform path a
#         trained policy would use). If the drone fails to converge on the target
#         and trigger arrival, the action/controller/coordinate mapping is broken
#         regardless of reward/learning.
# Usage (from OmniDrones/scripts):
#   python expert_probe.py task=NavVel algo=ppo headless=true wandb.mode=disabled \
#       rollout_steps=600 num_envs=128
import torch
import hydra
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS  # noqa: F401  (registers algo/* configs in ConfigStore)
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


class ScriptedTargetSeeker:
    """Mimics PPOPolicy.__call__: mutates td by setting ("agents","action")
    = raw target velocity (pre-transform). VelController then clamps speed to
    max_vel and hands it to the Lee controller -- exactly the trained path."""
    def __init__(self, base_env, v_nom: float = 1.5, tau: float = 1.0):
        self.base_env = base_env
        self.v_nom = v_nom
        self.tau = tau  # P-controller time constant: v = d/tau near target

    def __call__(self, td: torch.Tensor) -> torch.Tensor:
        env = self.base_env
        pos = env.drone.pos[..., 0, :3].float()          # (B,3) world pos
        tgt = env.target_pos[..., 0, :3].float()         # (B,3)
        delta = tgt - pos
        d = delta.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        # saturating P-control: far -> v_nom toward target, near -> v ~ d/tau -> 0
        speed = torch.minimum(self.v_nom * torch.ones_like(d), d / self.tau)
        vel = delta / d * speed
        yaw = torch.zeros_like(speed)
        action = torch.cat([vel, yaw], dim=-1).unsqueeze(1)   # (B,1,4)
        td.set(("agents", "action"), action)
        return td


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
        from omni_drones.utils.torchrl.transforms import ravel_composite
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform == "velocity":
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
        vl = cfg.task.get("vel_limit", {})
        transforms.append(VelController(
            controller,
            max_vel=vl.get("max_vel", None),
            max_yaw_rate=vl.get("max_yaw_rate", None),
        ))
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.eval()
    base_env.eval()
    base_env.enable_render(False)

    steps = int(cfg.get("rollout_steps", 600))
    policy = ScriptedTargetSeeker(base_env, v_nom=float(cfg.get("v_nom", 1.5)))
    td = env.reset()
    with torch.no_grad():
        env.rollout(
            max_steps=steps,
            policy=policy,
            tensordict=td,
            auto_reset=False,
            break_when_any_done=False,
        )

    d = torch.norm(base_env.rpos.float(), dim=-1)  # (num_envs,1)
    inside = (d < base_env.arrive_radius)
    arrival_count = base_env.episode_any_arrival.float().sum().item() if hasattr(base_env, "episode_any_arrival") else int(inside.sum())
    print("\n========== [expert_probe] scripted target-seeking over %d steps ==========" % steps)
    print(f"  |rpos| final      mean={d.mean().item():.3f}  std={d.std().item():.3f}  min={d.min().item():.3f}  max={d.max().item():.3f}")
    print(f"  inside radius(<{base_env.arrive_radius}) count = {int(inside.sum())}/{d.numel()}")
    print(f"  arrival_triggered      = {arrival_count}/{d.numel()}")
    r = lambda k: (lambda v: f"mean={v.mean().item():.3f}") (base_env.stats[k].float())
    for k in ("pos_error", "return", "episode_len"):
        if k in base_env.stats.keys():
            print(f"  {k:20s} {r(k)}")

    simulation_app.close()


if __name__ == "__main__":
    main()
