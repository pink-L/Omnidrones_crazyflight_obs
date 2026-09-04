# Bypass any controller: command a CONSTANT rotor cmd (all 4 motors) directly to
# base_env and see if the Crazyflie actuator/mapping can hover. cmd_hover is the
# rotor cmd that should produce mg of total thrust (Lee-consistent normalization).
# If the drone hovers -> actuator/mapping fine, problem is the Lee controller.
# If it still sinks -> actuator/mass/motor-lag mapping is off.
# Usage:
#   python motor_hover_probe.py task=NavVel algo=ppo headless=true wandb.mode=disabled \
#       task.env.num_envs=64 +steps=400 +cmd=0.25
import torch
import hydra
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS  # noqa: F401
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


class RawMotorPolicy:
    def __init__(self, cmd: float, act_dim: int):
        self.cmd = cmd
        self.act_dim = act_dim
    def __call__(self, td):
        B = td.batch_size[0]
        td.set(("agents", "action"),
               torch.full((B, 1, self.act_dim), self.cmd, dtype=torch.float32))
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

    # NOTE: build with NO action transform so the action fed = raw rotor cmds.
    env = TransformedEnv(base_env, Compose([InitTracker()])).train()
    env.eval(); base_env.eval(); base_env.enable_render(False)

    cmd = float(cfg.get("cmd", 0.25))
    steps = int(cfg.get("steps", 400))
    # how many rotor cmds does the base env expect? inspect action spec
    act_dim = base_env.action_spec[("agents", "action")].shape[-1]
    print(f"[motor_hover] base_env action dim = {act_dim}")

    pol = RawMotorPolicy(cmd, act_dim)
    td = env.reset()
    p0 = base_env.drone.pos[..., 0, :3].clone().float()
    with torch.no_grad():
        for i in range(steps):
            pol(td)
            td = env.step(td)
    p1 = base_env.drone.pos[..., 0, :3].clone().float()
    disp = p1 - p0
    ds = td[("info", "drone_state")][..., :13].float()
    vel = ds[..., 7:10]
    print("\n========== [motor_hover] const rotor cmd %.3f, %d steps ==========" % (cmd, steps))
    print(f"  displacement mean  dx={disp[...,0].mean().item():+.3f} dy={disp[...,1].mean().item():+.3f} dz={disp[...,2].mean().item():+.3f}")
    print(f"  final vel          vx={vel[...,0].mean().item():+.3f} vy={vel[...,1].mean().item():+.3f} vz={vel[...,2].mean().item():+.3f}")

    simulation_app.close()


if __name__ == "__main__":
    main()
