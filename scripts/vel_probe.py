# Direction-of-motion probe: command a CONSTANT world-frame +X velocity and
# measure the drone's actual displacement direction. Isolates whether the
# VelController/Lee path executes commanded WORLD velocity correctly, or
# whether there is a frame/direction bug.
# Usage:
#   python vel_probe.py task=NavVel algo=ppo headless=true wandb.mode=disabled \
#       task.env.num_envs=64 +steps=300 +cmd_vx=1.2
import torch
import hydra
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS  # noqa: F401
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


class ConstVelPolicy:
    """Commands a constant world-frame velocity [vx,0,0,0] (raw, pre-transform)."""
    def __init__(self, cmd):
        self.cmd = cmd  # (1,4) tensor

    def __call__(self, td):
        B = td.batch_size[0]
        td.set(("agents", "action"), self.cmd.expand(B, 1, 4).clone())
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
    if cfg.task.get("action_transform", None) == "velocity":
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
        vl = cfg.task.get("vel_limit", {})
        transforms.append(VelController(controller,
                                        max_vel=vl.get("max_vel", None),
                                        max_yaw_rate=vl.get("max_yaw_rate", None)))
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.eval(); base_env.eval(); base_env.enable_render(False)

    vx = float(cfg.get("cmd_vx", 1.2))
    steps = int(cfg.get("steps", 300))
    pol = ConstVelPolicy(torch.tensor([vx, 0.0, 0.0, 0.0], device=base_env.device))

    td = env.reset()
    p0 = base_env.drone.pos[..., 0, :3].clone().float()          # (B,3) initial
    # initial attitude (quat in drone_state[...,3:7]) & velocity
    ds0 = td.get(("info", "drone_state"), None)
    ds0 = ds0[..., :13].clone().float() if ds0 is not None else None
    q0 = ds0[..., 3:7] if ds0 is not None else None
    v0 = ds0[..., 7:10] if ds0 is not None else None

    import numpy as np
    def quat_yaw(q):  # q (B,4) (x,y,z,w) convention -> yaw around z
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # record actual (world) velocity direction every 50 steps
    vel_log = []
    with torch.no_grad():
        for i in range(steps):
            pol(td)
            td = env.step(td)
            if (i + 1) % 50 == 0:
                ds = td[("info", "drone_state")][..., :13].float()
                vel_log.append(ds[..., 7:10].clone())

    p1 = base_env.drone.pos[..., 0, :3].clone().float()          # final
    disp = p1 - p0                                               # actual displacement
    dx = disp[..., 0]; dy = disp[..., 1]; dz = disp[..., 2]
    ang = torch.atan2(dy, dx)  # displacement azimuth
    vel_log = torch.stack(vel_log, 0)  # (T,B,3) actual velocities sampled
    vfin = vel_log[-1]
    vang = torch.atan2(vfin[..., 1], vfin[..., 0])
    print("\n========== [vel_probe] const world +X %.2f m/s, %d steps ==========" % (vx, steps))
    print(f"  cmd: vx={vx} vy=0 vz=0")
    print(f"  displacement mean  dx={dx.mean().item():+.3f} dy={dy.mean().item():+.3f} dz={dz.mean().item():+.3f}")
    print(f"  displacement norm  mean={disp.norm(dim=-1).mean().item():.3f}")
    cosx = (dx / disp.norm(dim=-1).clamp(min=1e-4)).mean().item()
    print(f"  <dx/|disp|> (cos to +X, +1=perfect) = {cosx:+.3f}")
    print(f"  final actual vel   vx={vfin[...,0].mean().item():+.3f} vy={vfin[...,1].mean().item():+.3f} vz={vfin[...,2].mean().item():+.3f}  speed={vfin.norm(dim=-1).mean().item():.3f}")
    print(f"  <actual vx / |v|> (cos of motion to +X at end) = {(vfin[...,0]/vfin.norm(dim=-1).clamp(min=1e-4)).mean().item():+.3f}")
    if q0 is not None:
        yaw0 = quat_yaw(q0)
        print(f"  init yaw mean={torch.rad2deg(yaw0).mean().item():+.1f}° std={torch.rad2deg(yaw0).std().item():+.1f}°  (0°=nose +X)")
        # displacement azimuth vs init yaw: if body-frame cmd, az ≈ yaw0
        res = torch.rad2deg((ang - yaw0) % (2*torch.pi))
        res = torch.where(res > 180, res - 360, res)
        print(f"  az - init_yaw:  mean={res.mean().item():+.1f}° std={res.std().item():+.1f}°   (small std => body-frame cmd)")
    print("  sample env0-5: az vs init_yaw (deg):")
    for k in range(min(6, disp.shape[0])):
        y = torch.rad2deg(yaw0[k]).item() if q0 is not None else float('nan')
        print(f"    env{k}: az={torch.rad2deg(ang[k]).item():+.0f}°  init_yaw={y:+.0f}°  final_v_az={torch.rad2deg(vang[k]).item():+.0f}°")

    simulation_app.close()


if __name__ == "__main__":
    main()
