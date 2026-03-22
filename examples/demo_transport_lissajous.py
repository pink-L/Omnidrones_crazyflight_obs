"""
Multi-drone transport demo following a figure-eight (Lissajous) trajectory.

4 Hummingbird drones carry a payload along a smooth 3D Lissajous curve.
The trajectory traces a figure-eight in the XY plane while gently
oscillating in Z.

Usage:
    python demo_transport_lissajous.py headless=false steps=10000
    python demo_transport_lissajous.py headless=true steps=5000
"""

import os
import math

import hydra
import torch
from omegaconf import OmegaConf
from omni_drones import init_simulation_app

from tensordict import TensorDict


@hydra.main(version_base=None, config_path=".", config_name="demo")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.envs.transport.utils import TransportationGroup, TransportationCfg
    from omni_drones.robots.drone import MultirotorBase

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )

    drone_model_cfg = cfg.drone_model
    drone, controller = MultirotorBase.make(
        drone_model_cfg.name, drone_model_cfg.controller, cfg.sim.device
    )

    group_cfg = TransportationCfg(num_drones=4)
    group = TransportationGroup(drone=drone, cfg=group_cfg)
    group.spawn(translations=[(0, 0, 1.5)])

    scene_utils.design_scene()
    sim.reset()
    group.initialize()
    init_poses = group.get_world_poses(True)
    init_joint_pos = group.get_joint_positions(True)

    # Lissajous (figure-eight) trajectory parameters
    # x(t) = Ax * sin(a*t)
    # y(t) = Ay * sin(b*t)
    # z(t) = z0 + Az * sin(c*t)
    Ax = 3.0       # X amplitude
    Ay = 2.0       # Y amplitude
    Az = 0.5       # Z oscillation amplitude
    z0 = 3.0       # base height
    a = 1.0        # X frequency
    b = 2.0        # Y frequency (2x for figure-eight)
    c = 0.5        # Z frequency (slow bob)
    speed = 0.3    # angular speed multiplier

    # Drone offsets within the formation (relative to group center)
    drone_offsets = torch.tensor([
        [ 0.75,  0.5, 0.0],
        [ 0.75, -0.5, 0.0],
        [-0.75, -0.5, 0.0],
        [-0.75,  0.5, 0.0],
    ], device=sim.device)

    def compute_trajectory(t: float) -> torch.Tensor:
        """Compute target positions for all 4 drones at time t."""
        # Group center follows the Lissajous curve
        cx = Ax * math.sin(a * t)
        cy = Ay * math.sin(b * t)
        cz = z0 + Az * math.sin(c * t)
        center = torch.tensor([cx, cy, cz], device=sim.device)
        # Each drone maintains its formation offset
        return drone_offsets + center.unsqueeze(0)

    from tqdm import tqdm
    t = tqdm(range(cfg.steps))
    for i in t:
        if sim.is_stopped():
            break
        if not sim.is_playing():
            continue

        # Compute time-varying target positions
        sim_time = i * 0.01 * speed
        ref_pos = compute_trajectory(sim_time)

        drone_state = drone.get_state(False)[..., :13].squeeze(0)
        action = controller.compute(drone_state, target_pos=ref_pos)
        drone.apply_action(action)
        sim.step(i % 2 == 0)

    simulation_app.close()


if __name__ == "__main__":
    main()
