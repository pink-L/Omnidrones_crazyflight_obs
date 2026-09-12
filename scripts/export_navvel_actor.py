#!/usr/bin/env python
"""Export a NavVel (CFB ctrlSync-1) PPO actor as TorchScript + 对拍 test vectors.

Runs ON THE TRAINING SERVER inside the OmniDrones repo checkout (must be able to
`import omni_drones` WITHOUT Isaac Sim; verified with the `lz_env` conda python).

Usage (server):
  cd /home/lz/lzspace/drones/OmniDrones
  /home/hybrid/miniconda3/envs/lz_env/bin/python scripts/export_navvel_actor.py \
      --checkpoint /home/lz/lzspace/drones/OmniDrones/scripts/wandb/run-20260909_191309-hpzc2m3r/files/checkpoint_final.pt \
      --run-dir   /home/lz/lzspace/drones/OmniDrones/scripts/wandb/run-20260909_191309-hpzc2m3r \
      --outdir    /tmp/navvel_export

Outputs in --outdir:
  navvel_actor.ts         TorchScript actor: input [1,62] fp32 -> output [1,4] fp32 (actor MEAN loc)
  obs_test.npy            (N,62) fixed observation inputs (structured, deterministic)
  action_test_server.npy  (N,4)  actor mean computed by the REAL training PPOPolicy under
                                 ExplorationType.MODE  (authoritative reference for the NUC 对拍)
  cbf_test.npz            CBF filter reference vectors produced by the server's
                                 omni_drones.utils.cbf.filter_velocity (fp32, same as sim)
  meta.json               checkpoint sha256 + the exact geometry/control constants that the
                                 NUC config must match (auto-read from run config.yaml when given)

Also self-checks:  traced TorchScript vs reference action  max-abs-diff < 1e-5
                  actor state_dict strict-load over the 'actor.' prefix
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------- obs builder
# Mirrors nav_vel._compute_state_and_obs ordering ONLY for the purpose of producing
# realistic, structurally-valid fixed inputs. (The authoritative NUC implementation is
# cross-checked against this same file layout at deployment time.)
OBS_NORM_DIST = 5.0     # obstacle obs_dist_norm
OBS_NORM_RAD = 0.5      # obstacle obs_radius_norm
K_SLOTS = 8
MAX_STEPS = 1500


def quat_wxyz_from_rpy(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2.0)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2.0)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])   # wxyz


def quat_rotate(q, v):
    """q wxyz, v (3,)"""
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v * (2.0 * w ** 2 - 1.0) + 2.0 * w * np.cross(qv, v) + 2.0 * qv * np.dot(qv, v)


def build_obs(pos, target_pos, rpy, vel_w, omega_w, throttle01, target_rpy, step,
              obstacle_centers, obstacle_radii):
    """Return (62,) in the exact sim order:
    [rpos(3), quat_wxyz(4), vel_w6(lin_w3,ang_w3), heading(3), up(3), throttle*2-1(4),
     rheading(3), time_enc(4), obstacle_block(32)]
    """
    q = quat_wxyz_from_rpy(*rpy)
    heading = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    up = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    tq = quat_wxyz_from_rpy(*target_rpy)
    t_heading = quat_rotate(tq, np.array([1.0, 0.0, 0.0]))
    parts = [target_pos - pos, q, np.concatenate([vel_w, omega_w]), heading, up,
             np.asarray(throttle01) * 2.0 - 1.0, t_heading - heading]
    t = np.full(4, step / float(MAX_STEPS))
    parts.append(t)
    # obstacle block: nearest K sorted ascending, scaled, clamp; pad zeros
    d = np.linalg.norm(np.asarray(obstacle_centers) - pos, axis=-1)
    order = np.argsort(d)[:K_SLOTS]
    block = np.zeros((K_SLOTS, 4), dtype=np.float64)
    for i, idx in enumerate(order):
        rvec = (np.asarray(obstacle_centers)[idx] - pos) / OBS_NORM_DIST
        block[i, :3] = np.clip(rvec, -1.0, 1.0)
        block[i, 3] = np.clip(np.asarray(obstacle_radii)[idx] / OBS_NORM_RAD, 0.0, 1.0)
    parts.append(block.reshape(-1))
    return np.concatenate(parts)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-dir", default=None, help="wandb run dir with config.yaml")
    ap.add_argument("--outdir", default="/tmp/navvel_export")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) checkpoint sha256
    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    print(f"[export] checkpoint sha256 = {sha}")

    torch.manual_seed(0)
    np.random.seed(0)

    # 2) build the REAL PPOPolicy used by training/eval (no Isaac required)
    #     (run with cwd = the OmniDrones repo root so omni_drones resolves)
    import os
    cwd = Path.cwd()
    sys.path.insert(0, str(cwd))
    from omni_drones.utils.torchrl.compat import CompositeSpec, UnboundedContinuousTensorSpec
    from omni_drones.learning.ppo.ppo import PPOPolicy, PPOConfig

    obs_spec = CompositeSpec({"agents": CompositeSpec(
        {"observation": UnboundedContinuousTensorSpec((1, 62), device="cpu")})})
    act_spec = CompositeSpec({"agents": CompositeSpec(
        {"action": UnboundedContinuousTensorSpec((1, 4), device="cpu")})})
    rew_spec = CompositeSpec({"agents": CompositeSpec(
        {"reward": UnboundedContinuousTensorSpec((1, 1), device="cpu")})})
    policy = PPOPolicy(PPOConfig(name="ppo"), obs_spec, act_spec, rew_spec,
                       device=torch.device("cpu"))
    sd = torch.load(ck_path, map_location="cpu", weights_only=False)
    actor_keys = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    missing, unexpected = policy.actor.load_state_dict(actor_keys, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    print("[export] actor strict-load OK  ({} keys)".format(len(actor_keys)))
    policy.eval()

    # inner callable obs -> (loc, scale)
    net = policy.actor.module[0].module     # nn.Sequential(make_mlp([256,256,256]), Actor(4))

    # 3) build fixed, deterministic & structurally-valid test observations (N,62)
    rng = np.random.default_rng(7)
    # a representative pillar+free-ball layout (env frame, same stacking rule as sim)
    pillar_layers_z = np.linspace(0.4 + 0.354, 2.6 - 0.354, 4)
    pillar_xy = [(-1.6, -1.2), (0.4, 1.8), (1.9, -0.9), (-0.5, 0.6)]
    obs_c = [np.array([x, y, z]) for (x, y) in pillar_xy for z in pillar_layers_z]
    obs_r = [0.354] * 16
    free_centers = [np.array(c) for c in
                    [(-0.2, -2.0, 1.2), (2.2, 1.0, 0.9), (-2.3, 0.9, 1.6), (0.9, -1.5, 1.9),
                     (1.2, 2.2, 1.3), (-1.1, 2.4, 2.0), (2.6, -1.8, 1.1), (-2.0, -2.4, 1.8),
                     (0.3, 0.2, 2.3), (-2.5, -0.4, 0.8), (2.3, -0.2, 2.1), (0.0, 2.0, 0.7)]]
    obs_c += free_centers
    obs_r += list(rng.choice([0.2, 0.3, 0.4], 12))
    obs_c = np.array(obs_c)
    obs_r = np.array(obs_r)
    assert len(obs_c) == 28

    goal = np.array([2.8, 0.0, 1.0])
    cases = []
    # case 0: all-zero obs
    cases.append(np.zeros(62))
    # case 1: start hover, t=0
    cases.append(build_obs(np.array([-2.8, 0.0, 0.5]), goal, (0.0, 0.0, 0.0),
                           (0, 0, 0), (0, 0, 0), [0.55] * 4, (0, 0, 0.0), 0, obs_c, obs_r))
    # case 2: start hover, t=T (time enc = 1)
    cases.append(build_obs(np.array([-2.8, 0.0, 0.5]), goal, (0.0, 0.0, 0.0),
                           (0, 0, 0), (0, 0, 0), [0.55] * 4, (0, 0, 0.0), MAX_STEPS, obs_c, obs_r))
    # case 3: near a pillar, moving
    cases.append(build_obs(np.array([-1.4, -1.1, 1.0]), goal, (0.0, 0.0, 0.6),
                           (1.0, 0.2, 0.0), (0.0, 0.0, 0.0), [0.60] * 4, (0, 0, 0.0),
                           500, obs_c, obs_r))
    # case 4: goal side approach with non-zero yaw target
    cases.append(build_obs(np.array([2.0, 0.0, 0.9]), goal, (0.02, -0.02, 0.0),
                           (0.5, 0.0, 0.0), (0.0, 0.0, 0.0), [0.57] * 4, (0, 0, np.pi),
                           900, obs_c, obs_r))
    # cases 5..11: random valid states
    for _ in range(7):
        pos = rng.uniform([-2.8, -2.8, 0.4], [2.8, 2.8, 2.4])
        rpy = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0, 2 * np.pi)]
        vel = rng.uniform(-1.5, 1.5, 3)
        omg = rng.uniform(-2.0, 2.0, 3)
        thr = rng.uniform(0.4, 0.7, 4)
        ty = rng.uniform(0, 2 * np.pi)
        cases.append(build_obs(pos, goal, rpy, vel, omg, thr, (0, 0, ty),
                               int(rng.integers(0, MAX_STEPS + 1)), obs_c, obs_r))
    obs_test = np.stack(cases).astype(np.float32)          # (N,62)
    N = obs_test.shape[0]
    print(f"[export] built {N} structured test observations")

    # 4) authoritative reference actions under the real policy in MODE (mean)
    from tensordict import TensorDict
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    td = TensorDict({"agents": {"observation": torch.from_numpy(obs_test).unsqueeze(1)}}, [N])
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        policy(td)
    action_ref = td[("agents", "action")].squeeze(1).numpy()   # (N,4) = actor mean loc
    print("[export] reference action stats: min {:.3f} max {:.3f}".format(
        action_ref.min(), action_ref.max()))

    # 5) export TorchScript: obs (B,62) -> actor mean loc (B,4)
    class ActorExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            loc, _scale = self.net(x)
            return loc

    wrapped = ActorExport(net).eval()
    example = torch.randn(1, 62)
    ts = torch.jit.trace(wrapped, example)
    # self-check: trace == reference on obs_test
    ts_out = ts(torch.from_numpy(obs_test)).detach().numpy()
    diff = np.abs(ts_out - action_ref).max()
    print(f"[export] TorchScript vs reference max-abs-diff = {diff:.3e}")
    assert diff < 1e-5, "TorchScript export mismatch!"
    ts.save(str(outdir / "navvel_actor.ts"))

    # 6) save test vectors
    np.save(outdir / "obs_test.npy", obs_test)
    np.save(outdir / "action_test_server.npy", action_ref)

    # 7) CBF reference vectors (server sim fp32 filter_velocity)
    from omni_drones.utils.cbf import filter_velocity
    # geometry identical to the deployed model (brake OFF):
    drone_radius, inflation = 0.15, 0.05
    margin = 0.1
    r_cbf = obs_r + drone_radius + inflation + margin
    pos = np.array([-1.4, -1.1, 1.0])
    v_nom = np.array([1.5, -0.2, 0.3])
    p_obs = torch.from_numpy(obs_c).float().unsqueeze(0)
    r_safe_t = torch.from_numpy(r_cbf).float().unsqueeze(0)
    act_t = torch.ones(1, len(obs_c), dtype=torch.bool)
    def run_case(p, v):
        v_s, fix = filter_velocity(torch.from_numpy(p).float().unsqueeze(0),
                                   torch.from_numpy(v).float().unsqueeze(0),
                                   p_obs, r_safe_t, act_t, alpha=1.0, iterations=3)
        return v_s[0].numpy(), fix[0].item()
    cases_pv = [(np.array([-1.4, -1.1, 1.0]), np.array([1.5, -0.2, 0.3])),
                (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
                (np.array([1.9, -0.9, 1.0]), np.array([0.0, 2.0, 0.0])),
                (np.array([-2.8, 0.0, 0.5]), np.array([1.8, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.4]), np.array([-1.8, 0.0, 0.0]))]
    vs = []
    fixs = []
    for p, v in cases_pv:
        vs_, fix_ = run_case(p, v)
        vs.append(vs_)
        fixs.append(fix_)
    np.savez(outdir / "cbf_test.npz",
             drone_radius=np.float32(drone_radius), inflation=np.float32(inflation),
             margin=np.float32(margin), alpha=np.float32(1.0), iterations=np.int32(3),
             p_obs=obs_c.astype(np.float32), r_cbf=r_cbf.astype(np.float32),
             pos=np.array([c[0] for c in cases_pv]).astype(np.float32),
             v_nom=np.array([c[1] for c in cases_pv]).astype(np.float32),
             v_safe=np.stack(vs).astype(np.float32),
             fix_norm=np.array(fixs).astype(np.float32))
    print("[export] saved cbf_test.npz with {} cases".format(len(cases_pv)))

    # 8) meta.json (constants the NUC config must match + provenance)
    meta = {
        "checkpoint": str(ck_path),
        "sha256": sha,
        "model": "CFB ctrlSync-1 (run-20260909_191309-hpzc2m3r) dual-CBF PPO",
        "actor": {"obs_dim": 62, "action_dim": 4, "hidden": [256, 256, 256],
                  "activ": "leaky_relu(0.01) + layernorm per hidden layer",
                  "inference": "deterministic mean (no sample, no obs-norm, no tanh)",
                  "torchscript_input": [1, 62], "torchscript_output": [1, 4]},
        "geometry": {"drone_radius": 0.15, "inflation": 0.05,
                     "collision_margin": 0.05, "danger_radius": 0.6,
                     "obs_dist_norm": 5.0, "obs_radius_norm": 0.5,
                     "K_slots": 8, "arrive_radius": 0.5, "arrive_hold_steps": 50,
                     "arena_bound_xy": 3.0, "z_min": 0.15, "z_max": 3.0,
                     "max_episode_length": 1500,
                     "pillar_radius": 0.354, "pillar_z_lo": 0.4, "pillar_z_hi": 2.6,
                     "radius_choices": [0.2, 0.3, 0.4]},
        "cbf": {"alpha": 1.0, "r_safety_margin": 0.1, "use_brake_term": False,
                "a_max": 2.0, "max_vel": 1.8, "filter_iterations": 3,
                "r_si": "drone_radius + r_o + inflation",
                "r_cbf": "r_si + r_safety_margin (brake off)",
                "projection": "translational only, yaw pass-through",
                "pre_clamp": "component clamp +-1.8 (CBFVelocityFilter)",
                "post_magnitude_clamp": "<=1.8 direction-preserving (VelController)"},
        "action_post": {"vel": "world-frame m/s", "yaw_abs": "clamp(a3,+-1.5/pi)*pi",
                        "yaw_limit_rad": 1.5},
        "obs_time_encoding": "step/1500 in all 4 channels",
        "throttle_obs": "2*(rotor_throttle)-1, rotor_throttle in [0,1]",
    }
    if args.run_dir:
        cfgp = Path(args.run_dir) / "config.yaml"
        if cfgp.exists():
            overrides = [ln.strip()[2:].strip() for ln in cfgp.read_text().splitlines()
                         if ln.strip().startswith("- task.")]
            meta["run_task_overrides"] = overrides
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[export] wrote meta.json")
    print("[export] DONE ->", outdir)


if __name__ == "__main__":
    main()
