#!/usr/bin/env python
"""Export a NavVel PPO actor as TorchScript + 对拍 test vectors.  [v2, config-driven]

Runs inside the OmniDrones repo checkout with only CPU torch/torchrl/tensordict
(no Isaac Sim needed).  Verified with the `lz_env` conda python.

v1 hard-coded the whole v1.0.0 geometry (0.15/0.05/0.10 radius chain, 4x4 pillars
at fixed xy, K=8, T=1500, ...).  v2 derives **every** geometry/control constant and
the test-obstacle layout from a frozen config (a `cfg/profiles/*.yaml` snapshot or
a run's resolved `config.yaml`), because the geometry profile changes between
batches (P0.2 口径A, A1a..A4) and plan §2.4 makes re-export a red-line action for
any radius-chain change.

Test-obstacle layout: sampled with the env's own `ObstacleManager.sample_layout`
(CPU) so the exported vectors always describe a **valid, sim-consistent** scene
(pillar stacking rule = the real one) instead of a hand-written layout.

Usage:
  python scripts/export_navvel_actor.py \
      --checkpoint <run>/files/checkpoint_final.pt \
      --profile    cfg/profiles/A.yaml \
      --outdir     /tmp/navvel_export \
      --model-id   navvel-cfb-v1.1.0-dual-p1-s11

  # v1.0.0 regression (accepts a run config.yaml too):
  python scripts/export_navvel_actor.py --checkpoint <ckpt> \
      --profile <run>/files/config.yaml --outdir /tmp/x \
      --model-id navvel-cfb-v1.0.0-dual-p1-s11

Outputs in --outdir: navvel_actor.ts, obs_test.npy, action_test_server.npy,
cbf_test.npz, meta.json   (schemas unchanged from v1; new keys added only).

Self-checks: traced TorchScript vs the real policy's MODE mean  max|Δ| < 1e-5;
             actor state_dict strict-load over the 'actor.' prefix.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------------- config
def _unwrap(v):
    """wandb's config.yaml wraps every entry as `{value: <literal>}`."""
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return v["value"]
    return v


def load_task_cfg(path: Path):
    """Accept a frozen profile (nested task config) OR a wandb run config.yaml
    (flat `task.a.b` keys, values {value:}-wrapped).  Returns the task dict."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"[export] {path} is not a mapping")
    if any(k.startswith("task.") for k in raw):
        task = {}
        for k, v in raw.items():
            if not k.startswith("task."):
                continue
            node = task
            parts = k[len("task."):].split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _unwrap(v)
        return task
    return raw


def load_geo(task: dict):
    """Extract every geometry/control constant the export + deploy side needs."""
    ob = dict(task.get("obstacle", {}) or {})
    cb = dict(task.get("cbf", {}) or {})
    vl = dict(task.get("vel_limit", {}) or {})
    env = dict(task.get("env", {}) or {})

    K = int(ob.get("max_slots", 8))
    obs_safety = str(ob.get("obs_safety", "none"))
    if obs_safety != "none":
        raise SystemExit(
            f"[export] obs_safety={obs_safety!r} is not supported by this exporter "
            "(the safety channels are appended by the env, not mirrored here)")
    obs_safety_dim = 0

    drone_r = float(ob.get("drone_radius", 0.15))
    infl = float(ob.get("inflation", 0.05))
    margin = float(cb.get("r_safety_margin", 0.1))
    use_brake = bool(cb.get("use_brake_term", False))
    a_max = float(cb.get("a_max", 2.0))
    v_max = float(cb.get("max_vel", None) or vl.get("max_vel", 1.8))
    brake = (v_max ** 2 / (2.0 * a_max)) if (use_brake and a_max > 0) else 0.0
    cbf_extra = margin + brake

    max_steps = int(env.get("max_episode_length", 1500))
    return {
        "task_name": str(task.get("name", "NavVel")),
        "K": K,
        "M": int(ob.get("n_pillars", 0)) * max(1, int(ob.get("pillar_layers", 4)))
             + int(ob.get("n_free_obstacles", 0))
             if int(ob.get("n_pillars", 0)) > 0
             else int(ob.get("num_scene") or K),
        "obs_dim": 30 + 4 * K + obs_safety_dim,
        "obs_safety": obs_safety,
        "max_steps": max_steps,
        "obs_dist_norm": float(ob.get("obs_dist_norm", 5.0)),
        "obs_radius_norm": float(ob.get("obs_radius_norm", 0.5)),
        "drone_radius": drone_r,
        "inflation": infl,
        "collision_margin": float(ob.get("collision_margin", 0.05)),
        "danger_radius": float(ob.get("danger_radius", 0.6)),
        "arrive_radius": float(task.get("arrive_radius", 0.5)),
        "arrive_hold_steps": int(task.get("arrive_hold_steps", 50)),
        "arena_bound_xy": [float(x) for x in (task.get("arena_bound") or [3.0, 3.0])],
        "z_min": float(task.get("z_min", 0.15)),
        "z_max": float(task.get("z_max", 3.0)),
        "radius_choices": [float(x) for x in (ob.get("radius_choices") or [0.3])],
        "pillar_radius": float(ob.get("pillar_radius", 0.354)),
        "pillar_z_lo": float(ob.get("pillar_z_lo", 0.4)),
        "pillar_z_hi": float(ob.get("pillar_z_hi", 2.6)),
        "n_pillars": int(ob.get("n_pillars", 0)),
        "pillar_layers": max(1, int(ob.get("pillar_layers", 4))),
        "n_free_obstacles": int(ob.get("n_free_obstacles", 0)),
        "action_transform": str(task.get("action_transform", "velocity")),
        "vel_limit": {"max_vel": vl.get("max_vel", None),
                      "max_yaw_rate": vl.get("max_yaw_rate", None)},
        "time_encoding": bool(task.get("time_encoding", True)),
        "cbf": {
            "mode": str(cb.get("mode", "none")),
            "alpha": float(cb.get("alpha", 1.0)),
            "r_safety_margin": margin,
            "use_brake_term": use_brake,
            "a_max": a_max,
            "max_vel": v_max,
            "filter_iterations": int(cb.get("filter_iterations", 3)),
            "filter_grad": str(cb.get("filter_grad", "detach")),
            "penalty_intrude": bool(cb.get("penalty_intrude", True)),
            "penalty_src": str(cb.get("penalty_src", "nominal")),
            "reward_weight": float(cb.get("reward_weight", 0.5)),
            "correction_weight": float(cb.get("correction_weight", 0.0)),
            "correction_sigma": float(cb.get("correction_sigma", 0.5)),
            "h_penalty_weight": float(cb.get("h_penalty_weight", 0.0)),
            "h_penalty_buffer": float(cb.get("h_penalty_buffer", 0.0)),
            "r_si": "drone_radius + r_o + inflation",
            "r_cbf": ("r_si + r_safety_margin (+ v_max^2/(2 a_max) if use_brake_term)"),
        },
        "cbf_extra": cbf_extra,
        "_obstacle_cfg": ob,
    }


# ----------------------------------------------------------------------------- obs builder
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


def build_obs(G, pos, target_pos, rpy, vel_w, omega_w, throttle01, target_rpy, step,
              obstacle_centers, obstacle_radii):
    """Return (obs_dim,) in the exact sim order:
    [rpos(3), quat_wxyz(4), vel_w6(lin_w3,ang_w3), heading(3), up(3), throttle*2-1(4),
     rheading(3), time_enc(4), obstacle_block(4K)]
    """
    q = quat_wxyz_from_rpy(*rpy)
    heading = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    up = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    tq = quat_wxyz_from_rpy(*target_rpy)
    t_heading = quat_rotate(tq, np.array([1.0, 0.0, 0.0]))
    parts = [target_pos - pos, q, np.concatenate([vel_w, omega_w]), heading, up,
             np.asarray(throttle01) * 2.0 - 1.0, t_heading - heading]
    parts.append(np.full(4, step / float(G["max_steps"])))
    # obstacle block: nearest K sorted ascending, scaled, clamp; pad zeros
    K = G["K"]
    d = np.linalg.norm(np.asarray(obstacle_centers) - pos, axis=-1)
    order = np.argsort(d)[:K]
    block = np.zeros((K, 4), dtype=np.float64)
    for i, idx in enumerate(order):
        rvec = (np.asarray(obstacle_centers)[idx] - pos) / G["obs_dist_norm"]
        block[i, :3] = np.clip(rvec, -1.0, 1.0)
        block[i, 3] = np.clip(np.asarray(obstacle_radii)[idx] / G["obs_radius_norm"], 0.0, 1.0)
    parts.append(block.reshape(-1))
    return np.concatenate(parts)


def sample_test_layout(G, start, goal, seed):
    """Sample a valid scene with the env's own ObstacleManager (CPU).

    NOTE: loaded via importlib from the source file on purpose.  Importing the
    package path (`omni_drones.envs.single.nav_vel_obstacles`) executes
    `omni_drones/envs/single/__init__.py`, which pulls in Isaac Sim
    (`omni.isaac.core`) and fails without a SimulationApp.  The obstacle manager
    itself is pure torch.  Same technique as scripts/pillar_geometry_test.py.
    """
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    src = repo / "omni_drones" / "envs" / "single" / "nav_vel_obstacles.py"
    spec = importlib.util.spec_from_file_location("nav_vel_obstacles_export", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ObstacleManager = mod.ObstacleManager

    torch.manual_seed(seed)
    mgr = ObstacleManager(G["_obstacle_cfg"], num_envs=1, device="cpu")
    init = torch.as_tensor(start, dtype=torch.float32).reshape(1, 1, 3)
    tgt = torch.as_tensor(goal, dtype=torch.float32).reshape(1, 1, 3)
    pos, rad, act = mgr.sample_layout(init, tgt, mgr.M)
    pos, rad, act = pos[0].numpy(), rad[0].numpy(), act[0].numpy()
    keep = act.astype(bool)
    return pos[keep], rad[keep]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--profile", required=True,
                    help="cfg/profiles/*.yaml snapshot OR a run's files/config.yaml")
    ap.add_argument("--run-dir", default=None, help="wandb run dir (for config provenance)")
    ap.add_argument("--outdir", default="/tmp/navvel_export")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--start", default="-2.8,0.0,0.5", help="test start (x,y,z)")
    ap.add_argument("--goal", default="2.8,0.0,1.0", help="test goal (x,y,z)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cases", type=int, default=7, help="extra random obs cases")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    prof_path = Path(args.profile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    task = load_task_cfg(prof_path)
    G = load_geo(task)
    start = np.array([float(x) for x in args.start.split(",")])
    goal = np.array([float(x) for x in args.goal.split(",")])
    model_name = args.model_name or (
        f"{args.model_id} ({args.checkpoint}) "
        f"{G['cbf']['mode']}-CBF PPO")

    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    print(f"[export] checkpoint sha256 = {sha}")
    print(f"[export] profile={prof_path}")
    print(f"[export] obs_dim={G['obs_dim']} (K={G['K']}, M={G['M']}, obs_safety={G['obs_safety']})"
          f"  r_s=r_o+{G['drone_radius'] + G['inflation']:.2f}"
          f"  r_cbf=r_o+{G['drone_radius'] + G['inflation'] + G['cbf_extra']:.2f}"
          f"  brake={G['cbf']['use_brake_term']}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 2) build the REAL PPOPolicy used by training/eval (no Isaac required)
    import os
    cwd = Path.cwd()
    sys.path.insert(0, str(cwd))
    from omni_drones.utils.torchrl.compat import CompositeSpec, UnboundedContinuousTensorSpec
    from omni_drones.learning.ppo.ppo import PPOPolicy, PPOConfig

    obs_dim = G["obs_dim"]
    obs_spec = CompositeSpec({"agents": CompositeSpec(
        {"observation": UnboundedContinuousTensorSpec((1, obs_dim), device="cpu")})})
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
    net = policy.actor.module[0].module     # nn.Sequential(make_mlp([256,256,256]), Actor(4))

    # 3) build fixed, deterministic & structurally-valid test observations (N, obs_dim)
    obs_c, obs_r = sample_test_layout(G, start, goal, args.seed)
    print(f"[export] sampled test layout: {len(obs_c)} active obstacles"
          f" (n_pillars={G['n_pillars']}x{G['pillar_layers']}, n_free={G['n_free_obstacles']})")
    rng = np.random.default_rng(args.seed + 7)

    cases = []
    cases.append(np.zeros(obs_dim))                                   # all-zero obs
    cases.append(build_obs(G, start, goal, (0, 0, 0), (0, 0, 0), (0, 0, 0),
                           [0.55] * 4, (0, 0, 0.0), 0, obs_c, obs_r))  # hover @ t=0
    cases.append(build_obs(G, start, goal, (0, 0, 0), (0, 0, 0), (0, 0, 0),
                           [0.55] * 4, (0, 0, 0.0), G["max_steps"], obs_c, obs_r))  # t=T
    # near the first pillar, moving
    p0 = obs_c[0]
    cases.append(build_obs(G, p0 + np.array([0.05, 0.05, 0.0]), goal, (0.0, 0.0, 0.6),
                           (1.0, 0.2, 0.0), (0.0, 0.0, 0.0), [0.60] * 4, (0, 0, 0.0),
                           500, obs_c, obs_r))
    # goal-side approach with non-zero yaw target
    cases.append(build_obs(G, goal - np.array([0.8, 0.0, 0.1]), goal, (0.02, -0.02, 0.0),
                           (0.5, 0.0, 0.0), (0.0, 0.0, 0.0), [0.57] * 4, (0, 0, np.pi),
                           900, obs_c, obs_r))
    lo = np.array([-abs(G["arena_bound_xy"][0]) + 0.2, -abs(G["arena_bound_xy"][1]) + 0.2,
                   G["z_min"] + 0.25])
    hi = np.array([abs(G["arena_bound_xy"][0]) - 0.2, abs(G["arena_bound_xy"][1]) - 0.2,
                   G["z_max"] - 0.6])
    for _ in range(int(args.n_cases)):
        pos = rng.uniform(lo, hi)
        rpy = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0, 2 * np.pi)]
        vel = rng.uniform(-1.5, 1.5, 3)
        omg = rng.uniform(-2.0, 2.0, 3)
        thr = rng.uniform(0.4, 0.7, 4)
        ty = rng.uniform(0, 2 * np.pi)
        cases.append(build_obs(G, pos, goal, rpy, vel, omg, thr, (0, 0, ty),
                               int(rng.integers(0, G["max_steps"] + 1)), obs_c, obs_r))
    obs_test = np.stack(cases).astype(np.float32)          # (N, obs_dim)
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

    # 5) export TorchScript: obs (B,obs_dim) -> actor mean loc (B,4)
    class ActorExport(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            loc, _scale = self.net(x)
            return loc

    wrapped = ActorExport(net).eval()
    example = torch.randn(1, obs_dim)
    ts = torch.jit.trace(wrapped, example)
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
    cbf = G["cbf"]
    r_cbf = obs_r + G["drone_radius"] + G["inflation"] + G["cbf_extra"]
    v_lim = float(cbf["max_vel"])
    cases_pv = [
        (np.clip(obs_c[0] + np.array([0.05, 0.05, 0.0]), lo, hi), np.array([1.5, -0.2, 0.3])),
        (np.zeros(3), np.array([1.0, 0.0, 0.0])),
        (np.clip(obs_c[min(1, len(obs_c) - 1)], lo, hi), np.array([0.0, 2.0, 0.0])),
        (start.copy(), np.array([1.8, 0.0, 0.0])),
        (np.array([0.0, 0.0, G["z_min"] + 0.25]), np.array([-1.8, 0.0, 0.0])),
    ]
    p_obs_t = torch.from_numpy(obs_c).float().unsqueeze(0)
    r_safe_t = torch.from_numpy(r_cbf).float().unsqueeze(0)
    act_t = torch.ones(1, len(obs_c), dtype=torch.bool)
    vs, fixs = [], []
    for p, v in cases_pv:
        v_clamped = np.clip(v, -v_lim, v_lim)          # the env clamps BEFORE projecting
        v_s, fix = filter_velocity(
            torch.from_numpy(p).float().unsqueeze(0),
            torch.from_numpy(v_clamped).float().unsqueeze(0),
            p_obs_t, r_safe_t, act_t,
            alpha=cbf["alpha"], iterations=cbf["filter_iterations"])
        vs.append(v_s[0].numpy())
        fixs.append(fix[0].item())
    np.savez(outdir / "cbf_test.npz",
             drone_radius=np.float32(G["drone_radius"]),
             inflation=np.float32(G["inflation"]),
             margin=np.float32(cbf["r_safety_margin"]),
             alpha=np.float32(cbf["alpha"]),
             iterations=np.int32(cbf["filter_iterations"]),
             max_vel=np.float32(v_lim),
             a_max=np.float32(cbf["a_max"]),
             use_brake_term=np.bool_(cbf["use_brake_term"]),
             cbf_extra=np.float32(G["cbf_extra"]),
             p_obs=obs_c.astype(np.float32), r_cbf=r_cbf.astype(np.float32),
             pos=np.array([c[0] for c in cases_pv]).astype(np.float32),
             v_nom=np.array([np.clip(c[1], -v_lim, v_lim) for c in cases_pv]).astype(np.float32),
             v_safe=np.stack(vs).astype(np.float32),
             fix_norm=np.array(fixs).astype(np.float32))
    print("[export] saved cbf_test.npz with {} cases".format(len(cases_pv)))

    # 8) meta.json (constants the NUC config must match + provenance)
    meta = {
        "model_id": args.model_id,
        "checkpoint": str(ck_path),
        "sha256": sha,
        "model": model_name,
        "actor": {"obs_dim": obs_dim, "action_dim": 4, "hidden": [256, 256, 256],
                  "activ": "leaky_relu(0.01) + layernorm per hidden layer",
                  "inference": "deterministic mean (no sample, no obs-norm, no tanh)",
                  "torchscript_input": [1, obs_dim], "torchscript_output": [1, 4]},
        "geometry_profile": str(prof_path),
        "geometry": {"drone_radius": G["drone_radius"], "inflation": G["inflation"],
                     "collision_margin": G["collision_margin"],
                     "danger_radius": G["danger_radius"],
                     "obs_dist_norm": G["obs_dist_norm"],
                     "obs_radius_norm": G["obs_radius_norm"],
                     "K_slots": G["K"], "M_scene": G["M"],
                     "obs_safety": G["obs_safety"],
                     "arrive_radius": G["arrive_radius"],
                     "arrive_hold_steps": G["arrive_hold_steps"],
                     # keep the v1 SCALAR form for the primary key (the deploy side reads it),
                     # and expose the y half-width separately (additive -> backward compatible)
                     "arena_bound_xy": G["arena_bound_xy"][0],
                     "arena_bound_xy_y": G["arena_bound_xy"][1],
                     "z_min": G["z_min"], "z_max": G["z_max"],
                     "max_episode_length": G["max_steps"],
                     "pillar_radius": G["pillar_radius"],
                     "pillar_z_lo": G["pillar_z_lo"], "pillar_z_hi": G["pillar_z_hi"],
                     "radius_choices": G["radius_choices"],
                     "n_pillars": G["n_pillars"], "pillar_layers": G["pillar_layers"],
                     "n_free_obstacles": G["n_free_obstacles"]},
        "cbf": {"mode": cbf["mode"], "alpha": cbf["alpha"],
                "r_safety_margin": cbf["r_safety_margin"],
                "use_brake_term": cbf["use_brake_term"],
                "a_max": cbf["a_max"], "max_vel": cbf["max_vel"],
                "filter_iterations": cbf["filter_iterations"],
                "filter_grad": cbf["filter_grad"],
                "penalty_intrude": cbf["penalty_intrude"],
                "penalty_src": cbf["penalty_src"],
                "reward_weight": cbf["reward_weight"],
                "correction_weight": cbf["correction_weight"],
                "correction_sigma": cbf["correction_sigma"],
                "h_penalty_weight": cbf["h_penalty_weight"],
                "h_penalty_buffer": cbf["h_penalty_buffer"],
                "cbf_extra": G["cbf_extra"],
                # keep the v1 key name `r_si` (deploy side reads it) + an explicit alias
                "r_si": "drone_radius + r_o + inflation",
                "r_s": "drone_radius + r_o + inflation",
                "r_cbf": ("r_si + r_safety_margin (brake off)" if not cbf["use_brake_term"]
                          else "r_si + r_safety_margin + v_max^2/(2 a_max)"),
                "projection": "translational only, yaw pass-through",
                "pre_clamp": f"component clamp +-{v_lim} (CBFVelocityFilter)",
                "post_magnitude_clamp": f"<={v_lim} direction-preserving (VelController)"},
        "action_post": {"vel": "world-frame m/s",
                        "yaw_abs": "clamp(a3,+-1.5/pi)*pi",
                        "yaw_limit_rad": float(G["vel_limit"].get("max_yaw_rate") or 1.5)},
        "obs_time_encoding": (f"step/{G['max_steps']} in all 4 channels"
                              if G["time_encoding"] else "disabled"),
        "throttle_obs": "2*(rotor_throttle)-1, rotor_throttle in [0,1]",
        "test_vectors": {"start": start.tolist(), "goal": goal.tolist(),
                         "seed": int(args.seed), "n_obs": int(N),
                         "layout": "ObstacleManager.sample_layout (CPU, sim-consistent)",
                         "n_active_obstacles": int(len(obs_c))},
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
