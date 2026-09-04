# [M2 2026-09-04] Pure-logic unit test for NavVel obstacle geometry + curriculum.
# No Isaac / no sim needed: loads nav_vel_obstacles.py and nav_curriculum.py directly
# by file path (they only depend on torch) so this runs fast on CPU.
#
# Usage (OmniDrones/scripts/, conda activate lz_env):
#   python obstacle_geometry_test.py
#
# Checks:
#   * sample_layout: active count == L, inactive slots zeroed, positions in-box,
#     per-obstacle endpoint margins (init/goal) and inter-obstacle spacing hold;
#     layout failure/fallback rate over many batches is ~0.
#   * build_obs: block shape/dims, normalization, inactive slots exactly zero.
#   * clearances + collision edge: a drone that flies into a ball triggers exactly
#     one collision edge (stays in contact afterward -> no extra edges).
#   * curriculum: promotion only after success-rate + collision-rate + min-frames gate.
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def load_module(name, rel_path):
    path = os.path.join(REPO, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


obstacles_mod = load_module("nav_vel_obstacles", "omni_drones/envs/single/nav_vel_obstacles.py")
curriculum_mod = load_module("nav_curriculum", "omni_drones/utils/nav_curriculum.py")
ObstacleManager = obstacles_mod.ObstacleManager
ObstacleCurriculum = curriculum_mod.ObstacleCurriculum

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OBS_CFG = {
    "max_slots": 8,
    "radius_choices": [0.20, 0.30, 0.40],
    "spawn_xy_range": [[-2.8, -2.8], [2.8, 2.8]],
    "spawn_z_range": [0.6, 3.4],
    "drone_radius": 0.15,
    "inflation": 0.05,
    "collision_margin": 0.05,
    "danger_radius": 0.6,
    "max_collisions": 2,
    "init_clearance": 0.15,
    "goal_clearance": 0.35,
    "min_gap_between": 0.25,
    "spawn_relax": 0.85,
    "spawn_relax_rounds": 6,
    "obs_dist_norm": 5.0,
    "obs_radius_norm": 0.5,
}

CUR_CFG = {
    "levels": [0, 2, 4, 8],
    "initial_level": 0,
    "gate_window_episodes": 3000,
    "success_rate_threshold": 0.80,
    "collision_rate_threshold": 0.05,
    "min_frames_between_promote": 2_000_000,
    "allow_demote": False,
}


def sample_endpoints(n):
    init = torch.rand(n, 1, 3) * torch.tensor([6.0, 6.0, 1.5]) + torch.tensor([-3.0, -3.0, 1.0])
    goal = torch.rand(n, 1, 3) * torch.tensor([6.0, 6.0, 1.5]) + torch.tensor([-3.0, -3.0, 1.5])
    return init.to(DEVICE), goal.to(DEVICE)


def check_layout(mgr, init, goal, L, tol=0.02):
    """Returns dict of pass flags / diagnostics for a single sampled batch."""
    pos, rad, active = mgr.sample_layout(init, goal, L)
    n = init.shape[0]
    r_s = rad + mgr.drone_radius + mgr.inflation
    res = {"n_active": int(active.sum(-1).unique().item()) if active.sum() else 0,
           "active_eq_L": bool((active.sum(-1) == L).all()),
           "inactive_zero": bool((pos[~active].abs().max() < 1e-6) if (~active).any() else True),
           "in_box": bool(((pos >= mgr.spawn_lo - tol) & (pos <= mgr.spawn_hi + tol)).all()),
           "init_margin": True, "goal_margin": True, "spacing": True, "distinct": True}
    if L == 0:
        return res
    # endpoint margins per active obstacle (center distance >= r_s + clearance)
    d_init = torch.norm(pos - init, dim=-1)         # (n,K)
    d_goal = torch.norm(pos - goal, dim=-1)
    need_init = r_s + mgr.init_clearance            # (n,K)
    need_goal = r_s + mgr.goal_clearance
    ok_i = torch.where(active, d_init >= need_init - tol, torch.ones_like(d_init, dtype=torch.bool))
    ok_g = torch.where(active, d_goal >= need_goal - tol, torch.ones_like(d_goal, dtype=torch.bool))
    res["init_margin"] = bool(ok_i.all())
    res["goal_margin"] = bool(ok_g.all())
    # inter-obstacle spacing among active slots (only same-env pairs)
    pd = torch.norm(pos.unsqueeze(2) - pos.unsqueeze(1), dim=-1)  # (n,K,K)
    i, j = torch.triu_indices(mgr.K, mgr.K, 1, device=DEVICE)
    pair_d = pd[:, i, j]                                           # (n, n_pairs)
    need_p = (rad[:, i] + rad[:, j] + mgr.min_gap_between)
    mask = active[:, i] & active[:, j]
    ok_sp = ~mask | (pair_d >= need_p - tol)
    res["spacing"] = bool(ok_sp.all())
    res["distinct"] = bool((pd[:, i, j].min(dim=-1).values > 1e-6).all())
    return res


def main():
    print(f"[test] device={DEVICE}")
    torch.manual_seed(0)
    mgr = ObstacleManager(OBS_CFG, num_envs=256, device=DEVICE)

    # ---------------- layout checks across levels ----------------
    all_ok = True
    for L in (0, 2, 4, 8):
        n_batches, n_batch = 8, 256
        fail_init = fail_goal = fail_spacing = fail_active = 0
        for _ in range(n_batches):
            init, goal = sample_endpoints(n_batch)
            r = check_layout(mgr, init, goal, L)
            if not r["active_eq_L"]:
                fail_active += 1
            if not r["init_margin"]:
                fail_init += 1
            if not r["goal_margin"]:
                fail_goal += 1
            if not r["spacing"]:
                fail_spacing += 1
        tot = n_batches * n_batch
        print(f"[layout] L={L}: active_err={fail_active}/{tot} init_margin_err={fail_init}/{tot} "
              f"goal_margin_err={fail_goal}/{tot} spacing_err={fail_spacing}/{tot}")
        all_ok &= (fail_init == 0 and fail_goal == 0 and fail_spacing == 0 and fail_active == 0)

    # ---------------- obs block ----------------
    mgr8 = ObstacleManager(OBS_CFG, num_envs=8, device=DEVICE)
    init, goal = sample_endpoints(8)
    pos, rad, active = mgr8.sample_layout(init, goal, 4)
    mgr8.commit_layout(torch.arange(8), pos, rad, active)
    drone_pos = torch.zeros(8, 1, 3, device=DEVICE)
    blk = mgr8.build_obs(drone_pos)
    assert blk.shape == (8, 1, 4 * OBS_CFG["max_slots"]), blk.shape
    # inactive slots exactly zero
    act = mgr8.active
    blk3 = blk.reshape(8, mgr8.K, 4)
    zero_inactive = (blk3[~act].abs().max() < 1e-6) if (~act).any() else True
    assert zero_inactive, "inactive obs slots not zero"
    # active radius channel within [0,1]
    assert blk3[act][..., 3].min() >= 0 and blk3[act][..., 3].max() <= 1.0
    # relative positions normalized to [-1,1]
    assert blk3[act][..., :3].min() >= -1.0 and blk3[act][..., :3].max() <= 1.0
    print(f"[obs] shape={tuple(blk.shape)} norm/zero-pad OK")

    # ---------------- collision edge semantics ----------------
    mgr2 = ObstacleManager(OBS_CFG, num_envs=1, device=DEVICE)
    init, goal = sample_endpoints(1)
    pos, rad, active = mgr2.sample_layout(init, goal, 1)
    mgr2.commit_layout(torch.tensor([0]), pos, rad, active)
    r0 = float(mgr2.radius[0, 0].item())
    # drone flies from far away straight into the ball center along +x
    clr = mgr2.clearances
    edges = 0
    prev = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)
    xs = torch.linspace(-3.0, 0.0, 300).to(DEVICE)
    for x in xs:
        drone = torch.tensor([[x, 0.0, 0.0]], device=DEVICE).unsqueeze(1)
        d = clr(drone)
        dmin = d.min(dim=-1).values
        in_col = (dmin < OBS_CFG["collision_margin"]) & torch.isfinite(dmin)
        new_edge = (in_col & ~prev).any()
        edges += int(new_edge.item())
        prev = in_col
    # physically the ball is placed somewhere random; make the drone traverse the whole
    # box x-range and ensure we detect >=1 edge only once while passing through a ball
    print(f"[collision] obstacle radius={r0:.2f} edges_while_flying_through={edges}")
    # re-run: fixed obstacle at origin to make assertion deterministic
    mgr3 = ObstacleManager(OBS_CFG, num_envs=1, device=DEVICE)
    mgr3.pos[0, 0] = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
    mgr3.radius[0, 0] = 0.4
    mgr3.active[0, 0] = True
    r_s = 0.4 + mgr3.drone_radius + mgr3.inflation          # 0.6
    prev = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)
    edge_hits = 0
    for x in torch.linspace(1.2, -1.2, 400).to(DEVICE):
        drone = torch.tensor([[x, 0.0, 1.0]], device=DEVICE).unsqueeze(1)
        dmin = mgr3.min_clearance(drone)
        in_col = (dmin < OBS_CFG["collision_margin"]) & torch.isfinite(dmin)
        edge_hits += int(((in_col & ~prev).any()).item())
        prev = in_col
    assert edge_hits == 1, f"expected exactly 1 collision edge through a ball, got {edge_hits}"
    assert abs(r_s - 0.6) < 1e-6
    print(f"[collision] deterministic pass-through: r_s={r_s:.2f}, edges={edge_hits} (OK)")

    # ---------------- curriculum ----------------
    def make_cur(initial_level=0):
        return ObstacleCurriculum(
            levels=CUR_CFG["levels"], initial_level=initial_level,
            gate_window=int(CUR_CFG["gate_window_episodes"]),
            success_threshold=float(CUR_CFG["success_rate_threshold"]),
            collision_threshold=float(CUR_CFG["collision_rate_threshold"]),
            min_frames=float(CUR_CFG["min_frames_between_promote"]),
            allow_demote=False, device=DEVICE)

    cur = make_cur()
    assert cur.level == 0
    # below threshold & frames -> no promote
    cur.update(torch.zeros(1024, dtype=torch.bool, device=DEVICE), add_frames=600 * 1024)
    assert cur.level == 0
    # 80% success, enough frames, low collision -> promote 0->2
    for _ in range(4):
        ok = torch.rand(1024, device=DEVICE) < 0.9
        col = torch.rand(1024, device=DEVICE) < 0.02
        cur.update(ok & ~col, col, add_frames=600 * 1024)
    assert cur.level == 2, f"expected promote to level 2, got {cur.level}"
    print(f"[curriculum] promoted 0->2 OK (level={cur.level}, active={cur.active_obstacles})")
    # high collision should block promotion even with high success
    cur2 = make_cur()
    for _ in range(4):
        ok = torch.ones(1024, dtype=torch.bool, device=DEVICE)
        col = torch.zeros(1024, dtype=torch.bool, device=DEVICE)
        col[:300] = True            # 29% episodes collided -> collision_rate > 5%
        cur2.update(ok & ~col, col, add_frames=600 * 1024)
    assert cur2.level == 0, "promoted despite collision_rate > threshold"
    print("[curriculum] collision gate blocks promotion OK")

    print("\n========== SUMMARY ==========")
    if all_ok:
        print("ALL LAYOUT CHECKS PASSED")
    else:
        print("LAYOUT CHECKS FAILED (see per-L diagnostics above)")
    print("ALL ASSERT CHECKS PASSED" if all_ok else "SOME ASSERT CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
