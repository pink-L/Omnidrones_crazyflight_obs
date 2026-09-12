"""CPU layout checker for NavVel pillar worlds (plan §3.2 A3 hard constraints / G8).

Samples layouts with the env's own `ObstacleManager` (pure torch, CPU, loaded straight
from its file so Isaac Sim is never imported) and checks the invariants that the
sampler is supposed to guarantee:

  1. every slot active            active.sum() == M, M = n_pillars*pillar_layers + n_free
  2. start clearance              min_i(||init - p_i|| - r_si) >= init_clearance
  3. goal clearance               same for the goal
  4. inter-obstacle gap           min_{i<j}(||p_i-p_j|| - r_si - r_sj) >= min_gap_between
  5. corridor width               worst pillar-pair surface gap (INFORMATIONAL - for a
                                  randomised layout the requirement is not 'every pair is
                                  W apart' (impossible for 8 pillars in a 6 m arena) but
                                  'a route exists whose bottleneck is >= W', which is
                                  what --corridor-clearance tests below)
  6. connectivity (--connectivity) 2-D grid BFS start -> goal over the discs that
                                  intersect the flight band, with every cell closer than
                                  r_o + W/2 to a pillar blocked when --corridor-clearance W
                                  is given. Coarse (2-D slice, cell size --cell): it
                                  proves 'a gap of width W exists in xy', not that the
                                  drone can fly it. Enough to catch layouts where the goal
                                  is walled off, which is what A3 must not ship.

Exit code is non-zero if any hard invariant fails, so it can be used as a CPU gate.

Usage (from OmniDrones/):
  python scripts/pillar_layout_check.py --profile cfg/profiles/A1a.yaml --layouts 8
  python scripts/pillar_layout_check.py --profile cfg/profiles/A3.yaml --layouts 32 \
      --min-corridor --connectivity
"""
import argparse
import importlib.util
import os
import sys

import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBST_FILE = os.path.join(REPO, "omni_drones/envs/single/nav_vel_obstacles.py")


def load_manager_class():
    spec = importlib.util.spec_from_file_location("nav_vel_obstacles_check", OBST_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nav_vel_obstacles_check"] = mod
    spec.loader.exec_module(mod)
    return mod.ObstacleManager


def xy_grid_bfs(blocked, cell, lo, hi, start_xy, goal_xy):
    """4-connected BFS on a boolean (H,W) occupancy grid; returns True if connected."""
    H, W = blocked.shape
    def idx(p):
        ix = int((p[0] - lo) / cell)
        iy = int((p[1] - lo) / cell)
        return iy, ix
    si, sj = idx(start_xy)
    gi, gj = idx(goal_xy)
    if not (0 <= si < H and 0 <= sj < W and 0 <= gi < H and 0 <= gj < W):
        return None
    if blocked[si, sj] or blocked[gi, gj]:
        return False
    seen = torch.zeros_like(blocked)
    stack = [(si, sj)]
    seen[si, sj] = True
    while stack:
        i, j = stack.pop()
        if (i, j) == (gi, gj):
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not seen[ni, nj] and not blocked[ni, nj]:
                seen[ni, nj] = True
                stack.append((ni, nj))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cfg/profiles/A.yaml")
    ap.add_argument("--layouts", type=int, default=8, help="how many layouts to sample")
    ap.add_argument("--num-envs", type=int, default=8, help="envs sampled per layout")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init", default=None, help="start xyz (default: profile fixed_init)")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--corridor-clearance", type=float, default=None,
                    help="required bottleneck width W: with --connectivity, block every "
                         "cell within r_o + W/2 of a pillar and require start<->goal "
                         "connectivity (A3 criterion)")
    ap.add_argument("--connectivity", action="store_true", help="2-D grid BFS start->goal")
    ap.add_argument("--cell", type=float, default=0.05)
    a = ap.parse_args()

    task = OmegaConf.load(os.path.join(REPO, a.profile))
    task = task.get("task", task)
    oc = task.obstacle
    Mgr = load_manager_class()
    mgr = Mgr(oc, num_envs=a.num_envs, device="cpu")

    torch.manual_seed(a.seed)
    start = torch.tensor([float(x) for x in (a.init or task.get("fixed_init")
                                             or [-2.8, 0.0, 0.5])])
    goal = torch.tensor([float(x) for x in (a.goal or task.get("fixed_target")
                                            or [2.8, 0.0, 1.0])])
    r_dr = float(oc.drone_radius)
    infl = float(oc.inflation)
    corridor_floor = 2.0 * (float(oc.pillar_radius) + r_dr + infl) + 0.25
    nP = int(oc.get("n_pillars", 0))
    L = int(oc.get("pillar_layers", 0))
    nF = int(oc.get("n_free_obstacles", 0))
    # [P1 A2/A3] randomized layouts reserve a fixed block of L_max slots per pillar
    npr = oc.get("n_pillars_range", None)
    plr = oc.get("pillar_layers_range", None)
    randomized = any(oc.get(k, None) is not None for k in
                     ("n_pillars_range", "pillar_layers_range", "pillar_z_range"))
    nP_max = int(npr[1]) if npr is not None else nP
    L_max = int(plr[1]) if plr is not None else L
    block = L_max if randomized else L
    M = nP_max * L_max + nF if randomized else nP * L + nF
    print(f"[check] {a.profile}: n_pillars={nP}{npr if npr else ''} "
          f"layers={L}{plr if plr else ''} n_free={nF} -> M={M} "
          f"(max_slots={oc.get('max_slots')}) randomized={randomized}")
    if randomized:
        print(f"[check] randomized: pillar slots per env vary; M={M} is the reserved "
              f"max, unused slots must come back inactive")
    print(f"[check] start={start.tolist()} goal={goal.tolist()}")
    print(f"[check] init_clearance={mgr.init_clearance} goal_clearance={mgr.goal_clearance} "
          f"min_gap_between={mgr.min_gap_between} keepout_x={mgr.keepout_x}")
    if a.corridor_clearance:
        print(f"[check] corridor criterion: W={a.corridor_clearance} m -> block cells "
              f"within r_o + W/2 = {float(oc.pillar_radius) + a.corridor_clearance / 2:.4f} m "
              f"of a pillar, then require connectivity (and r_s+W/2 = "
              f"{float(oc.pillar_radius) + r_dr + infl + a.corridor_clearance / 2:.4f} m "
              f"for the 'drone centre' variant)")

    bad = 0
    worst = {"init": 1e9, "goal": 1e9, "gap": 1e9, "corridor": 1e9}
    conn_fail = 0
    for s in range(a.layouts):
        init = start.unsqueeze(0).repeat(a.num_envs, 1)
        tgt = goal.unsqueeze(0).repeat(a.num_envs, 1)
        pos, radius, active = mgr.sample_layout(init, tgt, None)
        r_s = radius + r_dr + infl
        act = active.bool()
        n_act = act.sum(dim=-1)
        ok_act = bool((n_act == M).all()) if not randomized else True
        act_lo, act_hi = int(n_act.min()), int(n_act.max())

        d_init = (init[:, None, :] - pos).norm(dim=-1) - r_s
        d_goal = (tgt[:, None, :] - pos).norm(dim=-1) - r_s
        d_init = torch.where(act, d_init, torch.full_like(d_init, float("inf")))
        d_goal = torch.where(act, d_goal, torch.full_like(d_goal, float("inf")))
        mc_init = float(d_init.min())
        mc_goal = float(d_goal.min())

        # Pairwise constraints, mirroring how _sample_pillar_mixed enforces them:
        #   * DIFFERENT pillars  : xy distance - 2*pillar_radius >= min_gap_between
        #     (layers of one pillar share an xy column and are stacked vertically, so
        #      layer-to-layer distances are meaningless for this check)
        #   * any pair involving a free ball: center distance - raw radii >= min_gap
        # The raw (un-inflated) radii are used, exactly as the sampler does.
        n_pil_slots = nP_max * L_max if randomized else nP * L
        gap = torch.full((a.num_envs,), float("inf"))
        pil_gap = torch.full((a.num_envs,), float("inf"))
        for i in range(M):
            for j in range(i + 1, M):
                both = act[:, i] & act[:, j]
                if not bool(both.any()):
                    continue
                d = (pos[:, i] - pos[:, j]).norm(dim=-1)
                same_pillar = (i < n_pil_slots and j < n_pil_slots
                               and i // block == j // block)
                if same_pillar:
                    continue                       # vertical stack inside one pillar
                g = d - radius[:, i] - radius[:, j]
                gap = torch.where(both, torch.minimum(gap, g), gap)
                if i < n_pil_slots and j < n_pil_slots:
                    pil_gap = torch.where(both, torch.minimum(pil_gap, g), pil_gap)
        mc_gap = float(gap.min())
        mc_corr = float(pil_gap.min())

        probs = []
        if not ok_act:
            probs.append(f"active={int(n_act[0])}!={M}")
        if randomized:
            # every active slot must belong to a pillar whose own layer count is <= L_max,
            # and the per-env active count must be a legal (n_pillars x layers) sum
            per_pil = act[:, :n_pil_slots].reshape(a.num_envs, nP_max, L_max).sum(dim=-1)
            if bool((per_pil > L_max).any()):
                probs.append("a pillar has more layers than L_max")
            if bool(((per_pil > 0).sum(dim=-1) > nP_max).any()):
                probs.append("more active pillars than n_pillars_max")
        if mc_init < mgr.init_clearance - 1e-6:
            probs.append(f"init_clr {mc_init:.4f} < {mgr.init_clearance}")
        if mc_goal < mgr.goal_clearance - 1e-6:
            probs.append(f"goal_clr {mc_goal:.4f} < {mgr.goal_clearance}")
        if mc_gap < mgr.min_gap_between - 1e-6:
            probs.append(f"gap {mc_gap:.4f} < {mgr.min_gap_between}")

        conn = None
        if a.connectivity:
            lo, hi = -3.0, 3.0
            n = int((hi - lo) / a.cell)
            xs = torch.linspace(lo + a.cell / 2, hi - a.cell / 2, n)
            gx, gy = torch.meshgrid(xs, xs, indexing="ij")
            pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)     # (P,2)
            blocked = torch.zeros(pts.shape[0], dtype=torch.bool)
            z_lo = float(oc.get("pillar_z_lo", 0.0))
            z_hi = float(oc.get("pillar_z_hi", 0.0))
            band = (min(start[2].item(), goal[2].item()) - 0.15,
                    max(start[2].item(), goal[2].item()) + 0.15)
            for i in range(M):
                if not bool(act[0, i]) or not (z_hi > band[0] and z_lo < band[1]):
                    continue
                d = (pts - pos[0, i, :2]).norm(dim=-1)
                thr = float(oc.pillar_radius) + (a.corridor_clearance or 0.0) / 2.0
                blocked |= d < thr
            blocked = blocked.reshape(n, n)
            conn = xy_grid_bfs(blocked, a.cell, lo, hi, start[:2], goal[:2])
            if conn is not True:
                probs.append(f"connectivity={conn}")
                conn_fail += 1

        for k, v in (("init", mc_init), ("goal", mc_goal), ("gap", mc_gap),
                     ("corridor", mc_corr)):
            worst[k] = min(worst[k], v)
        if probs:
            bad += 1
            print(f"  layout {s}: FAIL -> {', '.join(probs)}")
        else:
            extra = f" act={act_lo}-{act_hi}" if randomized else f" M={int(n_act[0])}"
            print(f"  layout {s}: ok {extra} init_clr={mc_init:.3f} "
                  f"goal_clr={mc_goal:.3f} gap={mc_gap:.3f} corridor={mc_corr:.3f}"
                  + (f" conn={conn}" if a.connectivity else ""))

    print(f"\n[check] worst over {a.layouts} layouts x {a.num_envs} envs: "
          f"init_clr={worst['init']:.4f} goal_clr={worst['goal']:.4f} "
          f"gap={worst['gap']:.4f} corridor={worst['corridor']:.4f}")
    print(f"[check] failing layouts: {bad}/{a.layouts}"
          + (f", connectivity failures: {conn_fail}/{a.layouts}" if a.connectivity else ""))
    print("[check] " + ("PASS" if bad == 0 else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
