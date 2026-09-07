# [Arena1 2026-09-07] 布局健康 / 可成功性一次性校验（CPU, 无需 isaac）。
# 检查改版 NavVel（6×6 全区域布障 + 固定 start(-2.5,0,0.5)->goal(2.5,0,1.5) + 总高 3m）:
#   1) 起点净空违反数   2) 终点净空违反数   3) 球-球 gap 违反数
#   4) start->goal 可行通路失败率 (保守 6-连通 voxel BFS, 无人机中心自由空间 = 球 r_s 外扩)
#   5) 球分布范围 (min/max xyz) 是否覆盖全 6×6 且在 [z_min,3] 内
# 用法: python arena1_layout_check.py [N=256] [res=0.25]
import importlib.util, os, sys, argparse
from collections import deque

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
def load_module(name, rel):
    p = os.path.join(REPO, rel)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
obs_mod = load_module("nav_vel_obstacles", "omni_drones/envs/single/nav_vel_obstacles.py")
ObstacleManager = obs_mod.ObstacleManager

# ---- 从 NavVel.yaml 读改版后配置 (NavVel.yaml 顶层即 task 键, 与训练/env 完全一致) ----
cfg = OmegaConf.load(os.path.join(REPO, "cfg/task/NavVel.yaml"))
task = cfg
ob = OmegaConf.to_container(task.obstacle, resolve=True)
fixed_init = torch.as_tensor(task.fixed_init, dtype=torch.float32)
fixed_goal = torch.as_tensor(task.fixed_target, dtype=torch.float32)
levels = list(task.curriculum.levels)
Z_MAX = float(task.z_max); Z_MIN = float(task.z_min)
DRONE_R = float(ob["drone_radius"]); INFL = float(ob["inflation"])
COL_M = float(ob["collision_margin"])
CLR_I = float(ob["init_clearance"]); CLR_G = float(ob["goal_clearance"])
GAP = float(ob["min_gap_between"])
spawn_lo = torch.as_tensor(ob["spawn_xy_range"][0] + [ob["spawn_z_range"][0]])
spawn_hi = torch.as_tensor(ob["spawn_xy_range"][1] + [ob["spawn_z_range"][1]])

def tol_pair(d_ij, ri, rj, need):  # 是否违反 (严格 < need - tol)
    return (d_ij < need - 1e-4)

def feasible_bfs(pos, rad, act, res, margin):
    """逐 env 6-连通 voxel BFS: 无人机中心路径是否存在 (start<->goal)."""
    rs = (rad + DRONE_R + INFL)                      # (L,) 判定半径
    lo = torch.tensor([-3.0, -3.0, 0.2]); hi = torch.tensor([3.0, 3.0, Z_MAX - 0.1])
    # voxel 网格 (不越界, z 从 0.2 起)
    dims = [int(np.floor((hi[i]-lo[i]) / res)) for i in range(3)]
    ax = [np.linspace(lo[i].item()+res/2, lo[i].item()+(dims[i]-0.5)*res, dims[i]) for i in range(3)]
    X, Y, Zz = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), Zz.ravel()], -1)         # (V,3)
    V = grid.shape[0]
    npass = 0
    for e in range(pos.shape[0]):
        L = int(act[e].sum().item())
        c = pos[e, :L].numpy(); r = rs[e, :L].numpy()
        if L == 0:
            npass += 1; continue
        d = np.linalg.norm(grid[:, None, :] - c[None, :, :], axis=-1)  # (V,L)
        blocked = (d < (r[None, :] + COL_M + margin)).any(axis=1)      # 中心需 ≥ r_s+col+margin
        bidx = grid[~blocked]                                        # 自由 voxel
        # map start/goal to nearest free voxel idx
        def nearest_idx(p):
            j = np.argmin(np.linalg.norm(grid - p[None, :], axis=1))
            return j
        si, gi = nearest_idx(fixed_init.numpy()), nearest_idx(fixed_goal.numpy())
        if blocked[si] or blocked[gi]:
            continue
        # BFS over axis neighbours on free voxels
        free = ~blocked
        order = {int(i): k for k, i in enumerate(np.where(free)[0])}
        # build neighbor lookup lazily
        dq = deque([si]); seen = {si}
        ok = False
        nx, ny, nz = dims
        while dq:
            cur = dq.popleft()
            if cur == gi:
                ok = True; break
            i, j, k = np.unravel_index(cur, dims)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        if abs(di)+abs(dj)+abs(dk) != 1:
                            continue
                        ni, nj, nk = i+di, j+dj, k+dk
                        if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                            continue
                        nb = ni*ny*nz + nj*nz + nk
                        if (not blocked[nb]) and nb not in seen:
                            seen.add(nb); dq.append(nb)
        npass += int(ok)
    return npass / pos.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", nargs="?", type=int, default=256)
    ap.add_argument("res", nargs="?", type=float, default=0.25)
    a = ap.parse_args()
    N, res = a.N, a.res
    torch.manual_seed(0); np.random.seed(0)
    mgr = ObstacleManager(dict(ob), N, "cpu")
    init = fixed_init.reshape(1, 1, 3).expand(N, 1, 3).clone()
    goal = fixed_goal.reshape(1, 1, 3).expand(N, 1, 3).clone()
    print(f"Arena1 layout health: N={N} layouts/level, voxel={res}, start={fixed_init.tolist()} goal={fixed_goal.tolist()}, z_max={Z_MAX}")
    print(f"spawn box xy=±3.0 z=[{ob['spawn_z_range'][0]},{ob['spawn_z_range'][1]}], levels={levels}\n")
    hdr = f"{'L':>3} | {'init_viol':>9} {'goal_viol':>9} {'gap_viol':>9} | {'feasible%':>9} | {'z_min':>6} {'z_max':>6} {'x_min':>6} {'x_max':>6}"
    print(hdr); print("-" * len(hdr))
    for L in levels:
        pos, rad, act = mgr.sample_layout(init, goal, L)
        active = act[:, :L]
        rs = rad[:, :L] + DRONE_R + INFL
        # endpoint net clearances (min over active per env)
        d_i = torch.norm(pos[:, :L] - init, dim=-1); d_g = torch.norm(pos[:, :L] - goal, dim=-1)
        net_i = (d_i - rs).min(dim=1).values          # 距起点表面净空
        net_g = (d_g - rs).min(dim=1).values
        init_viol = int((net_i < CLR_I - 1e-4).sum().item())   # 相对名义 init_clearance
        init_in   = int((net_i < COL_M).sum().item())          # 真的"生在球里"
        goal_viol = int((net_g < CLR_G - 1e-4).sum().item())
        # pairwise gap among active (only first L)
        p = pos[:, :L]; r = rad[:, :L]
        gp = 0
        for i in range(L - 1):
            d = torch.norm(p[:, i:i+1] - p[:, i+1:], dim=-1)     # (n, L-1-i)
            need = r[:, i:i+1] + r[:, i+1:] + GAP
            gp += int((d < need - 1e-4).sum().item())
        # distribution
        pa = p[active[:, :L]]
        xmin, xmax = pa[:, 0].min().item(), pa[:, 0].max().item()
        zmin, zmax = pa[:, 2].min().item(), pa[:, 2].max().item()
        feas = feasible_bfs(pos, rad, act, res, margin=0.02) * 100
        print(f"{L:>3} | {init_viol:>5}(in={init_in:>2}) {goal_viol:>9} {gp:>9} | {feas:>8.1f}% | "
              f"{zmin:>6.2f} {zmax:>6.2f} {xmin:>6.2f} {xmax:>6.2f}")
    print("\n注: init_viol 相对名义 init_clearance; gap_viol 相对 min_gap(逻辑半径); feasible 为保守 6-连通 BFS(中心需≥r_s+col+margin)。")

if __name__ == "__main__":
    main()
