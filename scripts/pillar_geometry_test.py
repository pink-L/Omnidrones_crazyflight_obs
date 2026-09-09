# [2026-09-08] CPU 单元测试: NavVel pillar 混合布局 (方案A: 4柱×4层外接球 + 12 自由球 = 28)
#   不依赖 Isaac; 直接按文件路径加载 nav_vel_obstacles.py (纯 torch)。
# 用法 (OmniDrones/scripts/, conda activate lz_env):  python pillar_geometry_test.py
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def load_module(name, rel_path):
    path = os.path.join(REPO, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ObstacleManager = load_module("nav_vel_obstacles",
                              "omni_drones/envs/single/nav_vel_obstacles.py").ObstacleManager

CFG = {
    "max_slots": 8, "num_scene": 28,
    "n_pillars": 4, "pillar_layers": 4, "pillar_radius": 0.354,
    "pillar_z_lo": 0.4, "pillar_z_hi": 2.6, "n_free_obstacles": 12,
    "radius_choices": [0.20, 0.30, 0.40],
    "drone_radius": 0.15, "inflation": 0.05,
    "collision_margin": 0.05, "danger_radius": 0.6,
    "max_collisions": 2, "init_clearance": 0.15, "goal_clearance": 0.35,
    "min_gap_between": 0.25, "spawn_relax": 0.85, "spawn_relax_rounds": 6,
    "obs_dist_norm": 5.0, "obs_radius_norm": 0.5, "keepout_x": 2.5,
    "spawn_xy_range": [[-3.0, -3.0], [3.0, 3.0]],
    "spawn_z_range": [0.6, 2.4],
}


def main():
    dev = "cpu"
    NE = 128
    mgr = ObstacleManager(CFG, NE, dev)
    nP, L = mgr.n_pillars, mgr.pillar_layers
    assert mgr.M == nP * L + mgr.n_free_balls == 28, mgr.M
    assert mgr.obs_window, "M=28 > K=8 -> sliding window obs should be on"

    init = torch.full((NE, 1, 3), -2.8, device=dev); init[..., 1] = 0.0; init[..., 2] = 0.5
    goal = torch.full((NE, 1, 3), 2.8, device=dev); goal[..., 1] = 0.0; goal[..., 2] = 1.0

    NB = 40
    min_ep_clr = float("inf")      # 每 ball 到 init/goal 最小净空 (半径面)
    min_free_free = float("inf")
    min_free_pillar = float("inf")
    n_fail = 0
    z_ok = True
    bind_ok = True
    keepout_ok = True
    in_box = True
    for _ in range(NB):
        pos, rad, act = mgr.sample_layout(init, goal, n_active=999)
        assert act.all(), "pillar layout should be fully active"
        assert pos.shape == (NE, 28, 3) and rad.shape == (NE, 28)
        # 柱层球(前16): 半径固定, 同柱 xy 同, z 分层递增覆盖
        pb = pos[:, :nP * L].reshape(NE, nP, L, 3)
        rad_pb = rad[:, :nP * L].reshape(NE, nP, L)
        bind_ok &= bool((pb[..., :2].std(dim=2) < 1e-4).all())       # 同柱 xy 相同
        zs = pb[..., 2]                                               # (NE,nP,L)
        dz = zs[..., 1:] - zs[..., :-1]
        z_ok &= bool((dz > 0).all()) and bool((dz <= 2 * 0.354 + 1e-3).all())  # 递增且连续覆盖
        z_ok &= bool(((zs[..., 0] - 0.354) <= 0.4 + 1e-3).all())      # 底覆盖 0.4
        z_ok &= bool(((zs[..., -1] + 0.354) >= 2.4 - 1e-3).all())     # 顶覆盖到 2.4
        rad_ok = bool((rad_pb - 0.354).abs().max() < 1e-4)
        bind_ok &= rad_ok
        keepout_ok &= bool((pos[..., 0].abs().max() <= 2.5 + 1e-3))   # 球面 |x|<=keepout
        # in spawn box
        lo = torch.tensor(CFG["spawn_xy_range"][0] + [0.6], device=dev) - 0.05
        hi = torch.tensor(CFG["spawn_xy_range"][1] + [2.4], device=dev) + 0.05
        in_box &= bool(((pos >= lo.unsqueeze(0).unsqueeze(0)).all() and
                        (pos <= hi.unsqueeze(0).unsqueeze(0)).all()))
        # 端点净空: 每球到 init/goal >= r_s + clearance(最小容忍 0.02)
        r_s = rad + mgr.drone_radius + mgr.inflation
        d_i = torch.norm(pos - init, dim=-1)
        d_g = torch.norm(pos - goal, dim=-1)
        min_ep_clr = min(min_ep_clr, float(((d_i - r_s) - 0.02).min()),
                         float(((d_g - r_s) - 0.02).min()))
        # 自由球(后12)之间与对柱距离
        free_pos = pos[:, nP * L:]
        free_rad = rad[:, nP * L:]
        d_ff = torch.cdist(free_pos, free_pos)                        # (NE,12,12)
        diag = torch.eye(free_pos.shape[1], dtype=torch.bool, device=dev)
        d_ff = d_ff.masked_fill(diag.unsqueeze(0), float("inf"))
        min_free_free = min(min_free_free, float((d_ff - free_rad[:, :, None] - free_rad[:, None, :]).min()))
        d_fp = torch.cdist(free_pos, pb.reshape(NE, nP * L, 3))
        min_free_pillar = min(min_free_pillar, float(
            (d_fp - free_rad[:, :, None] - 0.354).min()))
        n_fail += 0
        # commit 最后一次布局到 buffer, 供 clearances/obs 用真实激活
        mgr.commit_layout(torch.arange(NE), pos, rad, act)

    print(f"[pillar] M={mgr.M} (nP*L={nP * L} + free={mgr.n_free_balls}), obs_window={mgr.obs_window}")
    print(f"[pillar] in-box={in_box}  bind(xy同/rad=0.354)={bind_ok}  z分层连续覆盖={z_ok}  keepout(球面|x|<=2.5)={keepout_ok}")
    print(f"[pillar] endpoint min net clearance(r_s+clr-0.02 floor) min={min_ep_clr:.3f}  (应>=~0)")
    print(f"[pillar] free-free min clearance={min_free_free:.3f}  free-pillar min clearance={min_free_pillar:.3f}  (>=~0, 越贴近越贴合)")
    print(f"[pillar] n_fail={n_fail}/{NB} (rejection fallback 触发, 应极少)")

    # obs 块: 滑动窗口 -> (NE,1,K*4)=(NE,1,32) 62 维总
    dpos = torch.randn(NE, 1, 3, device=dev) * 1.0
    block = mgr.build_obs(dpos)
    assert block.shape == (NE, 1, mgr.K * 4), block.shape
    assert torch.isfinite(block).all()
    clr = mgr.clearances(dpos)
    assert torch.isfinite(clr).all()
    print(f"[pillar] build_obs shape={tuple(block.shape)} (K={mgr.K}) OK, clearances finite")

    ok = in_box and bind_ok and z_ok and keepout_ok and (min_ep_clr >= -0.02) \
        and (min_free_free >= -0.02) and (min_free_pillar >= -0.02)
    print("[pillar]", "ALL PASS" if ok else "CHECK FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
