# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
[M2-3] CPU unit tests for the first-order CBF layer (omni_drones/utils/cbf.py).

Run from OmniDrones/scripts with the lz_env python (no isaac needed):
    python cbf_test.py

Checks (m2_plan.md M2-3 acceptance):
  1. safety radius = geometric r_s + margin (+ brake v_max^2/(2a_max) when enabled).
  2. filter_velocity projects a HEAD-ON dangerous velocity away (constraint satisfied);
     a SAFE (tangential/away) velocity is left (near-)unchanged.
  3. multi-obstacle: flying into the gap between two CBF balls is corrected after the
     iterative projection (the most-violated constraint drives the fix each pass).
  4. cbf_violation (reward core): dangerous pre-filter v_nom -> negative (penalty),
     safe v_nom -> ~0, and an intruding position adds the h<0 penalty.
  5. config plumbing: cbf.mode=none -> build_cbf_filter returns None (naive, bit-identical
     regression guaranteed by not adding the transform at all).
"""

import math
import sys
import torch

sys.path.insert(0, "../")
from omni_drones.utils.cbf import (  # noqa: E402
    cbf_safety_radius,
    safety_radius_extra,
    safety_obs_channels,
    h_boundary_penalty,
    filter_velocity,
    cbf_violation,
    extract_cbf_params,
    build_cbf_filter,
)

torch.manual_seed(0)

# NavVel defaults (must mirror NavVel.yaml when writing new tests)
DRONE_R, INFL = 0.15, 0.05
MARGIN, A_MAX, V_MAX = 0.1, 2.0, 1.8
BRAKE = V_MAX ** 2 / (2.0 * A_MAX)          # 0.405
ALPHA = 1.0

_fail = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    if not cond:
        _fail.append(name)
    print(f"[{mark}] {name}  {detail}")


def t1_radius():
    # single obstacle logical r_o = 0.30 -> geometric r_s = 0.15+0.30+0.05 = 0.50
    r_o = 0.30
    extra = safety_radius_extra(DRONE_R, INFL, MARGIN, V_MAX, A_MAX, True)
    check("t1 extra= margin+brake", abs(extra - (MARGIN + BRAKE)) < 1e-9,
          f"extra={extra:.4f} exp={MARGIN+BRAKE:.4f}")
    rs = cbf_safety_radius(r_o, DRONE_R, INFL, MARGIN, V_MAX, A_MAX, True)
    check("t1 r_s^cbf = 0.5+margin+brake", abs(rs - (0.50 + MARGIN + BRAKE)) < 1e-9,
          f"rs={rs:.4f}")
    rs_no = cbf_safety_radius(r_o, DRONE_R, INFL, MARGIN, V_MAX, A_MAX, False)
    check("t1 no-brake -> +margin only", abs(rs_no - 0.60) < 1e-9, f"rs={rs_no:.4f}")


def _single(center, rad, pos, v, extra=MARGIN + BRAKE):
    """Return projected v given one obstacle at center (env frame tensors)."""
    pos = torch.as_tensor(pos, dtype=torch.float32).reshape(1, 3)
    v = torch.as_tensor(v, dtype=torch.float32).reshape(1, 3)
    p = torch.as_tensor(center, dtype=torch.float32).reshape(1, 1, 3)
    rs = torch.as_tensor([[rad + DRONE_R + INFL + extra]], dtype=torch.float32)
    act = torch.ones(1, 1, dtype=torch.bool)
    vs, fn = filter_velocity(pos, v, p, rs, act, ALPHA, iterations=5)
    return vs[0], fn[0]


def t2_head_on_projected():
    # obstacle at (2,0,1) radius 0.3 -> r_s^cbf = 0.5 + 0.505 = 1.005; drone at (0,0,1)
    # (2 m from center) => h = 2 - 1.005 = 0.995 > 0. Head-on at 1.8 m/s must be braked
    # to the critical approach speed alpha*h (n.v = -alpha*h), i.e. vx ~ alpha*h.
    center = [2.0, 0.0, 1.0]
    pos = [0.0, 0.0, 1.0]
    v = [1.8, 0.0, 0.0]                        # head-on
    rs_cbf = 0.30 + DRONE_R + INFL + MARGIN + BRAKE
    h = 2.0 - rs_cbf                            # 0.995
    vs, fn = _single(center, 0.30, pos, v)
    nv = (vs * torch.tensor([1.0, 0.0, 0.0])).sum()   # moving into obstacle = -x
    check("t2 head-on: constraint n.v+ah>=0", (nv + ALPHA * h) > -1e-4,
          f"n.v={nv.item():.4f} alpha*h={ALPHA*h:.4f}")
    check("t2 head-on: braked to critical approach speed", abs(vs[0].item() - ALPHA * h) < 1e-2,
          f"vx={vs[0].item():.4f} exp~{ALPHA*h:.4f}")
    check("t2 head-on: actually corrected (fn>0)", fn.item() > 0.5, f"fn={fn.item():.4f}")
    check("t2 head-on: no spurious y", abs(vs[1].item()) < 1e-4, f"vy={vs[1].item():.4f}")


def t3_safe_velocity_untouched():
    center = [2.0, 0.0, 1.0]
    pos = [0.0, 0.0, 1.0]
    # moving away from obstacle: no constraint violation -> unchanged
    v = [-1.0, 0.0, 0.0]
    vs, fn = _single(center, 0.30, pos, v)
    check("t3 away velocity unchanged", torch.allclose(vs, torch.tensor(v, dtype=torch.float32), atol=1e-5)
          and fn.item() == 0.0, f"v={vs.tolist()} fn={fn.item():.4f}")
    # tangential orbit (perpendicular, keeps distance): also untouched (h const -> dh=0>=0)
    v_t = [0.0, 1.8, 0.0]
    vs, fn = _single(center, 0.30, pos, v_t)
    check("t3 tangential velocity unchanged", torch.allclose(vs, torch.tensor(v_t, dtype=torch.float32), atol=1e-5),
          f"v={vs.tolist()}")


def t4_multiobstacle_gap():
    # two CBF balls the drone sits between, BOTH violated by a head-on command; the
    # iterative projector must satisfy every active constraint without inflating speed
    # (and must still keep a forward component - it picks one side, not stop forever).
    o1 = torch.tensor([[1.0, 0.5, 1.0]], dtype=torch.float32)
    o2 = torch.tensor([[1.0, -0.5, 1.0]], dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    v_nom = torch.tensor([[1.8, 0.0, 0.0]], dtype=torch.float32)
    p = torch.cat([o1.unsqueeze(0), o2.unsqueeze(0)], dim=1)      # (1,2,3)
    rs = torch.tensor([[0.7, 0.7]], dtype=torch.float32)          # geometric + margin radius
    act = torch.ones(1, 2, dtype=torch.bool)
    vs, fn = filter_velocity(pos, v_nom, p, rs, act, ALPHA, iterations=8)
    # recompute constraints after projection
    d = pos.unsqueeze(1) - p
    dist = torch.norm(d, dim=-1)
    n = d / torch.norm(d, dim=-1, keepdim=True)
    h = dist - rs
    g = (n * vs.unsqueeze(1)).sum(-1) + ALPHA * h                 # (1,2)
    check("t4 all active constraints satisfied", bool((g >= -1e-3).all()), f"g={g.tolist()}")
    # the two CBF balls (r=0.7, centers 1.0 apart) OVERLAP -> the corridor ahead is
    # physically blocked; the projector must not push the drone through it.
    check("t4 blocked corridor -> not pushed through", vs[0, 0].item() <= 1e-3,
          f"vx={vs[0, 0].item():.4f}")
    check("t4 projection did not inflate speed", torch.norm(vs).item() <= torch.norm(v_nom).item() + 1e-3,
          f"|v|={torch.norm(vs).item():.4f} |v_nom|={torch.norm(v_nom).item():.4f}")
    check("t4 fix was active", fn.item() > 1.0, f"fn={fn.item():.4f}")


def t5_violation_reward_core():
    center = [2.0, 0.0, 1.0]
    pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    p = torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0)
    rs_cbf = 0.30 + DRONE_R + INFL + MARGIN + BRAKE
    rs = torch.tensor([[rs_cbf]], dtype=torch.float32)
    act = torch.ones(1, 1, dtype=torch.bool)
    # dangerous v_nom straight at obstacle -> violation < 0
    v_bad = torch.tensor([[1.8, 0.0, 0.0]], dtype=torch.float32)
    viol_bad = cbf_violation(pos, v_bad, p, rs, act, ALPHA, penalty_intrude=True)
    check("t5 dangerous v_nom -> negative penalty", viol_bad.item() < -0.1,
          f"viol={viol_bad.item():.4f}")
    # safe v_nom away -> ~0
    v_ok = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)
    viol_ok = cbf_violation(pos, v_ok, p, rs, act, ALPHA, penalty_intrude=True)
    check("t5 safe v_nom -> ~0", abs(viol_ok.item()) < 1e-4, f"viol={viol_ok.item():.4f}")
    # intruding position (inside CBF ball): even a non-head-on cmd gets h<0 penalty
    pos_in = torch.tensor([[2.2, 0.0, 1.0]], dtype=torch.float32)   # 0.2 into the ball
    viol_in = cbf_violation(pos_in, v_ok, p, rs, act, ALPHA, penalty_intrude=True)
    viol_in_off = cbf_violation(pos_in, v_ok, p, rs, act, ALPHA, penalty_intrude=False)
    check("t5 intrude adds h<0 penalty", viol_in.item() < viol_in_off.item() - 1e-4,
          f"with={viol_in.item():.4f} without={viol_in_off.item():.4f}")
    # reward-core APPLICATION sign (regression for the 2026-09-05 bug where `reward -= w*viol`
    # turned the penalty into a bonus and return ballooned to ~1e4):
    #   reward_core = base + w * viol  (viol<=0  =>  safe: unchanged, unsafe: smaller)
    w = 0.5
    base = torch.zeros_like(viol_bad)
    reward_bad = base + w * viol_bad
    check("t5 reward core PENALIZES unsafe cmd", reward_bad.item() < 0.0,
          f"reward_core={reward_bad.item():.4f}")
    check("t5 reward core leaves safe cmd unchanged",
          (base + w * viol_ok).item() == 0.0)


def t7_safety_obs_channels():
    """[New2] obs 结构级 internalize 安全通道 (new2_plan.md §2) 的归一化数值回归。

    channel = clamp(d / norm, -1, 1), d = dmin (min surface clearance) or
    h = dmin - cbf_extra (CBF 边界余量, 0 穿越 = filter 介入边界)。
    """
    from omni_drones.utils.cbf import safety_obs_channels
    # no active obstacle: dmin=+inf -> both channels saturate at +1 (fully safe)
    d = torch.tensor([[float("inf")]], dtype=torch.float32)
    ch = safety_obs_channels(d, 0.1, 0.6, True, True)
    check("t7 no-obs -> clearance=+1", ch[0].item() == 1.0, f"v={ch[0].item():.3f}")
    check("t7 no-obs -> cbf_margin=+1", ch[1].item() == 1.0, f"v={ch[1].item():.3f}")
    check("t7 channel shapes (N,1,1)", all(c.shape == (1, 1, 1) for c in ch),
          f"shapes={[tuple(c.shape) for c in ch]}")

    d = torch.tensor([[0.3]], dtype=torch.float32)      # 0.3 m surface clearance
    # clearance only: single channel, value = 0.3/0.6
    ch = safety_obs_channels(d, 0.1, 0.6, True, False)
    check("t7 clearance-only len=1", len(ch) == 1)
    check("t7 clearance = dmin/norm", abs(ch[0].item() - 0.3 / 0.6) < 1e-6,
          f"v={ch[0].item():.4f} exp={0.3/0.6:.4f}")

    # cbf_margin only: h = dmin - extra (extra = margin + brake = 0.1+0.81=0.91 in
    # New1 hybrid/dual; pick dmin=1.2 so h/norm stays un-clamped)
    d12 = torch.tensor([[1.2]], dtype=torch.float32)
    extra = MARGIN + V_MAX ** 2 / (2.0 * A_MAX)         # 0.1 + 0.81 = 0.91
    ch = safety_obs_channels(d12, extra, 0.6, False, True)
    exp = (1.2 - extra) / 0.6
    check("t7 cbf_margin = (dmin-extra)/norm", abs(ch[0].item() - exp) < 1e-6,
          f"v={ch[0].item():.4f} exp={exp:.4f}")

    # both -> 2 channels in [clearance, cbf_margin] order (dmin=0.5 keeps both channels
    # inside (-1,1): clearance=0.833, cbf_margin=(0.5-0.91)/0.6=-0.683)
    ch = safety_obs_channels(torch.tensor([[0.5]]), extra, 0.6, True, True)
    check("t7 both -> 2 channels", len(ch) == 2)
    check("t7 both order [clearance, cbf_margin]",
          abs(ch[0].item() - 0.5 / 0.6) < 1e-6 and abs(ch[1].item() - (0.5 - extra) / 0.6) < 1e-6,
          f"v=[{ch[0].item():.4f},{ch[1].item():.4f}]")

    # clamp keeps "negative margin = danger" info down to -1 (no relu):
    # far away saturates +1; deep inside CBF/obstacle ball saturates -1
    ch = safety_obs_channels(torch.tensor([[5.0]]), extra, 0.6, True, True)
    check("t7 far -> saturate +1", ch[0].item() == 1.0 and ch[1].item() == 1.0)
    ch = safety_obs_channels(torch.tensor([[-2.0]]), extra, 0.6, True, True)
    check("t7 deep-inside -> saturate -1", ch[0].item() == -1.0 and ch[1].item() == -1.0)
    # negative h retained & unclamped: dmin=0.5 < extra=0.91 (crossed filter boundary,
    # still outside geometric ball) -> h=-0.41, channel=-0.683
    ch = safety_obs_channels(torch.tensor([[0.5]]), extra, 0.6, False, True)
    check("t7 crossed filter boundary -> negative h kept",
          -1.0 < ch[0].item() < 0.0, f"v={ch[0].item():.4f}")


def t8_h_boundary_penalty():
    """[New2/E1-v2] CBF 边界余量罚 (nav_vel reward core) 数值回归。

    pen = w * relu(buffer - h), h = dmin - extra (extra = brake-off cbf_extra = 0.1)。
    buffer=0 → E1 原版 relu(-h); buffer>0 提前 fire(软墙 dmin < extra+buffer)。
    """
    from omni_drones.utils.cbf import h_boundary_penalty
    extra = 0.1
    # buffer=0: E1 原版; far (dmin>extra) -> 0
    p0 = h_boundary_penalty(torch.tensor([[0.5]]), extra, 0.0, 40.0)
    check("t8 buffer0 far -> 0", p0.item() == 0.0, f"p={p0.item():.4f}")
    # buffer=0: inside filter zone (dmin<extra) fires w*(extra-dmin)
    p_in = h_boundary_penalty(torch.tensor([[0.05]]), extra, 0.0, 40.0)
    check("t8 buffer0 inside fires", abs(p_in.item() - 40.0 * (0.1 - 0.05)) < 1e-6,
          f"p={p_in.item():.4f} exp={40.0*0.05:.4f}")
    # buffer=0.1: fires BEFORE crossing (soft wall at dmin<0.2)
    pb = h_boundary_penalty(torch.tensor([[0.15]]), extra, 0.1, 40.0)  # h=0.05<0.1
    check("t8 buffer>0 fires early", abs(pb.item() - 40.0 * (0.1 - (0.15 - 0.1))) < 1e-6,
          f"p={pb.item():.4f} exp={40.0*0.05:.4f}")
    # buffer edge exactly h=buffer -> 0 (no fire)
    pz = h_boundary_penalty(torch.tensor([[0.2]]), extra, 0.1, 40.0)   # h=0.1=buffer
    check("t8 buffer edge -> 0", pz.item() == 0.0, f"p={pz.item():.4f}")
    # deeper inside scales up: dmin=0 (h=-0.1) with buffer 0.1 -> w*0.2
    pd = h_boundary_penalty(torch.tensor([[0.0]]), extra, 0.1, 40.0)
    check("t8 deeper -> bigger", abs(pd.item() - 40.0 * 0.2) < 1e-6, f"p={pd.item():.4f}")
    # no active obstacle (dmin=inf) -> 0 regardless of buffer
    pinf = h_boundary_penalty(torch.tensor([[float("inf")]]), extra, 0.1, 40.0)
    check("t8 no-obs inf -> 0", pinf.item() == 0.0)


def t6_config_plumbing():
    from omegaconf import OmegaConf
    base = OmegaConf.create({
        "task": {
            "cbf": {"mode": "none"},
            "obstacle": {"drone_radius": 0.15, "inflation": 0.05, "max_slots": 8},
            "vel_limit": {"max_vel": 1.8},
        }
    })
    p = extract_cbf_params(base)
    check("t6 params extracted", p is not None and p["mode"] == "none")
    check("t6 mode=none -> no filter transform", build_cbf_filter(base) is None)
    # reward_only -> transform present with do_filter=False (records v_nom only)
    base.task.cbf.mode = "reward_only"
    t = build_cbf_filter(base)
    check("t6 reward_only -> transform(no filter)", t is not None and not t.do_filter)
    # hybrid -> transform with filter
    base.task.cbf.mode = "hybrid"
    t = build_cbf_filter(base)
    check("t6 hybrid -> transform(filter)", t is not None and t.do_filter)
    check("t6 default params (alpha=1.0/margin=0.1)", p["alpha"] == 1.0
          and p["r_safety_margin"] == 0.1 and p["v_max"] == 1.8)


def t9_shadow_diag():
    """[K5 2026-09-12] CBFVelocityFilter 的 shadow / record_diag（纯诊断开关）。

    契约：
      * shadow=False, do_filter=True  -> 正常滤波（写回动作），diagnostics 记录。
      * shadow=True                   -> **不写回**动作（策略行为 = 撤掉 filter），
                                         但仍记录 ‖Δa‖/intervened/h_min（反事实诊断）。
      * record_diag=False             -> 不累积（与旧行为逐位兼容）。
      * do_filter=False, shadow=False -> 不做任何计算（reward_only 臂的旧行为）。
    """
    try:
        from tensordict import TensorDict
        from omni_drones.utils.cbf import CBFVelocityFilter
    except Exception as e:                                   # pragma: no cover
        check("t9 torchrl/tensordict available", False, str(e))
        return

    extra = MARGIN                     # brake off -> cbf_extra = 0.10（交付口径）
    r_cbf = 0.30 + DRONE_R + INFL + extra          # 0.60
    D = 0.75                                       # 中心距 -> h = 0.15, dmin = 0.25
    center = torch.tensor([[[1.0, 0.0, 1.0]]])     # (1,1,3)
    r_safe = torch.full((1, 1), r_cbf)
    # 无人机在障碍 +x 侧 D 处，指令沿 +x（正对障碍）-> n·v = -1.5, alpha*h = 0.15
    pos = torch.tensor([[[1.0 - D, 0.0, 1.0]]])    # (1,1,3)
    v_nom = torch.tensor([[[1.5, 0.0, 0.0]]])

    def run(shadow, record, v=None, p=None):
        f = CBFVelocityFilter(alpha=ALPHA, iterations=5, do_filter=True,
                              max_vel=V_MAX, shadow=shadow, record_diag=record)
        td = TensorDict({
            ("info", "drone_state"): torch.cat(
                [p if p is not None else pos, torch.zeros(1, 1, 10)], -1),
            ("info", "obstacle_cbf"): torch.cat(
                [center, r_safe.unsqueeze(-1)], -1).unsqueeze(1),      # (1,1,1,4)
            ("agents", "action"): (v if v is not None else v_nom).clone(),
        }, [1])
        td = f._inv_call(td)
        return f, td[("agents", "action")]

    # ---- 正常滤波 -------------------------------------------------------------
    f, a = run(shadow=False, record=True)
    dg = f.diag_tensor().reshape(-1, 4)
    check("t9 filter writes back projected v",
          abs(float(a[0, 0, 0]) - 0.15) < 1e-4, f"v_x={float(a[0,0,0]):.4f} exp=0.15")
    check("t9 diag corr = ||Δa||", abs(float(dg[0, 0]) - 1.35) < 1e-4,
          f"corr={float(dg[0,0]):.4f} exp=1.35")
    check("t9 diag intervened=1", float(dg[0, 1]) == 1.0)
    check("t9 diag h_min = dist-r_cbf", abs(float(dg[0, 2]) - 0.15) < 1e-4,
          f"h={float(dg[0,2]):.4f} exp=0.15")

    # ---- shadow：不写回，但诊断照样有 ------------------------------------------
    f2, a2 = run(shadow=True, record=True)
    dg2 = f2.diag_tensor().reshape(-1, 4)
    check("t9 shadow does NOT write back", float((a2 - v_nom).abs().max()) == 0.0,
          f"max|a-v_nom|={float((a2-v_nom).abs().max()):.3e}")
    check("t9 shadow still records same diag",
          abs(float(dg2[0, 0]) - 1.35) < 1e-4 and float(dg2[0, 1]) == 1.0,
          f"corr={float(dg2[0,0]):.4f}")

    # ---- 无介入：corr 精确 0 ---------------------------------------------------
    v_away = torch.tensor([[[-1.5, 0.0, 0.0]]])     # 远离障碍
    f3, a3 = run(shadow=False, record=True, v=v_away)
    dg3 = f3.diag_tensor().reshape(-1, 4)
    check("t9 no-violation -> action untouched & corr==0",
          float((a3 - v_away).abs().max()) == 0.0 and float(dg3[0, 0]) == 0.0
          and float(dg3[0, 1]) == 0.0,
          f"corr={float(dg3[0,0]):.1e} intervened={float(dg3[0,1]):.1f}")

    # ---- record_diag=False -> 不累积（旧行为） ---------------------------------
    f4, _ = run(shadow=False, record=False)
    check("t9 record_diag=False -> empty log", f4.diag_tensor() is None
          and f4.diag_steps == 0)

    # ---- do_filter=False & shadow=False -> 完全不计算（reward_only 旧行为） -----
    f5 = CBFVelocityFilter(alpha=ALPHA, iterations=5, do_filter=False,
                           max_vel=V_MAX, record_diag=True)
    td = TensorDict({
        ("info", "drone_state"): torch.cat([pos, torch.zeros(1, 1, 10)], -1),
        ("info", "obstacle_cbf"): torch.cat(
            [center, r_safe.unsqueeze(-1)], -1).unsqueeze(1),
        ("agents", "action"): v_nom.clone(),
    }, [1])
    td = f5._inv_call(td)
    check("t9 reward_only path unchanged (no compute, action untouched)",
          float((td[("agents", "action")] - v_nom).abs().max()) == 0.0
          and f5.diag_tensor() is None)

    # ---- config 接线 ----------------------------------------------------------
    from omegaconf import OmegaConf
    base = OmegaConf.create({"task": {
        "cbf": {"mode": "hybrid", "alpha": 1.0},
        "obstacle": {"drone_radius": DRONE_R, "inflation": INFL},
        "vel_limit": {"max_vel": V_MAX},
    }})
    tf = build_cbf_filter(base, shadow=True, record_diag=True)
    check("t9 build_cbf_filter passes shadow/record_diag",
          tf is not None and tf.shadow and tf.record_diag and tf.do_filter)
    check("t9 build defaults -> shadow=False (逐位兼容)",
          build_cbf_filter(base).shadow is False
          and build_cbf_filter(base).record_diag is False)


def t10_p_filter():
    """[P3 2026-09-16] p_filter：执行概率**只控执行、不控计算**（软蒸馏/退火机制）。

    契约：
      * p=1（默认）逐位等同旧行为，且**不消耗 RNG**（保 p1 臂的确定性）；
      * p=0 写回（已限幅）v_nom（裸奔执行），但 diag 里 corr>0（奖励仍有梯度）；
      * 0<p<1 由 rand 决定执行与否，消耗 RNG；
      * `p_filter_schedule(step)` 与 `p_filter` 用**同一个**自持计数器（sched_steps）；
      * emit_info=True 时写 info(p_filter/cbf_corr/cbf_intervened/cbf_executed)。
    """
    try:
        from tensordict import TensorDict
        from omni_drones.utils.cbf import CBFVelocityFilter
    except Exception as e:                                   # pragma: no cover
        check("t10 torchrl/tensordict available", False, str(e))
        return

    extra = MARGIN                                    # brake off -> cbf_extra = 0.10
    r_cbf = 0.30 + DRONE_R + INFL + extra             # 0.60
    D = 0.75                                          # h = 0.15, dmin = 0.25
    center = torch.tensor([[[1.0, 0.0, 1.0]]])
    r_safe = torch.full((1, 1), r_cbf)
    pos = torch.tensor([[[1.0 - D, 0.0, 1.0]]])
    v_nom = torch.tensor([[[1.5, 0.0, 0.0]]])         # 正对障碍 -> 违例 1.5-0.15=1.35

    def make_td():
        return TensorDict({
            ("info", "drone_state"): torch.cat([pos, torch.zeros(1, 1, 10)], -1),
            ("info", "obstacle_cbf"): torch.cat(
                [center, r_safe.unsqueeze(-1)], -1).unsqueeze(1),      # (1,1,1,4)
            ("agents", "action"): v_nom.clone(),
        }, [1])

    def run(p=1.0, sched=None, info=False):
        f = CBFVelocityFilter(alpha=ALPHA, iterations=5, do_filter=True,
                              max_vel=V_MAX, p_filter=p, record_diag=True,
                              p_filter_schedule=sched, emit_info=info)
        return f, f._inv_call(make_td())

    # ---- p=1：逐位旧行为，不动 RNG -------------------------------------------
    s0 = torch.get_rng_state()
    f1, td1 = run(p=1.0)
    s1 = torch.get_rng_state()
    ax = float(td1[("agents", "action")][0, 0, 0])
    check("t10 p=1 writes projected v", abs(ax - 0.15) < 1e-6, f"v_x={ax:.6f}")
    check("t10 p=1 consumes no RNG", torch.equal(s0, s1))
    check("t10 p=1 sched_steps counts once", f1.sched_steps == 1)

    # ---- p=0：裸奔执行（== v_nom），corr 仍>0 ---------------------------------
    s0 = torch.get_rng_state()
    f0, td0 = run(p=0.0)
    s1 = torch.get_rng_state()
    a0 = td0[("agents", "action")]
    check("t10 p=0 does NOT filter (action == v_nom)",
          float((a0 - v_nom).abs().max()) == 0.0,
          f"max|a-v|={float((a0-v_nom).abs().max()):.3e}")
    dg0 = f0.diag_tensor()
    check("t10 p=0 still computes corr>0 (gradient for soft distillation)",
          dg0 is not None and float(dg0.reshape(-1, 4)[0, 0]) > 1.0,
          f"corr={float(dg0.reshape(-1,4)[0,0]):.4f}")
    check("t10 p=0 consumes no RNG", torch.equal(s0, s1))

    # ---- emit_info：键齐全、数值正确 ------------------------------------------
    f0i, td0i = run(p=0.0, info=True)
    check("t10 emit_info(p=0) keys",
          float(td0i[("info", "p_filter")]) == 0.0
          and float(td0i[("info", "cbf_executed")]) == 0.0
          and float(td0i[("info", "cbf_intervened")]) == 1.0
          and float(td0i[("info", "cbf_corr")]) > 1.0)
    f1i, td1i = run(p=1.0, info=True)
    check("t10 emit_info(p=1) keys",
          float(td1i[("info", "cbf_executed")]) == 1.0
          and abs(float(td1i[("info", "p_filter")]) - 1.0) < 1e-7)

    # ---- 退火 callable：同一时钟；step0 p=1、step2 p=0；p=0.5 混合 --------
    calls = []

    def sched(step):
        calls.append(step)
        return max(0.0, 1.0 - step)

    f2 = CBFVelocityFilter(alpha=ALPHA, iterations=5, do_filter=True,
                           max_vel=V_MAX, p_filter_schedule=sched, emit_info=True)
    tds = [f2._inv_call(make_td()) for _ in range(3)]
    check("t10 schedule clock is 0,1,2", calls == [0, 1, 2])
    check("t10 schedule step0 filters (0.15)",
          abs(float(tds[0][("agents", "action")][0, 0, 0]) - 0.15) < 1e-6)
    check("t10 schedule step2 no filter & flag=0",
          float((tds[2][("agents", "action")] - v_nom).abs().max()) == 0.0
          and float(tds[2][("info", "cbf_executed")]) == 0.0)
    vx_mid = float(tds[1][("agents", "action")][0, 0, 0])
    check("t10 schedule step1 mixed branch (0.15 or 1.5)",
          abs(vx_mid - 0.15) < 1e-6 or abs(vx_mid - 1.5) < 1e-6,
          f"v_x={vx_mid:.4f}")

    # ---- build_cbf_filter：use_schedule / anneal_frac / always_off / p_override --
    from omegaconf import OmegaConf
    base = OmegaConf.create({
        "task": {
            "cbf": {"mode": "hybrid", "alpha": 1.0,
                    "p_filter_schedule": {"mode": "anneal", "start": 1.0,
                                          "end": 0.0, "anneal_frac": 0.5}},
            "obstacle": {"drone_radius": DRONE_R, "inflation": INFL},
            "vel_limit": {"max_vel": V_MAX},
            "env": {"num_envs": 1000},
        },
        "total_frames": 100000,
    })
    tf = build_cbf_filter(base, use_schedule=True, emit_info=True)
    # anneal_steps = 100000/1000 * 0.5 = 50 -> p(0)=1, p(25)=0.5, p>=50 -> 0
    check("t10 build anneal p(0/25/60) = 1/0.5/0",
          tf is not None and tf.p_filter_schedule is not None
          and abs(float(tf.p_filter_schedule(0)) - 1.0) < 1e-9
          and abs(float(tf.p_filter_schedule(25)) - 0.5) < 1e-9
          and abs(float(tf.p_filter_schedule(60)) - 0.0) < 1e-9)
    t_def = build_cbf_filter(base)
    check("t10 build default (use_schedule=False) -> constant p=1",
          t_def is not None and t_def.p_filter_schedule is None
          and float(t_def.p_filter) == 1.0 and t_def.emit_info is False)
    base_off = OmegaConf.create({
        "task": {
            "cbf": {"mode": "hybrid",
                    "p_filter_schedule": {"mode": "always_off"}},
            "obstacle": {"drone_radius": DRONE_R, "inflation": INFL},
            "vel_limit": {"max_vel": V_MAX},
            "env": {"num_envs": 1000},
        },
        "total_frames": 100000,
    })
    t_off = build_cbf_filter(base_off, use_schedule=True)
    check("t10 build always_off -> p=0",
          t_off is not None and float(t_off.p_filter) == 0.0
          and t_off.p_filter_schedule is None)
    t_ov = build_cbf_filter(base, use_schedule=True, p_override=1.0)
    check("t10 p_override beats schedule",
          t_ov is not None and t_ov.p_filter_schedule is None
          and float(t_ov.p_filter) == 1.0)


if __name__ == "__main__":
    t1_radius()
    t2_head_on_projected()
    t3_safe_velocity_untouched()
    t4_multiobstacle_gap()
    t5_violation_reward_core()
    t6_config_plumbing()
    t7_safety_obs_channels()
    t8_h_boundary_penalty()
    t9_shadow_diag()
    t10_p_filter()
    print()
    if _fail:
        print(f"RESULT: {len(_fail)} FAILED -> {_fail}")
        sys.exit(1)
    print("RESULT: all tests PASSED")
