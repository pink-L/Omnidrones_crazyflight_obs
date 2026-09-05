#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2-16 reward-core vs filter_only 训练轨迹出图（2026-09-05 22:5x 会话）。

数据来源：确定性 eval_ckpt@600 / 1024 env，D2-16（16球/±2.2, obs最近-8, brake off, w8 shaping,
  warm 自 D1 naive lmhvaqka-final）。mid-ckpt 逐帧 eval（6.5M→20M）。
结论要点（m2_plan #9 §6/§7）：
  * reward-core（corr/gauss）是"早峰型"：~6.5M 即确定性峰值后过训退化；
  * filter_only 是"爬升型"：无 reward-core 罚 → shaping 持续驱动 → 20M final 即峰。
  * peak 口径 corr-hybrid λ0.05 @6.5M(0.786) ≈ filter_only @20M(0.789)。

运行： python scripts/plot_d2_peak_trajectory.py → figures/d2_peak_trajectory.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/lz/lzspace/drones"
OUT = os.path.join(REPO, "figures", "d2_peak_trajectory.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# frames (M) at which deterministic evals were taken (same grid for all three)
F = [6.5, 9.9, 13.1, 16.4, 19.7, 20.0]

# name -> (joint, arrival, coll_env) per frame; run id in parens
TR = {
    "corr-hybrid $\\lambda$0.05 w8 (mu9vegr8)": dict(
        joint=[0.786, 0.721, 0.746, 0.710, 0.720, 0.708],
        arrival=[0.798, 0.732, 0.755, 0.725, 0.730, 0.723],
        coll=[0.017, 0.023, 0.015, 0.023, 0.020, 0.021],
        color="#d62728", peak=0.786, peak_at=6.5,
    ),
    "filter_only w8 (vaa0vtzz)": dict(
        joint=[0.734, 0.729, 0.748, 0.786, 0.774, 0.789],
        arrival=[0.742, 0.738, 0.763, 0.797, 0.785, 0.806],
        coll=[0.011, 0.022, 0.021, 0.014, 0.016, 0.022],
        color="#1f77b4", peak=0.789, peak_at=20.0,
    ),
    "gauss $\\sigma$0.5 $\\lambda$0.1 w8 (z36ag4ib)": dict(
        joint=[0.765, 0.746, 0.681, 0.663, 0.714, 0.700],
        arrival=[0.774, 0.750, 0.692, 0.673, 0.729, 0.716],
        coll=[0.016, 0.013, 0.017, 0.021, 0.021, 0.025],
        color="#2ca02c", peak=0.765, peak_at=6.5,
    ),
}

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.axvspan(3.0, 8.0, color="gray", alpha=0.10, label="early-peak window (~6.5M)")

for name, d in TR.items():
    ax.plot(F, d["joint"], marker="o", lw=2.2, color=d["color"], label=f"{name}  (joint)")
    ax.plot(F, d["arrival"], marker="x", lw=1.2, ls="--", color=d["color"], alpha=0.55,
            label=f"arrival")
    ip = F.index(d["peak_at"])
    ax.annotate(f"peak {d['peak']:.3f}\n@{d['peak_at']:.1f}M",
                xy=(d["peak_at"], d["peak"]), xytext=(d["peak_at"] + 0.4, d["peak"] + 0.018),
                fontsize=9, color=d["color"],
                arrowprops=dict(arrowstyle="->", color=d["color"], lw=1.0))

ax.set_xlabel("training frames (M)")
ax.set_ylabel("deterministic eval rate (@600 / 1024 env)")
ax.set_title("D2-16 (16 obs, ±2.2m, w8 shaping): reward-core (early-peak) vs filter_only (climbing) trajectories")
ax.set_xlim(5.5, 21.0)
ax.set_ylim(0.62, 0.85)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right", ncol=1)

# footnote: per-arm best coll in peak
note = ("peak coll:  corr 1.7% (6.5M) | filter_only 2.2% (20M) | gauss 1.6% (6.5M)\n"
        "value_loss ~0 throughout late training (critic converged early); policy still drifts -> reward-core overtrains after ~6.5M")
ax.text(0.01, -0.18, note, transform=ax.transAxes, fontsize=8, color="dimgray")

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
