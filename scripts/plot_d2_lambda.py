#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2-16 CBF reward_weight(lambda) 扫描 + hybrid 加长轨迹出图（2026-09-05 19:3x 会话）。

数据来源：确定性 eval_ckpt@600 / 1024 env，D2-16（16球/±2.2, obs最近-8, brake off），
  warm 自 D1 naive lmhvaqka-final。本次会话（18:35-19:4x）实测。
结论要点：
  * hybrid: lambda 越大越保守（arrival↓）；λ0.05@40M = 0.647/2.3%/0.634 全面超 filter_only 39M 峰值(0.639/3.3%/0.627)。
  * 加长至 60M 熵塌缩退化(arrival 0.372) → 40M 为甜点；filter_only 亦在 39M 峰值/80M 退化，机制一致。
  * reward_only 无滤波注定高碰撞(20M coll 35-45%)，80M 长训更死亡(0.002)。

运行： python scripts/plot_d2_lambda.py → figures/d2_lambda.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/lz/lzspace/drones"
OUT = os.path.join(REPO, "figures", "d2_lambda.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (label, arrival, joint, coll_env)
LAM20 = [
    ("hybrid\nλ0.05", 0.558, 0.546, 0.026),
    ("hybrid\nλ0.10", 0.502, 0.490, 0.022),
    ("hybrid\nλ0.20", 0.460, 0.452, 0.019),
    ("reward_only\nλ0.10", 0.662, 0.436, 0.453),
    ("reward_only\nλ0.20", 0.543, 0.390, 0.434),
    ("reward_only\nλ0.50", 0.371, 0.287, 0.352),
]
# hybrid 加长轨迹 (frames_M -> (arrival, coll_env))
HIST = {
    "λ0.05": {20: (0.558, 0.026), 40: (0.647, 0.023), 60: (0.372, 0.075)},
    "λ0.10": {20: (0.502, 0.022), 40: (0.539, 0.027)},
    "λ0.20": {20: (0.460, 0.019), 40: (0.494, 0.034)},
}
FILT = {39: (0.639, 0.033), 80: (0.401, 0.532)}          # filter_only（39M 峰值 / 80M 退化）
NAIVE = (0.001, 0.940)

C_ARR = "#1f77b4"; C_COL = "#d62728"
C_H = {"λ0.05": "#1f77b4", "λ0.10": "#2ca02c", "λ0.20": "#ff7f0e"}
C_F = "#9467bd"

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
fig.suptitle("D2-16 CBF reward_weight($\\lambda$) scan & hybrid long-finetune  (deterministic eval @600, 1024 env)",
             fontsize=12, y=0.99)

# ---- Fig A: 20M λ 扫描 eval ----
ax = axes[0]
labels = [l for l, *_ in LAM20]
x = np.arange(len(labels)); w = 0.26
ax.bar(x - w, [r[1] for r in LAM20], w, label="arrival", color=C_ARR, alpha=0.9)
ax.bar(x,     [r[2] for r in LAM20], w, label="joint success", color="#2ca02c", alpha=0.75)
ax.bar(x + w, [r[3] for r in LAM20], w, label="coll env frac", color=C_COL, alpha=0.55)
ax.axhline(FILT[39][0], color=C_F, ls="--", lw=1.2)
ax.text(5.35, FILT[39][0] + 0.02, "filter_only 39M peak", color=C_F, fontsize=8, ha="right")
for i, r in enumerate(LAM20):
    ax.text(i - w, r[1] + 0.02, f"{r[1]:.2f}", ha="center", fontsize=7)
    ax.text(i + w, r[3] + 0.02, f"{r[3]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
ax.set_title("A. 20M short-scan (warm D1 naive)")
ax.legend(fontsize=8)

# ---- Fig B: hybrid 加长轨迹 vs filter_only ----
ax = axes[1]
for lab, hist in HIST.items():
    fm = sorted(hist); ax.plot(fm, [hist[m][0] for m in fm], "-o", color=C_H[lab], label=f"hybrid {lab}")
fm_f = sorted(FILT); ax.plot(fm_f, [FILT[m][0] for m in fm_f], "-s", color=C_F, label="filter_only", lw=2)
ax.plot([78.7], [NAIVE[0]], "v", color="#555", label="naive 78.7M")
for lab, hist in HIST.items():
    for m, (a, c) in hist.items():
        ax.annotate(f"{a:.2f}", (m, a), textcoords="offset points", xytext=(0, 7),
                    fontsize=7.5, ha="center", color=C_H[lab])
for m, (a, c) in FILT.items():
    ax.annotate(f"{a:.2f}", (m, a), textcoords="offset points", xytext=(0, -15),
                fontsize=7.5, ha="center", color=C_F)
ax.axvspan(38, 41, color="#eee", alpha=0.9)
ax.annotate("sweet spot ~40M\n(entropy starts collapsing >40M)", xy=(41, 0.63), xytext=(52, 0.66),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("total training frames (M)"); ax.set_ylabel("arrival rate")
ax.set_ylim(0, 0.9); ax.set_title("B. hybrid long-finetune: peak at 40M then entropy-collapse")
ax.legend(fontsize=8, loc="lower left")

# ---- Fig C: 关键候选 arrival vs coll 散点 ----
ax = axes[2]
cands = [("hybrid λ0.05 @40M", 0.647, 0.023, C_H["λ0.05"], "^"),
         ("filter_only 39M", 0.639, 0.033, C_F, "o"),
         ("hybrid λ0.10 @40M", 0.539, 0.027, C_H["λ0.10"], "^"),
         ("hybrid λ0.20 @40M", 0.494, 0.034, C_H["λ0.20"], "^"),
         ("filter_only 80M", 0.401, 0.532, C_F, "s"),
         ("reward_only λ0.1 @20M", 0.662, 0.453, "#ff7f0e", "D"),
         ("naive 78.7M", 0.001, 0.940, "#555", "v")]
for name, a, c, col, mk in cands:
    ax.scatter(c, a, s=90, color=col, marker=mk, edgecolor="k", linewidth=0.5, zorder=3)
    dx = 0.02 if c < 0.5 else -0.02
    ax.annotate(name, (c, a), textcoords="offset points", xytext=(8 * (1 if dx > 0 else -1) - 8, 4),
                fontsize=7.5, ha="left" if dx > 0 else "right")
ax.set_xscale("log")
ax.set_xlabel("collision env fraction (log)"); ax.set_ylabel("arrival rate")
ax.set_xlim(0.015, 1.5); ax.set_ylim(0, 0.85); ax.grid(alpha=0.3)
ax.set_title("C. arrival vs collision trade-off (best ckpt per arm)")
ax.axvline(0.05, color="#888", ls=":", lw=1)
ax.text(0.06, 0.82, "usable: coll < 5%", fontsize=8, color="#555")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"[plot_d2_lambda] saved -> {OUT}")
print("\nbest candidates:")
for c in cands: print(f"  {c[0]:24s} arrival={c[1]:.3f} coll={c[2]:.3f}")
