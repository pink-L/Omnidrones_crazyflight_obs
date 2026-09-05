#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2-16 CBF-sensitivity / perturbation robustness chart (2026-09-06 00:0x).

Data: deterministic eval_ckpt@600 / 1024 env, D2-16 (16 obs / ±2.2m / w8 shaping).
Command-space perturbation injected on the raw policy action BEFORE the CBF runtime
filter (eval_ckpt.py +perturb=cmd_gauss/cmd_pulse). Arms (w8):
  corr-hybrid  = penalty_src=correction λ0.05 @6.5M (mu9vegr8)
  gauss-hybrid = penalty_src=gaussian σ0.5 λ0.1 @6.5M (z36ag4ib)
  filter_only  @20M (vaa0vtzz)
  reward_only  λ0.05 @20M (5m9d89kj, NO runtime filter)
Conclusion: arms with a CBF runtime filter (corr/gauss/filter_only) hold collision at
  a 1-3% plateau across all perturbations; reward_only (no filter) stays ~44%.
See drones/m2_plan.md #11.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/lz/lzspace/drones"
OUT = os.path.join(REPO, "figures", "d2_cbf_robustness.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

X = ["none", "gauss\nσ0.3", "gauss\nσ0.6", "pulse\n0.6"]
COLL = {  # arm -> [none, gauss0.3, gauss0.6, pulse0.6] collision env%
    "corr-hybrid λ0.05\n(runtime filter)": [1.8, 2.1, 1.4, 1.1],
    "gauss σ0.5 λ0.1\n(runtime filter)":   [0.9, 1.7, 1.1, 1.2],
    "filter_only w8\n(runtime filter)":    [2.9, 2.7, 1.7, 1.8],
    "reward_only w8\n(no runtime filter)": [44.0, 41.9, 43.6, 45.5],
}
COLORS = {"corr-hybrid λ0.05\n(runtime filter)": "#d62728",
          "gauss σ0.5 λ0.1\n(runtime filter)":   "#2ca02c",
          "filter_only w8\n(runtime filter)":    "#1f77b4",
          "reward_only w8\n(no runtime filter)": "#7f7f7f"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
x = np.arange(len(X))

for arm, col in COLL.items():
    c = COLORS[arm]
    if "no runtime" in arm:
        ax = axes[1]
        ax.plot(x, col, marker="o", lw=2.2, color=c, label=arm.replace("\n", " "))
    else:
        ax = axes[0]
        ax.plot(x, col, marker="o", lw=2.2, color=c, label=arm.replace("\n", " "))
    for xi, v in zip(x, col):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=c)

for ax, title, yl in ((axes[0], "arms WITH CBF runtime filter", (0, 6.0)),
                      (axes[1], "reward_only (NO runtime filter)", (0, 52))):
    ax.set_xticks(x)
    ax.set_xticklabels(X)
    ax.set_ylim(*yl)
    ax.set_xlabel("command-space perturbation")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

axes[0].set_ylabel("collision env% (deterministic eval @600)")
fig.suptitle("D2-16 CBF robustness under command perturbation (runtime filter on)")
note = ("perturbation injected on raw policy action before CBF filter; filtered arms hold collision at a ~1-3% plateau,\n"
        "reward_only (no filter) stays ~44% -> runtime CBF filter absorbs unsafe commands under uncertainty (CBF-RL claim)")
fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="dimgray")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
