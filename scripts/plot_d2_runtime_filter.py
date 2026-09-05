#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2-16 runtime-filter contribution chart (2026-09-06 00:4x).

Deterministic eval_ckpt@600 / 1024 env, D2-16 (16 obs / ±2.2m / w8 shaping).
Same checkpoint evaluated with the CBF runtime filter ON vs OFF (+runtime_filter=false,
eval_ckpt.py) under command-space perturbation {none, gauss σ0.3, σ0.6}.
Arms (w8): corr-hybrid λ0.05 @6.5M (mu9vegr8), gauss σ0.5 λ0.1 @6.5M (z36ag4ib),
  filter_only @20M (vaa0vtzz). Reference lines: reward_only @20M (5m9d89kj, no filter),
  naive @20M (suhpvg5q, no CBF).
Conclusion: turning the runtime filter OFF makes collision jump from ~1-3% to ~36-41%
  (same level as naive/reward_only ~44-47%) -> the CBF runtime filter is the decisive
  safety component at deployment; the trained policies alone are not safe enough.
See drones/m2_plan.md #12.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/lz/lzspace/drones"
OUT = os.path.join(REPO, "figures", "d2_runtime_filter.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

X = ["none", "gauss σ0.3", "gauss σ0.6"]
ON = {
    "corr-hybrid λ0.05": ([1.8, 2.1, 1.4], "#d62728"),
    "gauss σ0.5 λ0.1":   ([0.9, 1.7, 1.1], "#2ca02c"),
    "filter_only w8":    ([2.9, 2.7, 1.7], "#1f77b4"),
}
OFF = {
    "corr-hybrid λ0.05": [38.4, 36.5, 37.1],
    "gauss σ0.5 λ0.1":   [39.4, 36.5, 36.3],
    "filter_only w8":    [37.6, 39.2, 40.5],
}
NOFILT = {"reward_only w8 (no filter)": [44.0, 41.9, 43.6],
          "naive w8 (no CBF)":         [47.0, 47.9, 45.7]}

fig, ax = plt.subplots(figsize=(11, 6))
x = list(range(len(X)))
for name, (on, c) in ON.items():
    ax.plot(x, on, marker="o", lw=2.4, color=c, ls="-", label=f"{name}  runtime filter ON")
    ax.plot(x, OFF[name], marker="o", lw=2.0, color=c, ls="--", alpha=0.9,
            label=f"{name}  OFF")
for name, v in NOFILT.items():
    ax.plot(x, v, marker="s", lw=1.6, color="#7f7f7f", ls=":", label=name)
for i, v in enumerate([2.9, 2.7, 1.7]):  # annotate filter_on point
    pass
ax.set_yscale("log")
ax.set_ylim(0.5, 80)
ax.set_xticks(x)
ax.set_xticklabels(X)
ax.set_xlabel("command-space perturbation")
ax.set_ylabel("collision env%  (log scale)")
ax.set_title("D2-16: CBF runtime filter ON vs OFF (same ckpt) under command perturbation")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
note = ("Turning the runtime filter OFF makes collision jump from ~1-3% to ~36-41% (level of naive/reward_only ~44-47%)\n"
        "-> the CBF runtime filter is the decisive safety component at deployment; trained policies alone are not safe enough.")
fig.text(0.5, -0.03, note, ha="center", fontsize=9, color="dimgray")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
