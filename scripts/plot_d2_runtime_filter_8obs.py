#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2 runtime-filter ON/OFF @ 8 obstacles (2026-09-06 01:1x).

Same ckpts as the 16-obstacle study, evaluated at 8 obstacles (num_scene=8, +-2.2m,
obs K=8 full window) to isolate density dependence of the CBF runtime filter.
Deterministic eval_ckpt@600 / 1024 env, w8 shaping.
Arms: corr-hybrid λ0.05 @6.5M (mu9vegr8), gauss σ0.5 λ0.1 @6.5M (z36ag4ib),
  filter_only @20M (vaa0vtzz). Reference: reward_only @20M (5m9d89kj), naive @20M
  (suhpvg5q). See drones/m2_plan.md #13.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/lz/lzspace/drones"
OUT = os.path.join(REPO, "figures", "d2_runtime_filter_8obs.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

X = ["none", "gauss σ0.3"]
ON = {
    "corr-hybrid λ0.05": ([0.7, 0.6], "#d62728"),
    "gauss σ0.5 λ0.1":   ([0.9, 0.3], "#2ca02c"),
    "filter_only w8":    ([1.3, 0.3], "#1f77b4"),
}
OFF = {
    "corr-hybrid λ0.05": [20.0, 20.6],
    "gauss σ0.5 λ0.1":   [20.2, 19.1],
    "filter_only w8":    [19.6, 23.2],
}
NOFILT = {"reward_only w8 (no filter)": 23.8,
          "naive w8 (no CBF)":         27.0}

fig, ax = plt.subplots(figsize=(11, 6))
x = list(range(len(X)))
for name, (on, c) in ON.items():
    ax.plot(x, on, marker="o", lw=2.4, color=c, ls="-", label=f"{name}  runtime filter ON")
    ax.plot(x, OFF[name], marker="o", lw=2.0, color=c, ls="--", label=f"{name}  OFF")
for name, v in NOFILT.items():
    ax.axhline(v, color="#7f7f7f", ls=":", lw=1.6)
    ax.text(1.0, v + 0.6, f"{name} {v:.1f}%", color="#555", fontsize=8, ha="center")
ax.set_yscale("log")
ax.set_ylim(0.1, 80)
ax.set_xticks(x)
ax.set_xticklabels(X)
ax.set_xlabel("command-space perturbation")
ax.set_ylabel("collision env%  (log scale)")
ax.set_title("D2 @8 obstacles: CBF runtime filter ON vs OFF (same ckpt)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
note = ("Even at 8 obstacles (density halved vs 16), turning the runtime filter OFF makes collision jump from <1.5% "
        "to ~19-23% (~15-20x)\n(reward_only/naive baselines 24-27%; 16-obs OFF was 36-41% -> density dependent, but "
        "the filter stays decisive)")
fig.text(0.5, -0.03, note, ha="center", fontsize=9, color="dimgray")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved:", OUT)
