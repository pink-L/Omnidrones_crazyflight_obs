#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2-16 四臂消融确定性 eval 绘图（M2 收官出图，2026-09-05 17:5x 会话生成）。

确定性 eval 口径：eval_ckpt.py task=NavVel algo=ppo headless +rollout_steps=600
  1024 env、16 障碍/场、场 ±2.2、obs 最近-8 滑动窗口、brake off、各臂 cbf.mode 与训练一致。

图：
  Fig A  D2-16 四臂终值确定性 eval 分组柱状 (arrival / joint success / collision env%)
  Fig B  filter_only 沿训练步数的退化折线（39/46/52/80M 确定性 eval：arrival & collision env%）
  Fig C  D1(低密度) vs D2-16(高密度) 各臂到达率对照，展示密度升级下的机制差异

数据源注释：
  * D2 数字 = 本次会话 eval_ckpt@600 实测（17:52-17:57）。
    - naive/reward_only/hybrid 无 checkpoint_final（最终 eval 4 进程并行 OOM 崩溃），
      用各自 checkpoint_78675968.pt(~78.7M) 评估；filter_only/reward_only 用 checkpoint_final.pt(80M)。
    - filter_only 因 >60M 末期退化(collision 飙升)，补评中段 ckpt：39/46/52M 定位峰值(39M)。
  * D1 数字 = 交接摘要#2 (m2_plan.md) D1 四臂 train-final eval / 确定性口径（8障/5.6m，20-40M）。

运行：  python scripts/plot_d2_ablation.py
输出：  <repo>/figures/d2_ablation.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/lz/lzspace/drones"
OUT_DIR = os.path.join(REPO, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "d2_ablation.png")

# ---------------------------------------------------------------- D2-16 eval data
# (arm, ckpt_frames_M, arrival_rate, joint_success, collision_env_frac)
D2_FINAL = [
    ("naive",      78.7, 0.001, 0.000, 0.940),
    ("reward_only", 80.0, 0.002, 0.000, 0.854),
    ("hybrid",     78.7, 0.318, 0.312, 0.106),
    ("filter_only",80.0, 0.401, 0.343, 0.532),
]
# filter_only 中段峰值 (reference: 39M)
FO_PEAK = ("filter_only (39M peak)", 39.0, 0.639, 0.627, 0.033)

# filter_only 沿步数： (frames_M, arrival_rate, joint_success, collision_env_frac)
FO_TREND = [
    (39.0, 0.639, 0.627, 0.033),
    (46.0, 0.599, 0.581, 0.037),
    (52.0, 0.502, 0.466, 0.076),
    (80.0, 0.401, 0.343, 0.532),
]

# ---------------------------------------------------------------- D1 reference
# (arm, arrival, collision_episodes); source = m2_plan.md 交接摘要#2
D1 = [
    ("naive",       0.824),
    ("reward_only", 0.580),
    ("hybrid",      0.916),
    ("filter_only", 0.934),
]
D2_ARR = [r[2] for r in D2_FINAL]

# ---------------------------------------------------------------- colors / labels
COLORS = {"naive": "#d62728", "reward_only": "#ff7f0e", "hybrid": "#2ca02c",
          "filter_only": "#1f77b4", "filter_only_peak": "#9467bd"}
ORDER = ["naive", "reward_only", "hybrid", "filter_only"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
fig.suptitle("D2-16 (16 obstacles/env, field $\\pm$2.2 m, obs=nearest-8 sliding window)  —  deterministic eval @600 steps (1024 env)",
             fontsize=12, y=0.99)

# ---------------- Fig A: D2-16 四臂终值 grouped bars ----------------
ax = axes[0]
labels = [r[0] for r in D2_FINAL]
x = range(len(labels))
w = 0.26
MET = {"arrival": "#1f77b4", "joint": "#2ca02c", "collision": "#d62728"}
HATCH = {"naive": "", "reward_only": "///", "hybrid": "\\\\", "filter_only": "xx"}
b1 = ax.bar([i - w for i in x], [r[2] for r in D2_FINAL], w, label="arrival rate",
            color=MET["arrival"], edgecolor=[COLORS[lab] for lab in labels], hatch=[HATCH[l] for l in labels])
b2 = ax.bar([i for i in x],     [r[3] for r in D2_FINAL], w, label="joint success",
            color=MET["joint"], edgecolor=[COLORS[lab] for lab in labels], hatch=[HATCH[l] for l in labels])
b3 = ax.bar([i + w for i in x], [r[4] for r in D2_FINAL], w, label="collision env frac",
            color=MET["collision"], edgecolor=[COLORS[lab] for lab in labels], hatch=[HATCH[l] for l in labels])
ax.axhline(FO_PEAK[3], color=COLORS["filter_only_peak"], ls="--", lw=1.2)
ax.text(3.5, FO_PEAK[3] + 0.02, "filter_only 39M peak", color=COLORS["filter_only_peak"], fontsize=8, ha="right")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
ax.set_title("A. 4-arm final (78.7-80M) eval")
ax.legend(fontsize=8, loc="upper right")
for i, r in enumerate(D2_FINAL):
    ax.text(i - w, r[2] + 0.02, f"{r[2]:.3f}", ha="center", fontsize=7.5)
    ax.text(i,     r[3] + 0.02, f"{r[3]:.3f}", ha="center", fontsize=7.5)
    ax.text(i + w, r[4] + 0.02, f"{r[4]:.3f}", ha="center", fontsize=7.5)

# ---------------- Fig B: filter_only 退化轨迹 ----------------
ax = axes[1]
fM = [t[0] for t in FO_TREND]; fArr = [t[1] for t in FO_TREND]; fColl = [t[3] for t in FO_TREND]
ax.plot(fM, fArr, "-o", color=COLORS["filter_only"], label="arrival rate")
ax.plot(fM, fColl, "-s", color="#d62728", label="collision env frac")
ax.fill_between(fM, 0, 0.2, color="#ffcccc", alpha=0.35, label="peak window (arr>=0.5, coll<0.1)")
for i, (a, c) in enumerate(zip(fArr, fColl)):
    ax.annotate(f"{a:.2f}", (fM[i], a), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.annotate(f"{c:.2f}", (fM[i], c), textcoords="offset points", xytext=(0, -14), fontsize=8, ha="center", color="#a00")
ax.axvspan(36, 55, color="#ccc", alpha=0.25)
ax.annotate("late-train degradation\n(>60M entropy diverge)", xy=(74, 0.48), xytext=(60, 0.72),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("training frames (M)"); ax.set_ylabel("fraction")
ax.set_ylim(-0.03, 1.05); ax.set_title("B. filter_only degradation (eval @39/46/52/80M)")
ax.legend(fontsize=8, loc="center left")

# ---------------- Fig C: D1 vs D2 arrival 对照 ----------------
ax = axes[2]
D2_ARR_END = {"naive": 0.001, "reward_only": 0.002, "hybrid": 0.318, "filter_only": 0.401}
D2_ARR_PEAK = {"naive": 0.001, "reward_only": 0.002, "hybrid": 0.318, "filter_only": 0.639}
x = range(len(ORDER)); w = 0.27
ax.bar([i - w for i in x], [dict(D1)[k] for k in ORDER], w, label="D1 (8 obs / 5.6m)", color="#555")
ax.bar([i for i in x], [D2_ARR_END[k] for k in ORDER], w, label="D2-16 final (78.7-80M)", color=[COLORS[k] for k in ORDER], alpha=0.9)
ax.bar([i + w for i in x], [D2_ARR_PEAK[k] for k in ORDER], w, label="D2-16 best-ckpt", color=[COLORS[k] for k in ORDER], alpha=0.45)
for i, k in enumerate(ORDER):
    ax.text(i - w, dict(D1)[k] + 0.02, f"{dict(D1)[k]:.2f}", ha="center", fontsize=7.5, color="#333")
    ax.text(i + w, D2_ARR_PEAK[k] + 0.02, f"{D2_ARR_PEAK[k]:.3f}", ha="center", fontsize=7)
ax.set_xticks(list(x)); ax.set_xticklabels(ORDER, rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel("arrival rate")
ax.set_title("C. density upgrade: arrival (D1 vs D2-16)")
ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"[plot_d2_ablation] saved -> {OUT}")

# text summary for quick copy into report
print("\n===== D2-16 deterministic eval summary (eval_ckpt @600) =====")
hdr = f"{'arm':12s} {'ckpt(M)':8s} {'arrival':8s} {'joint':7s} {'coll_env':9s}"
print(hdr)
for r in D2_FINAL:
    print(f"{r[0]:12s} {r[1]:<8.1f} {r[2]:<8.3f} {r[3]:<7.3f} {r[4]:<9.3f}")
print(f"{'filter_only*':12s} {FO_PEAK[1]:<8.1f} {FO_PEAK[2]:<8.3f} {FO_PEAK[3]:<7.3f} {FO_PEAK[4]:<9.3f}")
print("* best mid-training ckpt (39M); final 80M ckpt shows late-train degradation")
