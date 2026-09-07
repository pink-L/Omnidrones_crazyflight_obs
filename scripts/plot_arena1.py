# Arena1 汇报图: 四臂 v3 训练对照曲线 + 严格单命 eval 柱状图 (2026-09-07)
# 用法 (OmniDrones/scripts, lz_env): python plot_arena1.py
# 产出: /home/lz/lzspace/drones/figures/arena1_training.png, arena1_eval_bars.png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb, os

API = wandb.Api()
RUNS = [  # (label, run_id, color)   wandb project=arena1
    ("dual (hybrid)", "04ykbpii", "#1f77b4"),
    ("filter_only", "gh5c0qf0", "#ff7f0e"),
    ("reward_only", "e5gul4zg", "#2ca02c"),
    ("naive", "z1duerma", "#d62728"),
]
FPB = 1024 * 32                      # frames per training iteration (num_envs*train_every)
FIG = "/home/lz/lzspace/drones/figures"
os.makedirs(FIG, exist_ok=True)

def fetch(rid):
    h = API.run(f"fly-hust/arena1/{rid}").history(samples=100000)
    df = h[["_step"]].copy()
    for k in ["train/stats.curriculum_level", "train/stats.success_rate",
              "train/stats.collision_episodes"]:
        col = k.split("/", 1)[1]            # strip 'train/' -> stats.xxx
        df[col] = h[k] if k in h.columns else np.nan
    df["frames_M"] = df["_step"] * FPB / 1e6
    df["entropy"] = h["entropy"] if "entropy" in h.columns else np.nan
    return df

def roll(x, w=7):
    from pandas import Series
    return Series(x.astype(float)).rolling(w, center=True, min_periods=1).mean().to_numpy()

series = {lbl: fetch(rid) for lbl, rid, _ in RUNS}

# ---------------- 图 1: Arena1 四臂 v3 训练对照曲线 (2x2) ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plots = [("stats.curriculum_level", "Curriculum level (active obstacles)", [0, 16]),
         ("stats.success_rate", "Success rate (window: arr & 0 coll)", [0, 1]),
         ("stats.collision_episodes", "Window collision episodes", [0, 1]),
         ("entropy", "Policy entropy", None)]
for ax, (key, title, ylim) in zip(axes.flat, plots):
    for lbl, _, col in RUNS:
        df = series[lbl]
        y = roll(df[key].to_numpy())
        ax.plot(df["frames_M"], y, label=lbl, color=col, lw=1.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("frames (M)"); ax.grid(alpha=.3)
    if ylim: ax.set_ylim(*ylim)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False)
fig.suptitle("Arena1 four-arm training (fixed crossing, curriculum [0,2,4,8,12,16], 24M)", fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(f"{FIG}/arena1_training.png", dpi=150)
plt.close(fig)

# ---------------- 图 2: Arena1 严格单命 eval 柱状图 ----------------
# joint(arr&0coll); 各臂在其实际档 + @16 参考; ON(带filter)仅 dual/fo 用其最优ON档,
# OFF 全臂用各自档; 本图强调 filter 决定性与 fo>dual。
# 注: Arena1 固定穿越难度高, 绝对 joint 远低于旧 F1(随机起终点); 数字如实呈现任务难度。
E = {  # arm -> {"actual-level": [joint_ON@actual, joint_OFF@actual], "16obs": [ON, OFF]}
    # dual 实际档=4, fo 实际档=8, ro/naive 实际档=2(天然无filter只OFF)
    "dual (hybrid)":  {"act": [0.005, 0.017], "16": [0.039, 0.060]},
    "filter_only":    {"act": [0.035, 0.049], "16": [0.087, 0.058]},
    "reward_only":    {"act": [None, 0.007],  "16": [None, 0.043]},
    "naive":          {"act": [None, 0.056],  "16": [None, 0.056]},
}
ACT_LBL = {"dual (hybrid)": "dual @4", "filter_only": "fo @8",
           "reward_only": "ro @2", "naive": "naive @2"}
COL = {"dual (hybrid)": "#1f77b4", "filter_only": "#ff7f0e",
       "reward_only": "#2ca02c", "naive": "#d62728"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, (dens, lblkey) in zip(axes, [("act", "attained level"), ("16", "@16 obstacles")]):
    arms = list(E.keys())
    x = np.arange(len(arms)); w = 0.36
    on_vals, off_vals = [], []
    for a in arms:
        on, off = E[a][dens]
        on_vals.append(on); off_vals.append(off)
    b_on = ax.bar(x - w/2, [v if v is not None else 0 for v in on_vals], w,
                  label="with runtime CBF filter (ON)" if dens == "act" else None,
                  color=[COL[a] if v is not None else "#cccccc" for a, v in zip(arms, on_vals)],
                  edgecolor="k", linewidth=.5)
    b_off = ax.bar(x + w/2, off_vals, w,
                   label="no runtime filter (OFF)" if dens == "act" else None,
                   color=[COL[a] for a in arms],
                   alpha=.45, edgecolor="k", linewidth=.5, hatch="//")
    for xi, (a, on, off) in enumerate(zip(arms, on_vals, off_vals)):
        if on is not None:
            ax.text(xi - w/2, on + .004, f"{on:.3f}", ha="center", fontsize=8)
        ax.text(xi + w/2, off + .004, f"{off:.3f}", ha="center", fontsize=8)
        if on is None:
            ax.text(xi - w/2, .003, "N/A\n(no filter)", ha="center", fontsize=6, color="#888")
    ax.set_xticks(x); ax.set_xticklabels([ACT_LBL[a] for a in arms], rotation=12)
    ax.set_ylim(0, .15); ax.set_ylabel("joint success (strict single-life)")
    ax.set_title(f"{dens} · {lblkey} (soft_respawn=false, 600 steps)")
    ax.grid(axis="y", alpha=.3)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
fig.suptitle("Arena1 four-arm eval — joint = arr & 0 collision (fixed 5m crossing, low absolute = task difficulty)", fontsize=12)
fig.tight_layout(rect=[0, 0.10, 1, 0.93])
fig.savefig(f"{FIG}/arena1_eval_bars.png", dpi=150)
plt.close(fig)

print("saved:", f"{FIG}/arena1_training.png", f"{FIG}/arena1_eval_bars.png")
