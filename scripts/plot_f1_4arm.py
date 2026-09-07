# F1 四臂汇报图: 训练对照曲线 + 严格单命 eval 柱状图 (2026-09-07)
# 用法 (OmniDrones/scripts, lz_env): python plot_f1_4arm.py
# 产出: /home/lz/lzspace/drones/figures/f1_4arm_training.png, f1_4arm_eval_bars.png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb, os

API = wandb.Api()
RUNS = [  # (label, run_id, color)
    ("dual (hybrid)", "pg6q5ji5", "#1f77b4"),
    ("filter_only", "scvypn2v", "#ff7f0e"),
    ("reward_only", "clqz6lyn", "#2ca02c"),
    ("naive", "rlf9l0bn", "#d62728"),
]
FPB = 1024 * 32                      # frames per training iteration (num_envs*train_every)
FIG = "/home/lz/lzspace/drones/figures"
os.makedirs(FIG, exist_ok=True)

def fetch(rid):
    h = API.run(f"fly-hust/new_reward/{rid}").history(samples=100000)
    df = h[["_step"]].copy()
    for k in ["train/stats.curriculum_level", "train/stats.success_rate",
              "train/stats.collision_episodes", "train/stats.cbf_violation"]:
        col = k.split("/", 1)[1]            # strip 'train/' -> stats.xxx
        df[col] = h[k] if k in h.columns else np.nan
    df["frames_M"] = df["_step"] * FPB / 1e6
    df["entropy"] = h["entropy"] if "entropy" in h.columns else np.nan
    return df

def roll(x, w=7):
    from pandas import Series
    return Series(x.astype(float)).rolling(w, center=True, min_periods=1).mean().to_numpy()

series = {lbl: fetch(rid) for lbl, rid, _ in RUNS}

# ---------------- 图 1: 训练对照曲线 (2x2) ----------------
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
fig.suptitle("F1 four-arm training comparison (from-scratch curriculum [2,4,8,16], 12M)", fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(f"{FIG}/f1_4arm_training.png", dpi=150)
plt.close(fig)

# ---------------- 图 2: 严格单命 eval 柱状图 ----------------
# 每臂 joint(arr&0coll) @{8,16}obs; 带 filter(ON, 仅 dual/fo) vs 无 filter(OFF)
E = {  # arm -> {8obs:[joint_ON, joint_OFF], 16obs:[joint_ON, joint_OFF]}
    "dual (hybrid)":  {"8obs": [0.971, 0.787], "16obs": [0.944, 0.616]},
    "filter_only":    {"8obs": [0.980, 0.771], "16obs": [0.932, 0.626]},
    "reward_only":    {"8obs": [None, 0.764],  "16obs": [None, 0.565]},
    "naive":          {"8obs": [None, 0.755],  "16obs": [None, 0.641]},
}
COL = {"dual (hybrid)": "#1f77b4", "filter_only": "#ff7f0e",
       "reward_only": "#2ca02c", "naive": "#d62728"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, dens in zip(axes, ["8obs", "16obs"]):
    arms = list(E.keys())
    x = np.arange(len(arms)); w = 0.36
    on_vals, off_vals = [], []
    for a in arms:
        on, off = E[a][dens]
        on_vals.append(on); off_vals.append(off)
    b_on = ax.bar(x - w/2, [v if v is not None else 0 for v in on_vals], w,
                  label="with runtime CBF filter (ON)" if dens == "8obs" else None,
                  color=[COL[a] if v is not None else "#cccccc" for a, v in zip(arms, on_vals)],
                  edgecolor="k", linewidth=.5)
    b_off = ax.bar(x + w/2, off_vals, w,
                   label="no runtime filter (OFF)" if dens == "8obs" else None,
                   color=[COL[a] for a in arms],
                   alpha=.45, edgecolor="k", linewidth=.5, hatch="//")
    for xi, (a, on, off) in enumerate(zip(arms, on_vals, off_vals)):
        if on is not None:
            ax.text(xi - w/2, on + .015, f"{on:.2f}", ha="center", fontsize=9)
        ax.text(xi + w/2, off + .015, f"{off:.2f}", ha="center", fontsize=9)
        if on is None:
            ax.text(xi - w/2, .02, "N/A\n(no filter)", ha="center", fontsize=7, color="#888")
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=12)
    ax.set_ylim(0, 1.08); ax.set_ylabel("joint success (strict single-life)")
    ax.set_title(f"{dens} · 严格单命 (soft_respawn=false, 600 steps)")
    ax.grid(axis="y", alpha=.3)
    ax.set_title(f"{dens} obstacles - strict single-life, 600 steps")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
fig.suptitle("F1 four-arm eval @8/16 obstacles — joint = arr & 0 collision", fontsize=13)
fig.tight_layout(rect=[0, 0.08, 1, 0.94])
fig.savefig(f"{FIG}/f1_4arm_eval_bars.png", dpi=150)
plt.close(fig)

print("saved:", f"{FIG}/f1_4arm_training.png", f"{FIG}/f1_4arm_eval_bars.png")
