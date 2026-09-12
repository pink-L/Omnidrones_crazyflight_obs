#!/usr/bin/env python
"""Aggregate NavVel acceptance-eval logs into a plan-§4.2/§4.3 report + gate verdict.

Reads `<logdir>/s<seed>_<on|off>.log` files produced by `scripts/acceptance_eval.sh`
(each contains one `[eval_metrics] {...}` JSON line) and emits:

  * a markdown table (per-seed + mean/range) for both filter modes,
  * the plan §4.2 (阶段 1) and §4.3 (阶段 2) gate verdicts,
  * a machine-readable JSON (--out) to archive next to the model artifacts.

Usage:
  python scripts/aggregate_acceptance_eval.py \
      --logdir /tmp/navvel_eval \
      --out /home/lz/lzspace/navvel_export/<model_id>/eval_metrics.json \
      --label "A0-legacy"
"""

import argparse
import glob
import json
import os
import re
import statistics as st

METRIC_ORDER = [
    "arrival_at", "collision_envs", "collision_edges", "oob_envs_ever",
    "crash_envs_ever", "min_clearance_global_min", "h_min_train",
    "zero_intervention_rate", "intervened_step_frac", "corr_mean", "corr_p50",
    "corr_p95", "stall_frac", "dropped_relevant_frac",
    "dropped_relevant_step_frac",
]


def load(logdir):
    runs = []
    for path in sorted(glob.glob(os.path.join(logdir, "s*_on.log")) +
                       glob.glob(os.path.join(logdir, "s*_off.log"))):
        m = re.search(r"([^/]+)\.log$", path)
        tag = m.group(1)
        mode = "on" if tag.endswith("_on") else "off"
        seed = int(tag[1:].split("_")[0])
        rec = None
        with open(path) as fh:
            for line in fh:
                if line.startswith("[eval_metrics] "):
                    rec = json.loads(line[len("[eval_metrics] "):])
        if rec is None:
            print(f"  !! {tag}: no [eval_metrics] line (run failed?)")
            continue
        rec["_tag"] = tag
        rec["_mode"] = mode
        rec["_seed"] = seed
        runs.append(rec)
    return sorted(runs, key=lambda r: (r["_seed"], r["_mode"]))


def _num(v):
    return None if v is None else float(v)


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return (round(st.mean(vals), 4),
            round(max(vals) - min(vals), 4) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="unnamed")
    args = ap.parse_args()

    runs = load(args.logdir)
    if not runs:
        raise SystemExit(f"[aggregate] no usable logs in {args.logdir}")
    seeds = sorted({r["_seed"] for r in runs})
    modes = ["on", "off"]
    by = {(r["_seed"], r["_mode"]): r for r in runs}

    def cell(seed, mode, key, sub=None):
        r = by.get((seed, mode))
        if r is None:
            return None
        v = r.get(key)
        if sub is not None:
            v = (v or {}).get(sub)
        return _num(v)

    out = {
        "label": args.label,
        "protocol": {
            "task": runs[0].get("task"),
            "eval_points": runs[0].get("eval_points"),
            "rollout_steps": runs[0].get("rollout_steps"),
            "num_envs": runs[0].get("num_envs"),
            "seeds": seeds,
            "design": "512 envs x 600 steps, deterministic MODE, fixed start->goal, "
                      "set_seed=1000+train_seed shared by ON/OFF",
            "note": "runtime_filter=false -> CBF filter runs in SHADOW mode: a_cbf is "
                    "computed for diagnostics only, the action is passed through "
                    "unchanged (identical to no-filter behaviour).",
        },
        "runs": runs,
        "layout_fp": {r["_tag"]: r.get("layout_fp") for r in runs},
        "summary": {},
    }

    # ---- table ---------------------------------------------------------------
    rows = []
    for seed in seeds:
        for mode in modes:
            rows.append((seed, mode))
    lines = []
    hdr = "| metric | " + " | ".join(f"s{s} {m.upper()}" for s, m in rows) + " | ON mean | OFF mean |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(rows) + 3))

    def fmt(v):
        return "-" if v is None else f"{v:.4f}".rstrip("0").rstrip(".")

    def row(label, key, sub=None):
        vals = [cell(s, m, key, sub) for s, m in rows]
        on_m, on_r = mean_sd([v for (s, m), v in zip(rows, vals) if m == "on"])
        off_m, off_r = mean_sd([v for (s, m), v in zip(rows, vals) if m == "off"])
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals)
                     + f" | {fmt(on_m)} | {fmt(off_m)} |")
        out["summary"][label] = {"on_mean": on_m, "off_mean": off_m,
                                 "on_range": on_r, "off_range": off_r,
                                 "per_run": vals}

    for r in sorted({k for x in runs for k in x if isinstance(x.get(k), dict)}):
        for sub in sorted(runs[0][r].keys()):
            row(f"arrival@{sub}", r, sub)
    for k in METRIC_ORDER:
        if k == "arrival_at":
            continue
        if any(k in x for x in runs):
            row(k, k)

    # ---- gates ---------------------------------------------------------------
    def m(key):
        return out["summary"].get(key, {})

    arr_off = m("arrival@0.2")["off_mean"]
    arr_on = m("arrival@0.2")["on_mean"]
    dep = round(arr_off / arr_on, 4) if (arr_off and arr_on) else None
    out["gates"] = {
        "filter_dependency_OFF_over_ON": dep,
        "filter_dependency_gate>=0.95": None if dep is None else dep >= 0.95,
        "zero_intervention_rate_gate>=0.95": (m("zero_intervention_rate")["on_mean"] or 0) >= 0.95
        if m("zero_intervention_rate")["on_mean"] is not None else None,
        "h_min_train_gate>=0": (m("h_min_train")["on_mean"] or 0) >= 0.0
        if m("h_min_train")["on_mean"] is not None else None,
        "arrival@0.2_gate>=0.85": None if arr_on is None else arr_on >= 0.85,
        "zero_collision_gate": (m("collision_envs")["on_mean"] == 0
                                and m("collision_envs")["off_mean"] == 0),
        "zero_oob_gate": (m("oob_envs_ever")["on_mean"] == 0
                          and m("oob_envs_ever")["off_mean"] == 0),
        "stall_gate<=0.10": (m("stall_frac")["on_mean"] or 0) <= 0.10,
        "dropped_relevant_gate<0.01": (m("dropped_relevant_frac")["on_mean"] or 0) < 0.01,
    }

    print(f"\n===== acceptance report: {args.label} =====")
    print(f"protocol: {out['protocol']['design']}")
    print(f"seeds   : {seeds}   layout_fp ON/OFF identical per seed: "
          + str(all(out['layout_fp'].get(f's{s}_on') == out['layout_fp'].get(f's{s}_off')
                    for s in seeds)))
    print()
    print("\n".join(lines))
    print("\ngates (plan 4.1/4.3):")
    for k, v in out["gates"].items():
        if isinstance(v, bool) or v is None:
            mark = "PASS" if v is True else ("FAIL" if v is False else "n/a")
        else:
            mark = "VALUE"
        print(f"  [{mark}] {k} = {v}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\n[aggregate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
