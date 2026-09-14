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
    # [2026-09-14] horizon-independent speed metric (see eval_ckpt.py)
    "arrival_steps_median", "arrival_steps_mean", "arrival_steps_p90",
    # [2026-09-14] A3 samples 2-8 pillars per env.  Reported so the randomization can be
    #   WATCHED (min_active_slots stays 0 by decision: in per-pillar mode ">= K" would mean
    #   "every env must have 8 pillars" and would destroy the range).  A healthy A3 run shows
    #   a spread; a single distinct value means the randomization has stopped varying.
    "active_pillars_min", "active_pillars_mean", "active_pillars_max",
    "active_pillars_distinct",
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
    ap.add_argument("--allow-partial", action="store_true",
                    help="emit the table even if some logs have no [eval_metrics] line")
    args = ap.parse_args()

    runs = load(args.logdir)
    if not runs:
        raise SystemExit(f"[aggregate] no usable logs in {args.logdir}")
    # ---- completeness guard ---------------------------------------------------
    # load() silently skips a log whose [eval_metrics] line is absent - which happens when
    # the aggregator runs while the last eval is still finishing. That produced a table
    # with a '-' column AND a bogus "layout_fp identical: False" verdict (2026-09-12, A1a),
    # so refuse to emit a partial table instead.
    expected = sorted(os.path.basename(p) for p in
                      glob.glob(os.path.join(args.logdir, "s*_on.log")) +
                      glob.glob(os.path.join(args.logdir, "s*_off.log")))
    got = {f"{r['_tag']}.log" for r in runs}
    missing = [e for e in expected if e not in got]
    if missing and not args.allow_partial:
        print(f"[aggregate] ERROR incomplete batch: {len(runs)}/{len(expected)} logs carry "
              f"[eval_metrics]; missing {missing}")
        print("[aggregate] refusing to emit a partial table "
              "(pass --allow-partial to override)")
        return 2
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
            # [2026-09-14] DERIVED, not hard-coded.  This used to be the literal string
            #   "512 envs x 600 steps", which silently mislabelled every report once the
            #   protocol changed (384 x 600 alignment / 384 x 1500 closure, plan 0.6.6).
            #   A wrong protocol banner on a correct table is exactly the kind of error
            #   that makes numbers non-comparable later, so read it from the run records.
            "design": f"{runs[0].get('num_envs')} envs x {runs[0].get('rollout_steps')} "
                      "steps, deterministic MODE, fixed start->goal, "
                      "set_seed=1000+train_seed shared by ON/OFF",
            "note": "runtime_filter=false -> CBF filter runs in SHADOW mode: a_cbf is "
                    "computed for diagnostics only, the action is passed through "
                    "unchanged (identical to no-filter behaviour).",
        },
        "runs": runs,
        "layout_fp": {r["_tag"]: r.get("layout_fp") for r in runs},
        "summary": {},
    }

    # ---- mixed-batch guard ----------------------------------------------------
    # Numbers from different protocols are NOT comparable (plan 0.5.16(3)), so refuse to
    # silently average them: a mixed logdir is almost always a mistake.
    _ne = {r.get("num_envs") for r in runs}
    _rs = {r.get("rollout_steps") for r in runs}
    if len(_ne) > 1 or len(_rs) > 1:
        print(f"[aggregate] *** MIXED PROTOCOL: num_envs={sorted(_ne)} "
              f"rollout_steps={sorted(_rs)} ***")
        print("[aggregate] *** refusing to average across protocols; split the logdir ***")
        raise SystemExit(2)

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

    # [2026-09-14] GATE REDESIGN, decided by the user after A2L3 exposed that three of the
    # nine gates were horizon rulers rather than controller properties (plan 0.6.7):
    #
    # A) arrival@0.2 stays as the PASS/FAIL closure gate, but it saturates at a long horizon
    #    (A2L3 = 0.9983 at 1500 steps vs 0.8034 at 600), so it can only coarse-screen.  The
    #    ladder is ORDERED by the horizon-stable arrival_steps_* metrics instead.
    # B) filter_dependency = arrival_OFF/arrival_ON is NOT a gate any more.  At 1500 steps
    #    arrival_OFF is 0.9957, i.e. 0.0043 from saturation, so the ratio tends to 1 by
    #    construction and the old "PASS" was vacuous.  Replaced by a SPEED-based gate.
    # C) zero_intervention_rate = 0.6001 -> 0.8329 is purely 1 - 245/T: the ABSOLUTE
    #    intervened step count is the same on both horizons (~245 steps), so the rate
    #    describes the arena (how long the near-obstacle transit takes), not the controller,
    #    and >= 0.95 is unreachable in a 6x6 m corridor by construction.  Replaced by an
    #    absolute step budget.
    # D) zero_collision_gate demanded 0 collisions on BOTH sides, so it FAILed at 512x600
    #    only because the filter-OFF run hit once (ON: 0).  That single hit is evidence the
    #    filter DOES something, so the gate now requires 0 with the filter ON and merely
    #    records the OFF count.
    med_on = m("arrival_steps_median")["on_mean"]
    med_off = m("arrival_steps_median")["off_mean"]
    speed_cost = round(med_on / med_off, 4) if (med_on and med_off) else None

    _steps = int(runs[0].get("rollout_steps") or 0)
    int_on = m("intervened_step_frac")["on_mean"]
    int_off = m("intervened_step_frac")["off_mean"]
    int_steps_on = round(int_on * _steps, 1) if (int_on is not None and _steps) else None
    int_steps_off = round(int_off * _steps, 1) if (int_off is not None and _steps) else None

    out["gates"] = {
        # --- A: closure gate (coarse) + the ordering metrics -------------------------
        "arrival@0.2_gate>=0.85": None if arr_on is None else arr_on >= 0.85,
        "arrival_steps_median_ON/OFF": speed_cost,
        # --- B: filter cost, RECORD-ONLY (user decision 2026-09-14 20:0x) ------------
        # The metric is nearly horizon-stable but not exactly: drift is +0.0022 (A2L3) to
        # +0.0092 (A1a), so a hard threshold at 1.10 flips A1a's verdict between protocols
        # (1.0991 at 600 PASS vs 1.1083 at 1500 FAIL).  A verdict that flips with the
        # protocol is not a verdict, so this is a monitor, not a gate.  The reason it is
        # not promoted even with margin: at 1500 steps arrival_OFF is already 0.983-0.996,
        # i.e. the filter barely affects the ARRIVAL RATE any more - its real value shows up
        # in COLLISIONS, and the collision gate below carries that dimension.
        "info_filter_speed_cost_on_over_off(warn>1.10)": speed_cost,
        # --- C: absolute intervention budget (horizon-free) --------------------------
        "intervened_steps_gate<=300": None if int_steps_on is None else int_steps_on <= 300,
        # --- D: the filter must not collide; the shadow run only needs recording -----
        "zero_collision_ON_gate": (m("collision_envs")["on_mean"] == 0
                                   if m("collision_envs")["on_mean"] is not None else None),
        "collision_envs_OFF_recorded": m("collision_envs")["off_mean"],
        "zero_oob_gate": (m("oob_envs_ever")["on_mean"] == 0
                          and m("oob_envs_ever")["off_mean"] == 0),
        "h_min_train_gate>=0": (m("h_min_train")["on_mean"] or 0) >= 0.0
        if m("h_min_train")["on_mean"] is not None else None,
        "stall_gate<=0.10": (m("stall_frac")["on_mean"] or 0) <= 0.10,
        # --- informational only (NOT gates) ------------------------------------------
        "info_filter_dependency_OFF_over_ON(retired)": dep,
        "info_zero_intervention_rate(horizon_ruler)": m("zero_intervention_rate")["on_mean"],
        "info_intervened_steps_ON/OFF": [int_steps_on, int_steps_off],
        # [2026-09-14] NAME THE METRIC.  These used to be one entry called
        # `dropped_relevant_gate`, which conflated two different quantities:
        #   * dropped_relevant_frac      = share of *obstacles* that were relevant but
        #                                  fell outside the K-slot obs window
        #   * dropped_relevant_step_frac = share of *steps* that lost >=1 such obstacle
        # On A2 (ON) they are 0.0032 vs 0.1886 - about 60x apart.  So the old gate said
        # PASS while the plan's 3.2 escalation criterion was already exceeded, i.e. the
        # trigger was masked by a same-prefix name.  This entry now measures exactly what
        # its name says, and the escalation rule is reported separately below.
        "dropped_relevant_frac_gate<0.01": (m("dropped_relevant_frac")["on_mean"] or 0) < 0.01,
    }
    step_on = m("dropped_relevant_step_frac")["on_mean"]
    out["triggers"] = {
        # plan 3.2: "若 ... dropped_relevant > 0 的步占比 > 1%，则提前升级 P4 的 K 部分"
        "K_escalation_step_frac>0.01": (None if step_on is None else step_on > 0.01),
        "dropped_relevant_frac_on": m("dropped_relevant_frac")["on_mean"],
        "dropped_relevant_frac_off": m("dropped_relevant_frac")["off_mean"],
        "dropped_relevant_step_frac_on": step_on,
        "dropped_relevant_step_frac_off": m("dropped_relevant_step_frac")["off_mean"],
    }

    print(f"\n===== acceptance report: {args.label} =====")
    fp_bad = [s for s in seeds
              if out['layout_fp'].get(f's{s}_on') != out['layout_fp'].get(f's{s}_off')]
    print(f"protocol: {out['protocol']['design']}")
    print(f"seeds   : {seeds}"
          + (f"   layout_fp ON/OFF identical per seed: True" if not fp_bad else
             f"   !! layout_fp MISMATCH for seeds {fp_bad} - ON/OFF not the same layout"))
    print()
    print("\n".join(lines))
    print("\ngates (plan 4.1/4.3, redesigned 2026-09-14 - see plan 0.6.7):")
    _warn = []
    for k, v in out["gates"].items():
        if k.startswith("info_"):
            # retired / horizon-dependent quantities kept for the record only.  They must
            # never show up as PASS/FAIL, because that is exactly how the old
            # filter_dependency and zero_intervention_rate gates misled: both looked like
            # real gates but were functions of the rollout length.
            mark = "INFO"
            # [2026-09-14] The speed-cost metric is horizon-stable but not horizon-EXACT:
            # drift between 600 and 1500 steps is +0.0022 (A2L3) to +0.0092 (A1a), up to
            # 0.9%.  It is record-only (see the comment on the gate dict), but exceeding
            # 1.10 still deserves a warning that accumulates for the margin decision.
            if "filter_speed_cost" in k and isinstance(v, float) and v > 1.10:
                _warn.append(f"`filter_speed_cost` = {v} > 1.10 (record-only; needs a "
                             "margin decision before it can gate)")
        elif isinstance(v, bool) or v is None:
            mark = "PASS" if v is True else ("FAIL" if v is False else "n/a")
        else:
            mark = "VALUE"
        print(f"  [{mark}] {k} = {v}")
    for w in _warn:
        print(f"  [WARN] {w}")

    # Output-only quantities, and the plan's escalation rule - deliberately NOT a
    # pass/fail gate: exceeding it does not fail the batch, it demands a decision ("go
    # raise K").  Printing it beside the gates is the whole point of the rename above.
    trg = out["triggers"]
    print("\ntriggers (plan 3.2 - decisions, not pass/fail):")
    tmark = {True: "TRIGGERED", False: "not triggered", None: "n/a"}
    esc = trg["K_escalation_step_frac>0.01"]
    print(f"  [{tmark[esc]}] K escalation rule: dropped_relevant_step_frac(ON) = "
          f"{trg['dropped_relevant_step_frac_on']} (threshold 0.01)")
    print(f"  per-OBSTACLE share dropped_relevant_frac(ON/OFF)      = "
          f"{trg['dropped_relevant_frac_on']} / {trg['dropped_relevant_frac_off']}")
    print(f"  per-STEP share dropped_relevant_step_frac(ON/OFF)     = "
          f"{trg['dropped_relevant_step_frac_on']} / "
          f"{trg['dropped_relevant_step_frac_off']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\n[aggregate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
