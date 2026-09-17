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
    # [2026-09-16] plan 4.2 rows that were never collected before, plus the three clearance
    #   conventions.  These MUST be listed here: `row()` is only called for keys in this
    #   list, so a metric that eval_ckpt emits but this list omits never reaches
    #   out["summary"] and every gate reading it silently becomes None.  That is how
    #   `min d_min` went missing from the gate table in the first place, and it is the same
    #   way the clearance pads vanished on the first run of the new code.
    "clearance_pad_inflation", "clearance_pad_center",
    "min_clearance_surface_global_min", "min_clearance_center_global_min",
    "min_clearance_surface_env_mean",
    "z_err_rmse", "z_err_n", "terminal_z_err",
    "terminal_speed_xy", "terminal_speed_win_steps", "terminal_speed_win_sec",
    "path_length_ratio",
    # `cbf_extra` is the CBF's own extra margin = safety_radius_extra(...) = r_safety_margin
    #   when the brake term is off (profile A).  It is the quantity plan 4.2's numeric d_min
    #   bar actually constrains, so it has to reach the summary to be gateable.
    "cbf_extra",
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
    ap.add_argument("--baseline", default=None, metavar="AGG.JSON",
                    help="agg.json of the plan-4.2 baseline (A0).  Needed to decide the "
                         "two 'not worse than A0' rows; without it they print n/a, which "
                         "is indistinguishable from 'not measured' (pit 14).")
    ap.add_argument("--d-min-inflation-pad", type=float, default=None, metavar="M",
                    help="legacy logdirs only: the `inflation` cushion, needed to convert "
                         "min_clearance_global_min into the SURFACE convention.  Profile A "
                         "uses 0.02.  Do not hardcode; eval_ckpt reports it as "
                         "clearance_pad_inflation.")
    ap.add_argument("--d-min-pad", type=float, default=None, metavar="M",
                    help="legacy logdirs only: inflation + drone_radius, needed to convert "
                         "min_clearance_global_min into the CENTRE convention.  Profile A "
                         "uses 0.12.  Do not hardcode; eval_ckpt reports "
                         "clearance_pad_center.")
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
    # [P3 2026-09-17] `.get()` not `[]`: the naive arm (cbf.mode=none) has no CBF
    #   transform, so eval_ckpt.py never emits `intervened_step_frac` (unlike
    #   `h_min_train`/`zero_intervention_rate` which it emits as None).  `m()` then
    #   returns {} and the old `["on_mean"]` raised KeyError, aborting the whole
    #   aggregation.  A missing CBF metric must read as n/a, never as a crash (pit 14).
    int_on = m("intervened_step_frac").get("on_mean")
    int_off = m("intervened_step_frac").get("off_mean")
    int_steps_on = round(int_on * _steps, 1) if (int_on is not None and _steps) else None
    int_steps_off = round(int_off * _steps, 1) if (int_off is not None and _steps) else None

    # [2026-09-16] CLEARANCE CONVENTIONS - the plan's `min d_min >= 0.10` is UNDECIDED.
    #   `ObstacleManager.clearances()` returns |p-p_oi| - (r_oi + drone_radius + inflation).
    #   The env header (nav_vel_obstacles.py L27-28) says drone_radius is the drone's
    #   physical sphere and inflation is an extra cushion, so that raw number is
    #   drone-surface-to-obstacle-surface MINUS the cushion.  Three documents then reuse the
    #   name `d_min` for three different quantities, and the difference is not cosmetic on
    #   profile A (pads 0.02 / 0.12 m): the observed minimum is 0.0501, so
    #       cbf     = 0.0501  -> FAIL  (>= 0.10)
    #       surface = 0.0701  -> FAIL
    #       centre  = 0.1701  -> PASS
    #   Earlier I "resolved" this by adding drone_radius+inflation back and declaring PASS,
    #   which is only right under the CENTRE reading.  That was a guess dressed as a
    #   conversion, so now all three are reported and the gate is not decided by me.
    raw_on = m("min_clearance_global_min").get("on_mean")
    pad_inf = m("clearance_pad_inflation").get("on_mean")
    pad_cen = m("clearance_pad_center").get("on_mean")
    if pad_inf is None:
        pad_inf = args.d_min_inflation_pad      # legacy logdirs only
    if pad_cen is None:
        pad_cen = args.d_min_pad
    dmin_cbf = raw_on
    dmin_surface = m("min_clearance_surface_global_min").get("on_mean")
    if dmin_surface is None and raw_on is not None and pad_inf is not None:
        dmin_surface = round(raw_on + pad_inf, 4)
    dmin_center = m("min_clearance_center_global_min").get("on_mean")
    if dmin_center is None and raw_on is not None and pad_cen is not None:
        dmin_center = round(raw_on + pad_cen, 4)
    out["clearance_conventions"] = {
        "raw_is": "cbf  (|p-p_oi| - (r_oi+drone_radius+inflation)); the CBF acts on this",
        "pad_inflation_m": pad_inf,
        "pad_center_m": pad_cen,
        "min_cbf": dmin_cbf,
        "min_surface(=cbf+pad_inflation)": dmin_surface,
        "min_center(=cbf+pad_center)": dmin_center,
        "plan_4.2_text": "`min d_min` | 全程最小表面净空 | >= 0.10 m",
        "UNRESOLVED": ("plan 4.2 says 'surface clearance' but never fixes the origin; "
                       "under `centre` A3/A4 pass 0.10, under `surface`/`cbf` they fail. "
                       "Which one is meant must be settled in the plan, not inferred "
                       "from which reading lets the rung pass."),
    }

    zn = m("z_err_rmse").get("on_mean")
    vt = m("terminal_speed_xy").get("on_mean")
    plr = m("path_length_ratio").get("on_mean")

    cbf_extra_on = m("cbf_extra").get("on_mean")

    # [2026-09-16] plan 4.2's two RELATIVE rows ("z_err RMSE / 终端速度 | 不劣于 A0").
    #   They were `None` for as long as A0 had never been measured, and a permanent `n/a`
    #   in a gate table reads exactly like a gate that was checked and passed - the same
    #   failure mode this whole step exists to remove (pit 14).  So the baseline is an
    #   input, and a delta smaller than the seed spread of either side is NOT reported as
    #   a pass on merit (A4 vs A2L3 looked ordered at 3 seeds and dissolved at 5).
    rel = {}
    _bnote = None
    if args.baseline:
        try:
            with open(args.baseline) as fh:
                _bj = json.load(fh)
            _bp, _bs = _bj.get("protocol", {}), _bj.get("summary", {})
            _here = (runs[0].get("num_envs"), runs[0].get("rollout_steps"))
            if (_bp.get("num_envs"), _bp.get("rollout_steps")) != _here:
                _bnote = (f"baseline is {_bp.get('num_envs')}x{_bp.get('rollout_steps')} "
                          f"but this batch is {_here[0]}x{_here[1]}: refused")
            else:
                for _k, _dirn in (("z_err_rmse", "lower"),
                                  ("terminal_speed_xy", "lower")):
                    _b, _r = _bs.get(_k, {}), m(_k)
                    _bon, _ron = _b.get("on_mean"), _r.get("on_mean")
                    if _bon is None or _ron is None:
                        _bnote = f"{_k}: baseline or batch value missing"
                        rel[_k] = None
                        continue
                    _d = round(_ron - _bon, 4)
                    _spread = max(_b.get("on_range") or 0.0, _r.get("on_range") or 0.0)
                    _nw = (_d <= 0) if _dirn == "lower" else (_d >= 0)
                    _tag = ("within seed spread" if abs(_d) <= _spread and _d != 0
                            else ("equal" if _d == 0
                                  else ("better" if _nw else "worse")))
                    rel[_k] = {"pass": bool(_nw), "note": _tag,
                               "baseline_on": _bon, f"{args.label}_on": _ron,
                               "delta": _d, "seed_spread": _spread,
                               "label": _bj.get("label")}
        except Exception as _e:
            _bnote = f"baseline unreadable: {type(_e).__name__}: {_e}"
    else:
        _bnote = ("no --baseline given: A0 has to be measured to decide these rows "
                  "(scripts/compare_to_baseline.py drills into them)")

    # ---- PLAN 4.2: stage 1 (geometric generalisation) --------------------------------
    # 4.1 is explicit that the two stages use DIFFERENT gate sets and disagree about the
    # same metric on purpose: "CBF intervention rate" is "record only, high is allowed" in
    # stage 1 and "zero-intervention rate >= 0.95" in stage 2.  Keeping them in one table
    # is what let me call zero_intervention_rate a stage-1 blocker when 4.1 says it is not.
    out["gates_stage1"] = {
        "arrival@0.2_gate>=0.85": None if arr_on is None else arr_on >= 0.85,
        # Three rows because the plan's convention is undecided; see above.  Whichever the
        # plan picks, the other two stay visible so the choice cannot hide a failure.
        # [2026-09-16] DECISION (user, this date): the numeric clearance bar moves OFF
        #   `min d_min` and ONTO the CBF's own margin.  Reason, established by MEASUREMENT
        #   rather than by reading: with the filter ON the observed `min d_min` tracks
        #   `cbf_extra` and nothing else (A3/A4: cbf_extra = 0.05 -> min = 0.0501 with
        #   h_min = 0.0001), so `min d_min >= 0.10` is not a claim about the policy - it is
        #   a claim about a config constant that no rollout can move.  Gating it would have
        #   made a controller question look answered by a number the controller cannot
        #   influence.  So `min d_min` is RECORDED in all three conventions and the gate
        #   goes on `cbf_extra` itself (which for profile A equals cbf.r_safety_margin).
        #   Caveat kept on the record: A0 is a counter-example to "h_min == 0 always" - its
        #   cbf_extra is 0.10 yet min d_min was 0.0512 with h_min = -0.05, i.e. an
        #   over-constrained filter can overshoot its own boundary.  So h_min >= 0 is
        #   evidence about the filter, not about the clearance floor.
        #   ---
        #   [2026-09-16, second decision] Relocating the bar was not enough.  Measured:
        #   `cbf_extra` = 0.10 for A0 and 0.05 for every 口径-A rung, so the moved gate
        #   PASSED for the old conservative baseline and FAILED for every "improved" rung -
        #   it inverted the ladder.  That is the definition of 口径 A (P0.2 lowered
        #   `cbf.r_safety_margin` 0.10 -> 0.05 to remove useless intervention), not a
        #   statement about any policy.  So the numeric clearance bar is DELETED from plan
        #   4.2 and recorded in all three conventions instead; the operative safety
        #   requirement is the hard "0 collision / 0 OOB" row, which is what
        #   `collision_margin` actually enforces.  A numeric clearance FLOOR, if wanted, is
        #   a TRAINING constraint (ask the policy to keep 10 cm), i.e. a different batch,
        #   not a threshold.  Evidence for that: with the filter OFF the value is still
        #   0.0503 (A4, 5 seeds), so the filter is not what pins it.
        "info_min_d_min_is_record_only": ("plan 4.2 numeric clearance bar deleted "
                                          "2026-09-16; see 0.5.24"),
        "zero_collision_gate(ON and OFF)": (None if m("collision_envs")["on_mean"] is None
                                            else m("collision_envs")["on_mean"] == 0
                                            and m("collision_envs")["off_mean"] == 0),
        "zero_oob_gate(ON and OFF)": (m("oob_envs_ever")["on_mean"] == 0
                                      and m("oob_envs_ever")["off_mean"] == 0),
        "dropped_relevant_frac_gate<0.01": (m("dropped_relevant_frac")["on_mean"] or 0) < 0.01,
        # "not worse than A0" - needs the v1.0.0 baseline to compare, so these stay n/a until
        # the baseline values are recorded; the guide's absolute bars are 0.10 m / 0.15 m/s.
        # Relative rows; `None` when no --baseline was supplied (and then the report says
        # so out loud rather than leaving a bare n/a that looks like a pass).
        "z_err_rmse_gate(not worse than A0)": (rel.get("z_err_rmse") or {}).get("pass"),
        "terminal_speed_gate(not worse than A0)": (rel.get("terminal_speed_xy") or {}).get("pass"),
        # ---- stage-1 RECORD-ONLY (4.2 says "record", not gate) ----------------------
        "info_stage1_cbf_intervened_step_frac": m("intervened_step_frac").get("on_mean"),
        "info_stage1_cbf_intervened_steps_ON/OFF": [int_steps_on, int_steps_off],
        # plan 4.2 `min d_min`: RECORDED, in all three conventions (see the decision above)
        "info_min_d_min_cbf": dmin_cbf,
        "info_min_d_min_surface": dmin_surface,
        "info_min_d_min_center": dmin_center,
        "info_cbf_extra_ON/OFF": [cbf_extra_on, m("cbf_extra").get("off_mean")],
        "info_h_min_train_ON": m("h_min_train").get("on_mean"),
        "info_arrival_steps_median_ON/OFF": speed_cost,
        "info_z_err_rmse": zn,
        "info_terminal_speed_xy": vt,
        "info_path_length_ratio": plr,
        "info_active_pillars_mean": m("active_pillars_mean")["on_mean"],
        "info_baseline_note": _bnote,
        "info_vs_A0_z_err_rmse": (None if not rel.get("z_err_rmse")
                                  else rel["z_err_rmse"]["note"] + " (delta "
                                  f"{rel['z_err_rmse']['delta']}, spread "
                                  f"{rel['z_err_rmse']['seed_spread']})"),
        "info_vs_A0_terminal_speed": (None if not rel.get("terminal_speed_xy")
                                      else rel["terminal_speed_xy"]["note"] + " (delta "
                                      f"{rel['terminal_speed_xy']['delta']}, spread "
                                      f"{rel['terminal_speed_xy']['seed_spread']})"),
        "info_terminal_z_err": m("terminal_z_err").get("on_mean"),
    }

    # ---- PLAN 4.3: stage 2 (remove the runtime filter) -------------------------------
    zero_int = (None if int_steps_on is None or not _steps
                else round(1.0 - int_steps_on / _steps, 4))
    out["gates_stage2"] = {
        "filter_dependency_gate>=0.95": None if dep is None else dep >= 0.95,
        "zero_intervention_rate_gate>=0.95": None if zero_int is None else zero_int >= 0.95,
        # The SAME gate expressed in absolute steps.  This is the correction to a mistake of
        # mine: on 2026-09-14 I replaced zero_intervention_rate with `intervened_steps <=
        # 300` - a number with no basis in the plan, chosen because A4 happened to pass it at
        # 238.  300 steps at T=1500 is a rate of 0.80, i.e. it silently RELAXED the stage-2
        # bar by 4x.  The plan's 0.95 at T=1500 means <= 0.05*T = 75 steps.
        "zero_intervention_steps_equiv<=0.05*T": (None if int_steps_on is None or not _steps
                                                  else int_steps_on <= 0.05 * _steps),
        "h_min_train_gate>=0": (m("h_min_train")["on_mean"] or 0) >= 0.0
        if m("h_min_train")["on_mean"] is not None else None,
        # [2026-09-16, second decision] 4.2's numeric clearance bar was deleted, and this
        #   row goes with it.  Relocating it from `min d_min` to `cbf_extra` did not fix the
        #   defect: `cbf_extra` measures 0.10 for A0 and 0.05 for every 口径-A rung, so the
        #   moved gate PASSED for the old conservative baseline and FAILED for every
        #   "improved" rung - it inverted the ladder.  That is the definition of 口径 A
        #   (P0.2 lowered `cbf.r_safety_margin` 0.10 -> 0.05 to remove useless
        #   intervention), not a statement about any policy.  The extension from 4.2 to 4.3
        #   is deliberate: leaving a numeric clearance gate here while deleting it there
        #   would make the document self-contradictory.  What 4.3 still requires is
        #   `h_min^train >= 0` (the filter's own boundary) plus 0 collision, which together
        #   are the real safety statement.  A numeric clearance FLOOR, if wanted, has to
        #   arrive as a training constraint -- evidence: with the filter OFF the value is
        #   still 0.0503 (A4, 5 seeds), so it is not the filter pinning it.
        "info_min_d_min_is_record_only": ("plan 4.3 numeric clearance bar deleted "
                                          "2026-09-16 (extended from 4.2); see 0.5.24"),
        "info_min_d_min_surface": dmin_surface,
        "info_cbf_extra_ON/OFF": [cbf_extra_on, m("cbf_extra").get("off_mean")],
        "zero_collision_gate(ON and OFF)": (None if m("collision_envs")["on_mean"] is None
                                            else m("collision_envs")["on_mean"] == 0
                                            and m("collision_envs")["off_mean"] == 0),
        "stall_gate<=0.10": (m("stall_frac")["on_mean"] or 0) <= 0.10,
        "arrival@0.2_OFF_gate>=0.85": None if arr_off is None else arr_off >= 0.85,
        "info_ex_mine_intervened_steps<=300(no basis in plan)": int_steps_on,
    }

    # Kept flat for backwards compatibility with anything reading out["gates"]; it is the
    # stage-1 set plus the retired monitors, which is what the old single table held.
    out["gates"] = {
        "arrival@0.2_gate>=0.85": out["gates_stage1"]["arrival@0.2_gate>=0.85"],
        "arrival_steps_median_ON/OFF": speed_cost,
        "info_filter_speed_cost_on_over_off(warn>1.10)": speed_cost,
        "zero_collision_ON_gate": (m("collision_envs")["on_mean"] == 0
                                   if m("collision_envs")["on_mean"] is not None else None),
        "collision_envs_OFF_recorded": m("collision_envs")["off_mean"],
        "zero_oob_gate": (m("oob_envs_ever")["on_mean"] == 0
                          and m("oob_envs_ever")["off_mean"] == 0),
        "h_min_train_gate>=0": out["gates_stage2"]["h_min_train_gate>=0"],
        "stall_gate<=0.10": out["gates_stage2"]["stall_gate<=0.10"],
        "info_filter_dependency_OFF_over_ON(retired)": dep,
        "info_zero_intervention_rate(horizon_ruler)": m("zero_intervention_rate").get("on_mean"),
        "info_intervened_steps_ON/OFF": [int_steps_on, int_steps_off],
        "dropped_relevant_frac_gate<0.01": out["gates_stage1"]["dropped_relevant_frac_gate<0.01"],
        "info_min_d_min_is_record_only": out["gates_stage1"]["info_min_d_min_is_record_only"],
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
    print("\ngates (plan 4.2 / 4.3, split by stage 2026-09-16 - see plan 0.6.7):")
    print("  -- plan 4.2, STAGE 1 (geometric generalisation) --")
    _warn = []
    for label, table in (("stage1", out["gates_stage1"]), ("stage2", out["gates_stage2"])):
        if label == "stage2":
            print("  -- plan 4.3, STAGE 2 (remove the runtime filter) - shown for "
                  "forward-looking information; a stage-1 batch is not expected to pass "
                  "these --")
        for k, v in table.items():
            if k.startswith("info_"):
                mark = "INFO"
            elif isinstance(v, bool):
                mark = "PASS" if v else "FAIL"
            elif v is None:
                mark = "n/a"
            else:
                mark = "VALUE"
            # [2026-09-14] The speed-cost metric is horizon-stable but not horizon-EXACT:
            # drift between 600 and 1500 steps is +0.0022 (A2L3) to +0.0092 (A1a), up to
            # 0.9%.  It is record-only, but exceeding 1.10 still deserves a warning that
            # accumulates for the margin decision.
            if "filter_speed_cost" in k and isinstance(v, float) and v > 1.10:
                _warn.append(f"`filter_speed_cost` = {v} > 1.10 (record-only; needs a "
                             "margin decision before it can gate)")
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
