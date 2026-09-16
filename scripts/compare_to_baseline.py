#!/usr/bin/env python
"""Judge plan 4.2's "not worse than A0" rows by comparing two `agg.json` files.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
plan 4.2 has two rows whose bar is *relative* rather than absolute:

    | `z_err RMSE` / 终端速度 | 沿用 NAVVEL_RETRAIN_GUIDE.md 第二部分 §4 | 不劣于 A0 | 全部 |

Nothing in the repository compared a rung against A0, so "not worse than A0" was
untestable and A0's own values had never been measured at all.  Doing it in a shell
one-liner guarantees the numbers in the plan get transcribed by hand, which is exactly
how a comparison conclusion got flipped earlier in this project (a remembered
`arrival@0.2 = 0.8194` versus the real `0.7975`).

TWO DISCIPLINES ARE BUILT IN
----------------------------
1. Every metric's direction is declared, not assumed.  "Not worse" means a different
   inequality for `z_err_rmse` (lower) than for `arrival@0.2` (higher), and a table that
   hides that is a table that can be read backwards.
2. A mean-only verdict is refused.  A4 vs A2L3 looked like a clean ordering at 3 seeds and
   dissolved at 5, so a delta that is smaller than the seed spread of EITHER side is
   reported as "within seed spread", never as a pass on merit.

Usage:
  python scripts/compare_to_baseline.py \
      --a0  /tmp/navvel_p1/a0_384x1500/agg.json \
      --rung /tmp/navvel_p1/a4_384x1500/agg.json \
      --label A4 --out /tmp/navvel_p1/a4_vs_a0.json
"""

import argparse
import json
import sys

# key -> (direction, gate?, note)
#   direction: "lower" or "higher" = which way is better
#   gate: True -> plan 4.2 makes this a real pass/fail row against A0
#         False -> plan 4.2 says "record"; still printed so a trend cannot hide
METRICS = {
    "z_err_rmse": ("lower", True, "plan 4.2: 不劣于 A0"),
    "terminal_speed_xy": ("lower", True, "plan 4.2: 不劣于 A0"),
    "terminal_z_err": ("lower", False, "diagnostic added 2026-09-16 (end-point height)"),
    "path_length_ratio": ("lower", False, "plan 4.2: 记录（判绕远）"),
    "arrival@0.2": ("higher", False, "already an absolute gate (>=0.85); shown for context"),
    "arrival_steps_median": ("lower", False, "horizon-stable speed metric; ordering only"),
    "stall_frac": ("lower", False, "absolute gate elsewhere (<=0.10)"),
    "corr_p50": ("lower", False, "plan 4.3 records the trend, p50/p95"),
    "corr_p95": ("lower", False, "plan 4.3 records the trend, p50/p95"),
    "min_clearance_global_min": ("higher", False,
                                 "cbf/raw convention; RECORD only since 2026-09-16"),
    "h_min_train": ("higher", False, "stage-2 row (>=0); recorded here"),
}


def load(path):
    with open(path) as fh:
        j = json.load(fh)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True, help="agg.json of the A0 (v1.0.0) baseline")
    ap.add_argument("--rung", required=True, help="agg.json of the rung under test")
    ap.add_argument("--label", default="rung")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    J0, JR = load(a.a0), load(a.rung)

    # ---- protocol guard: relative comparisons across protocols are meaningless --------
    p0, pr = J0.get("protocol", {}), JR.get("protocol", {})
    for k in ("num_envs", "rollout_steps", "task", "eval_points"):
        if p0.get(k) != pr.get(k):
            print(f"[compare] refuse: protocol mismatch on {k}: "
                  f"A0={p0.get(k)!r} vs {a.label}={pr.get(k)!r}")
            print("[compare] 2026-09-14 lesson: metrics of these two rows move up to 55x "
                  "between 600 and 1500 steps, so a cross-protocol 'not worse' verdict is "
                  "not a measurement.")
            return 2

    S0, SR = J0.get("summary", {}), JR.get("summary", {})
    rows, verdicts = [], {}
    for key, (direction, is_gate, note) in METRICS.items():
        m0, mr = S0.get(key, {}), SR.get(key, {})
        on0, onr = m0.get("on_mean"), mr.get("on_mean")
        r0, rr = m0.get("on_range"), mr.get("on_range")
        if on0 is None or onr is None:
            rows.append((key, on0, onr, None, None, None, "n/a", note))
            verdicts[key] = None
            continue
        delta = round(onr - on0, 4)
        better = (delta < 0) if direction == "lower" else (delta > 0)
        # "not worse" = NOT worse = equal or better
        not_worse = (delta <= 0) if direction == "lower" else (delta >= 0)
        # spread guard: if the gap is inside the seed spread of either side, the means do
        # not establish an ordering at all
        spread = max(x for x in (r0 or 0.0, rr or 0.0))
        if abs(delta) <= spread and delta != 0:
            tag = "within seed spread"
            if is_gate:
                not_worse = True   # cannot be called worse on this evidence
        elif delta == 0:
            tag = "equal"
        else:
            tag = "worse" if not not_worse else "better"
        mark = ("PASS" if not_worse else "FAIL") if is_gate else "-"
        rows.append((key, on0, onr, delta, spread, None, f"{mark} {tag}", note))
        verdicts[key] = {"gate": is_gate, "a0_on": on0, f"{a.label}_on": onr,
                         "delta": delta, "seed_spread": spread,
                         "not_worse": not_worse if is_gate else None,
                         "verdict": f"{mark} {tag}"}

    print(f"\n===== {a.label} vs A0 (plan 4.2 relative rows) =====")
    print(f"protocol : {pr.get('num_envs')} envs x {pr.get('rollout_steps')} steps, "
          f"task={pr.get('task')}, eval_points={pr.get('eval_points')}")
    print(f"seeds    : A0 {J0.get('protocol', {}).get('seeds')}   "
          f"{a.label} {pr.get('seeds')}")
    print(f"\n| metric | A0 ON | {a.label} ON | delta | seed spread | verdict |")
    print("|---|---|---|---|---|---|")
    for key, on0, onr, delta, spread, _, verdict, _note in rows:
        f = lambda v: "-" if v is None else f"{v}"
        print(f"| `{key}` | {f(on0)} | {f(onr)} | {f(delta)} | {f(spread)} | {verdict} |")
    print("\nnotes:")
    for key, _a, _b, _d, _s, _, _v, note in rows:
        print(f"  {key:26s} {note}")

    gates = {k: v for k, v in verdicts.items()
             if v and v["gate"]}
    bad = [k for k, v in gates.items() if not v["not_worse"]]
    print(f"\nrelative gates: {len(gates) - len(bad)}/{len(gates)} pass"
          + (f"   FAIL: {bad}" if bad else ""))
    for k, v in gates.items():
        print(f"  [{'PASS' if v['not_worse'] else 'FAIL'}] {k}: {v['a0_on']} -> "
              f"{v[f'{a.label}_on']} (delta {v['delta']}, spread {v['seed_spread']})")

    out = {"label": a.label, "protocol": pr, "a0_protocol": p0,
           "rows": verdicts, "relative_gate_failures": bad}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\n[compare] wrote {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
