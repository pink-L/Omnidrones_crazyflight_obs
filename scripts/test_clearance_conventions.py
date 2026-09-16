#!/usr/bin/env python
"""Assertion test: the THREE clearance conventions must stay distinguishable.

WHY THIS TEST EXISTS
--------------------
`ObstacleManager.clearances()` computes

    |p - p_oi| - (r_oi + drone_radius + inflation)

The env header (`nav_vel_obstacles.py` L27-28) states that `drone_radius` is the drone's
physical sphere and `inflation` is an *extra* cushion, so the raw number is
drone-surface-to-obstacle-surface MINUS the cushion.  Three documents then reuse the name
`d_min` for three different quantities:

    cbf     = raw                              what the CBF acts on / the collision test
    surface = raw + inflation                  drone body surface <-> obstacle surface
    centre  = raw + inflation + drone_radius   drone centre     <-> obstacle surface

Plan 4.2 gates `min d_min >= 0.10 m` and calls it "surface clearance", but never fixes
the origin.  On profile A (inflation 0.02, drone_radius 0.10) the ladder's observed
minimum is 0.0501, so the SAME rollout reads

    cbf 0.0501 -> FAIL      surface 0.0701 -> FAIL      centre 0.1701 -> PASS

A units mismatch like this is invisible in a table of numbers: it looks like a result,
not a bug.  (I already made exactly that mistake once - I added drone_radius+inflation
back, declared PASS, and only afterwards checked which reading that corresponded to.)
This test pins the arithmetic so a future edit cannot quietly collapse the three rows
into one, and so "the convention is still undecided" stays visible instead of being
resolved by whichever reading lets the rung pass.

Runs standalone (`python test_clearance_conventions.py`) or under pytest.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "aggregate_acceptance_eval.py")
RAWDIR = "/tmp/navvel_p1/acc_a4_384x1500"          # the real A4 logs (integration check)

# ---------------------------------------------------------------------------------
# Synthetic `[eval_metrics]` records.
#
# Numbers are taken from the real A4 run so the thresholds behave as they did in
# production: raw min clearance 0.0501, which is what made the ambiguity decisive.
# Every key that the aggregator indexes with `[...]` (not `.get`) must be present, or
# the test would fail on a KeyError instead of on the property under test.
# ---------------------------------------------------------------------------------
RAW = 0.0501
PAD_INFLATION = 0.02
PAD_CENTER = 0.12


def _rec(seed, mode, raw=RAW, with_new_fields=False, surface=None, center=None):
    rec = {
        "task": "NavVel",
        "eval_points": 1,
        "num_envs": 384,
        "rollout_steps": 1500,
        "seed": seed,
        "runtime_filter": (mode == "on"),
        "arrival_at": {"0.5": 1.0, "0.3": 0.999, "0.2": 0.9974},
        "collision_envs": 0,
        "collision_edges": 0,
        "oob_envs_ever": 0,
        "crash_envs_ever": 0,
        "joint_success_envs": 383,
        "min_clearance_global_min": raw,
        "min_clearance_env_mean": raw + 0.01,
        "h_min_train": 0.0001,
        "h_below0_frac": 0.0,
        "zero_intervention_rate": 0.8413,
        "intervened_step_frac": 0.1588,
        "corr_mean": 0.01,
        "corr_p50": 0.0,
        "corr_p95": 0.1,
        "stall_frac": 0.01,
        "dropped_relevant_frac": 0.0,
        "dropped_relevant_step_frac": 0.0,
        "relevant_obstacles_total": 100,
        "arrival_steps_median": 418.0 if mode == "on" else 390.0,
        "arrival_steps_mean": 420.0,
        "arrival_steps_p90": 500.0,
        "active_pillars_min": 2,
        "active_pillars_mean": 3.85,
        "active_pillars_max": 8,
        "active_pillars_distinct": 7,
        "layout_fp": "1.0/2.0/48/0.100",
    }
    rec["cbf_extra"] = 0.05
    if with_new_fields:
        rec["clearance_pad_inflation"] = PAD_INFLATION
        rec["clearance_pad_center"] = PAD_CENTER
        rec["min_clearance_surface_global_min"] = surface
        rec["min_clearance_center_global_min"] = center
        rec["z_err_rmse"] = 0.05
        rec["terminal_speed_xy"] = 0.03
        rec["terminal_z_err"] = 0.04
        rec["path_length_ratio"] = 1.05
    return rec


def _write_logdir(recs):
    d = tempfile.mkdtemp(prefix="navvel_clrtest_")
    for r in recs:
        tag = f"s{r['seed']}_{'on' if r['runtime_filter'] else 'off'}"
        with open(os.path.join(d, tag + ".log"), "w") as fh:
            fh.write("noise before\n")
            fh.write("[eval_metrics] " + json.dumps(r, sort_keys=True) + "\n")
    return d


def _run(logdir, *extra):
    out = os.path.join(logdir, "agg.json")
    cmd = [sys.executable, AGG, "--logdir", logdir, "--out", out, "--label", "test"]
    cmd += list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"aggregator rc={p.returncode}\n{p.stdout}\n{p.stderr}")
    with open(out) as fh:
        return json.load(fh)


def _dmin(j, which):
    """The RECORDED clearance, not a gate.

    After the 2026-09-16 decision the clearance bar no longer lives on `min d_min` at all
    (it moved to `cbf_extra`), so these three rows are informational by design.  Reading
    them through a helper keeps the change visible: if a future edit turns any of them
    back into a PASS/FAIL row, `test_no_clearance_gate_exists` fails.
    """
    return j["gates_stage1"][f"info_min_d_min_{which}"]


def _cbf_extra(j):
    """`cbf_extra` is RECORDED too, as of the second 2026-09-16 decision.

    Relocating the numeric bar from `min d_min` onto `cbf_extra` did not fix the defect:
    measured, `cbf_extra` is 0.10 for A0 and 0.05 for every 口径-A rung, so the moved gate
    passed for the old baseline and failed for every "improved" rung.  A gate that inverts
    the ladder because of a profile definition is not a gate.  So no clearance-shaped row
    gates anything in either stage, and `test_no_clearance_gate_exists` checks all of them.
    """
    return j["gates_stage1"]["info_cbf_extra_ON/OFF"][0]


# ---------------------------------------------------------------------------------
# 1. The three conventions must be distinguishable for the SAME rollout.
# ---------------------------------------------------------------------------------
def test_three_conventions_differ_on_one_rollout():
    d = _write_logdir([_rec(11, "on", with_new_fields=True,
                            surface=round(RAW + PAD_INFLATION, 4),
                            center=round(RAW + PAD_CENTER, 4)),
                       _rec(11, "off")])
    try:
        j = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    cc = j["clearance_conventions"]
    assert cc["min_cbf"] == RAW, cc
    assert cc["min_surface(=cbf+pad_inflation)"] == round(RAW + PAD_INFLATION, 4), cc
    assert cc["min_center(=cbf+pad_center)"] == round(RAW + PAD_CENTER, 4), cc
    # the decisive part: one raw number, three different readings, and 0.10 straddles them
    assert _dmin(j, "cbf") < 0.10 < _dmin(j, "center"), j["gates_stage1"]
    # and the aggregator must say out loud that the choice is not its to make
    assert "UNRESOLVED" in cc and cc["UNRESOLVED"], cc


# ---------------------------------------------------------------------------------
# 2. Legacy logdirs (pre-2026-09-16) must still be re-scorable, but only from pads
#    that were passed explicitly on the command line.
# ---------------------------------------------------------------------------------
def test_legacy_logs_derive_from_explicit_pads_only():
    d = _write_logdir([_rec(11, "on"), _rec(11, "off")])
    try:
        j = _run(d, "--d-min-inflation-pad", str(PAD_INFLATION),
                 "--d-min-pad", str(PAD_CENTER))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert j["clearance_conventions"]["min_surface(=cbf+pad_inflation)"] == round(
        RAW + PAD_INFLATION, 4)
    assert j["clearance_conventions"]["min_center(=cbf+pad_center)"] == round(
        RAW + PAD_CENTER, 4)


def test_no_pads_means_no_guess():
    """Anti-regression for the mistake I actually made.

    Without pads, a legacy logdir cannot be converted, so the two derived gates MUST be
    n/a.  If this test fails because someone hardcoded 0.12 somewhere, that is the bug it
    is meant to catch: a silent conversion that makes an undecided convention look
    decided, and in the PASS direction.
    """
    d = _write_logdir([_rec(11, "on"), _rec(11, "off")])
    try:
        j = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert j["clearance_conventions"]["min_surface(=cbf+pad_inflation)"] is None
    assert j["clearance_conventions"]["min_center(=cbf+pad_center)"] is None
    assert _dmin(j, "surface") is None, j["gates_stage1"]
    assert _dmin(j, "center") is None, j["gates_stage1"]
    assert _dmin(j, "cbf") == RAW, j["gates_stage1"]
    assert _cbf_extra(j) == 0.05, j["gates_stage1"]


# ---------------------------------------------------------------------------------
# 3. No double counting: when the log already carries the converted value, the pads must
#    not be added on top of it again.
# ---------------------------------------------------------------------------------
def test_no_double_counting_when_field_present():
    surf = round(RAW + PAD_INFLATION, 4)
    cen = round(RAW + PAD_CENTER, 4)
    d = _write_logdir([_rec(11, "on", with_new_fields=True, surface=surf, center=cen),
                       _rec(11, "off")])
    try:
        # pads deliberately supplied AND fields present -> fields win
        j = _run(d, "--d-min-inflation-pad", str(PAD_INFLATION),
                 "--d-min-pad", str(PAD_CENTER))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    cc = j["clearance_conventions"]
    assert cc["min_surface(=cbf+pad_inflation)"] == surf, cc
    assert cc["min_center(=cbf+pad_center)"] == cen, cc
    # a second addition would land exactly on these values, so check we did NOT
    assert cc["min_surface(=cbf+pad_inflation)"] != round(surf + PAD_INFLATION, 4), cc
    assert cc["min_center(=cbf+pad_center)"] != round(cen + PAD_CENTER, 4), cc


# ---------------------------------------------------------------------------------
# 4. Integration check against the real A4 logs, if they are still on disk.  These are
#    pre-2026-09-16 logs, so they exercise the pad path on production data.
# ---------------------------------------------------------------------------------
def test_real_a4_logs_reproduce_the_known_ambiguity():
    if not os.path.isdir(RAWDIR):
        print(f"  [skip] {RAWDIR} not present")
        return
    j = _run(RAWDIR, "--d-min-inflation-pad", "0.02", "--d-min-pad", "0.12")
    cc = j["clearance_conventions"]
    assert abs(cc["min_cbf"] - 0.0501) < 0.005, cc
    assert _dmin(j, "cbf") < 0.10 < _dmin(j, "center"), j["gates_stage1"]
    assert _cbf_extra(j) == 0.05, j["gates_stage1"]
    print(f"  A4 384x1500 (recorded, no clearance gate): cbf={cc['min_cbf']} surface="
          f"{cc['min_surface(=cbf+pad_inflation)']} center="
          f"{cc['min_center(=cbf+pad_center)']} -> 0.10 straddles them")



# ---------------------------------------------------------------------------------
# 5. NOTHING clearance-shaped may gate, in EITHER stage.
#
#    This is the anti-regression for the decision, and it is deliberately stronger than the
#    one it replaces.  The first attempt at this fix moved the numeric bar from `min d_min`
#    onto `cbf_extra`; that looked like a repair and was not one, because `cbf_extra` is a
#    profile constant (0.10 for A0, 0.05 for every 口径-A rung), so the moved gate passed
#    for the old baseline and failed for every improved rung.  A test that only checked "the
#    gate reads cbf_extra now" would have PASSED on that wrong fix.  So the property under
#    test is the outcome, not the mechanism: no PASS/FAIL row anywhere may be a function of
#    a clearance or of `cbf_extra`.
# ---------------------------------------------------------------------------------
def test_no_clearance_like_row_gates_in_either_stage():
    d = _write_logdir([_rec(11, "on", with_new_fields=True,
                            surface=round(RAW + PAD_INFLATION, 4),
                            center=round(RAW + PAD_CENTER, 4)),
                       _rec(11, "off")])
    try:
        j = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for stage in ("gates_stage1", "gates_stage2"):
        bad = [k for k, v in j[stage].items()
               if not k.startswith("info_") and isinstance(v, bool)
               and ("d_min" in k or "clearance" in k or "cbf_extra" in k)]
        assert not bad, f"{stage} still gates on a clearance-like row: {bad}"
    # and both stages must still SAY that the bar was removed, rather than dropping the
    # topic - a silently deleted row is indistinguishable from a passing one
    for stage in ("gates_stage1", "gates_stage2"):
        assert "info_min_d_min_is_record_only" in j[stage], stage


def test_cbf_extra_is_recorded_not_gated():
    """`cbf_extra` must be visible (it is what the numeric bar used to sit on) but inert."""
    for extra in (0.05, 0.10, 0.12):
        r_on = _rec(11, "on", with_new_fields=True,
                    surface=round(RAW + PAD_INFLATION, 4),
                    center=round(RAW + PAD_CENTER, 4))
        r_on["cbf_extra"] = extra
        d = _write_logdir([r_on, _rec(11, "off")])
        try:
            j = _run(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        assert _cbf_extra(j) == extra, j["gates_stage1"]
        assert not [k for k, v in j["gates_stage1"].items()
                    if not k.startswith("info_") and isinstance(v, bool)
                    and "cbf_extra" in k], j["gates_stage1"]


def test_no_clearance_gate_exists():
    """`min d_min` must be RECORDED in all three conventions, not gated."""
    d = _write_logdir([_rec(11, "on", with_new_fields=True,
                            surface=round(RAW + PAD_INFLATION, 4),
                            center=round(RAW + PAD_CENTER, 4)),
                       _rec(11, "off")])
    try:
        j = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    gated = [k for k in j["gates_stage1"] if "d_min" in k and "info_" not in k]
    assert not gated, f"a clearance row is still gating: {gated}"
    for which in ("cbf", "surface", "center"):
        assert f"info_min_d_min_{which}" in j["gates_stage1"]
    # the cbf convention must always be present, because it is the one the filter and the
    # collision test actually use
    assert j["gates_stage1"]["info_min_d_min_cbf"] == RAW



def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
