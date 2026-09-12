#!/usr/bin/env python
"""Freeze a COMPLETE `task.*` snapshot from a wandb run `config.yaml` into a
hydra-loadable geometry-profile file.

Motivation (NAVVEL_VERSION_AND_RETRAIN_PLAN.md §2.6-2 / §2.7-4, 缺口 G3):
  * `scripts/wandb/` is gitignored (OmniDrones/.gitignore:135 `wandb/`), so the
    delivery run's *resolved* task config exists only as an untracked on-disk file.
  * The repo default `cfg/task/NavVel.yaml` is NOT the delivery config
    (e.g. `n_pillars: 0` vs 4, `n_free_obstacles: 0` vs 12, T=1000 vs 1500).
  * A `geometry_profile` must therefore be a self-contained snapshot so that
    "回滚 = 换 profile 文件" and so that batches A0/A0'/A1a..A4 stay attributable.

The generated file deliberately does **not** rely on `cfg/base/env_base.yaml` /
`cfg/base/sim_base.yaml` (those are inlined), so later edits to the base files
cannot silently change a frozen profile.

Usage (from the OmniDrones repo root):
    python scripts/make_geometry_profile.py \
        --run-dir  scripts/wandb/run-20260909_191309-hpzc2m3r \
        --profile-id A0-legacy \
        --out      cfg/profiles/A0-legacy.yaml

Exit codes: 0 ok, 1 bad input, 2 verification failure.
"""

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "cfg" / "task" / "NavVel.yaml"


# --------------------------------------------------------------------------- helpers
def nested_set(tree, dotted_key, value):
    """Insert `value` at `dotted_key` inside `tree`, merging dicts."""
    parts = dotted_key.split(".")
    node = tree
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    if parts[-1] in node and isinstance(node[parts[-1]], dict) and isinstance(value, dict):
        node[parts[-1]].update(value)
    else:
        node[parts[-1]] = value


def unwrap_value(v):
    """wandb's config.yaml serialises every entry as `key: {value: <literal>}`.
    Peel that single-level wrapper (and only that one)."""
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return v["value"]
    return v


def load_task_snapshot(run_dir: Path):
    """Read <run_dir>/files/config.yaml -> (nested task dict, raw overrides list|None)."""
    cfg_path = run_dir / "files" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"[profile] config.yaml not found under {run_dir}/files/ or {run_dir}/")
    raw = yaml.safe_load(cfg_path.read_text())

    task = {}
    n_keys = 0
    for k, v in raw.items():
        if k.startswith("task."):
            nested_set(task, k[len("task."):], unwrap_value(v))
            n_keys += 1

    # the full CLI override list wandb recorded (authoritative provenance)
    overrides = None
    try:
        w = raw["_wandb"]["value"]
        for v in w.values():
            if isinstance(v, dict) and "args" in v:
                overrides = list(v["args"])
                break
    except Exception:
        pass
    return task, overrides, raw


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def diff_vs_baseline(task, baseline_path: Path = DEFAULT_BASELINE):
    """Return (only_in_profile, differing, only_in_baseline) dotted-key dicts."""
    if not baseline_path.exists():
        return {}, {}, {}
    base = yaml.safe_load(baseline_path.read_text()) or {}
    # the generated profile is fully expanded (base files inlined), so expand the
    # baseline the same way for a like-for-like comparison
    for extra in ("env_base.yaml", "sim_base.yaml"):
        p = baseline_path.parent.parent / "base" / extra
        if p.exists():
            base = {**(yaml.safe_load(p.read_text()) or {}), **base}
    fp, fb = _flatten(task), _flatten(base)
    only_p = {k: v for k, v in fp.items() if k not in fb}
    only_b = {k: v for k, v in fb.items() if k not in fp}
    diff = {k: (fb[k], v) for k, v in fp.items() if k in fb and fb[k] != v}
    return only_p, diff, only_b


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="wandb run dir (with files/config.yaml)")
    ap.add_argument("--profile-id", required=True, help="e.g. A0-legacy")
    ap.add_argument("--out", default=None, help="default cfg/profiles/<profile-id>.yaml")
    ap.add_argument("--notes", default="", help="one-line note stored in the header")
    ap.add_argument("--header-extra", default="", help="extra header lines (one string)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else REPO_ROOT / "cfg" / "profiles" / f"{args.profile_id}.yaml"
    task, overrides, raw = load_task_snapshot(run_dir)

    if not task:
        raise SystemExit(f"[profile] no `task.*` keys found in {run_dir}")
    if "name" not in task:
        raise SystemExit("[profile] snapshot has no task.name -> refusing to write")

    body = yaml.safe_dump(task, sort_keys=True, default_flow_style=False,
                          allow_unicode=True, width=1000).rstrip()

    stamp = _dt.datetime.now().strftime("%Y-%m-%d")
    sha_note = ""
    try:
        import hashlib
        ck = next((run_dir / "files").glob("checkpoint_final.pt"), None)
        if ck and ck.exists():
            sha_note = ("\n#   checkpoint sha256 : "
                        + hashlib.sha256(ck.read_bytes()).hexdigest())
    except Exception:
        pass

    header = [
        "# =====================================================================",
        f"# geometry_profile : {args.profile_id}",
        "# type             : COMPLETE frozen `task` snapshot (self-contained)",
        f"# generated        : {stamp} by scripts/make_geometry_profile.py",
        f"# source run       : {run_dir}",
        "# source config    : <run>/files/config.yaml   (gitignored: wandb/)",
        "# NOTE: cfg/base/{env,sim}_base.yaml are INLINED on purpose -> edits to the",
        "#       base files can never silently change this frozen profile.",
        "# NOTE: no defaults list, no CLI override required.  Use as",
        "#       `python scripts/train.py task=<group path of this file> ...`",
    ]
    if args.notes:
        header.append(f"# notes            : {args.notes}")
    if args.header_extra:
        header.extend(f"# {ln}" for ln in args.header_extra.splitlines())
    if overrides:
        header.append("# ---- CLI overrides recorded by wandb (historical; NOT needed to rerun) ----")
        header.extend(f"#   {a}" for a in overrides)
    header.append("# =====================================================================")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + body + "\n")

    # verification: re-read (comments stripped) and compare the flattened keys
    stripped = "\n".join(l for l in out.read_text().splitlines()
                         if not l.lstrip().startswith("#"))
    back = yaml.safe_load(stripped)
    if _flatten(back) != _flatten(task):
        raise SystemExit("[profile] VERIFY FAILED: round-trip mismatch")
    print(f"[profile] wrote {out}  ({len(_flatten(task))} leaf keys, round-trip OK)")

    only_p, diff, only_b = diff_vs_baseline(task)
    print(f"[profile] vs repo default cfg/task/NavVel.yaml (缺口 G3):")
    print(f"          only in profile : {len(only_p)} keys"
          + (f" -> {json.dumps({k: only_p[k] for k in sorted(only_p)[:12]}, default=str)}"
             if only_p else ""))
    if diff:
        print(f"          DIFFERENT       : {len(diff)} keys")
        for k in sorted(diff)[:40]:
            print(f"            {k}: repo={diff[k][0]!r}  profile={diff[k][1]!r}")
        if len(diff) > 40:
            print(f"            ... (+{len(diff) - 40} more)")
    print(f"          only in repo    : {len(only_b)} keys"
          + (f" -> {sorted(only_b)[:12]}" if only_b else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
