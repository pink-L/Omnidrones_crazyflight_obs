#!/usr/bin/env python
"""NavVel acceptance-eval batch runner (plan §4.2 / §4.3).

Runs the SAME evaluation protocol for a batch of checkpoints x {filter ON, OFF} and
prints one `[eval_metrics] {...}` JSON line per run, so
`scripts/aggregate_acceptance_eval.py` can build the acceptance table.

Why a runner: plan §4.3's "filter 依赖度 = arrival_OFF / arrival_ON" needs two
evaluations that share seed / distribution / protocol; doing that by hand is
error-prone and easy to get silently wrong.

Fixed protocol (do NOT change mid-batch):
  task        : profiles/A0-legacy        self-contained geometry snapshot
  eval_points : fixed                     start [-2.8,0,0.5] -> goal [2.8,0,1.0]
  policy      : MODE (deterministic mean)
  num_envs    : 512                       VRAM: 512x1500 ~26GiB, 512x600 ~13.6GiB
  steps       : 600                       -> peak ~13.6 GiB, allows 2-way parallel
  set_seed    : 1000 + train_seed         shared by ON/OFF => identical layouts
                                          (verified via the report's `layout_fp`)

⚠ VRAM: torchrls non-stop rollout accumulates every step's tensordict before
stacking. Measured on the 5090D (32 GiB):  1024x1500 -> CUDA OOM;
512x1500 -> 26.4 GiB peak (no parallelism);  512x600 -> 13.6 GiB peak (2-way OK).

Usage:
  python scripts/acceptance_eval.py 11:on:/path/ckpt.pt 11:off:/path/ckpt.pt --parallel 2
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("LZ_PYTHON", "/home/hybrid/miniconda3/envs/lz_env/bin/python")


def parse_spec(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"bad eval spec {spec!r}; expected <train_seed>:<on|off>:<ckpt>")
    seed, mode, ckpt = parts
    if mode not in ("on", "off"):
        raise SystemExit(f"bad mode {mode!r} in {spec!r}; expected on|off")
    return int(seed), mode, ckpt


def run_one(spec, a):
    seed, mode, ckpt = parse_spec(spec)
    # the worker runs with cwd=HERE (scripts/), so a relative --checkpoint would resolve
    # against scripts/ and silently fail with FileNotFoundError deep inside torch.load.
    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        print(f"[acceptance] SKIP {spec}: checkpoint not found ({ckpt})")
        return 1, spec
    rf = "true" if mode == "on" else "false"
    tag = f"s{seed}_{mode}"
    log = os.path.join(a.outdir, f"{tag}.log")
    cmd = [
        PYTHON, "eval_ckpt.py",
        f"task={a.profile}", "algo=ppo", "headless=true", "wandb.mode=disabled",
        f"task.env.num_envs={a.num_envs}",
        "task.fixed_init=[-2.8,0.0,0.5]", "task.fixed_target=[2.8,0.0,1.0]",
        f"+checkpoint={ckpt}", f"+rollout_steps={a.steps}",
        f"+runtime_filter={rf}", "+cbf_diag=true",
        f"+model_id={a.model_id}", f"seed={seed}", f"+set_seed={1000 + seed}",
    ]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print(f"[acceptance] start {tag}  envs={a.num_envs} steps={a.steps} runtime_filter={rf}")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
    line = ""
    with open(log) as fh:
        for ln in fh:
            if ln.startswith("[eval_metrics] "):
                line = ln.rstrip()
    print(f"[acceptance] end   {tag}  exit={rc}  -> {log}")
    if line:
        print(line)
    else:
        print(f"[acceptance] WARN {tag}: no [eval_metrics] line")
    return rc, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="<train_seed>:<on|off>:<ckpt_path>")
    ap.add_argument("--parallel", type=int, default=2,
                    help="simultaneous eval processes (13.6 GiB each; use <=2 on 32 GiB)")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--outdir", default="/tmp/navvel_eval")
    ap.add_argument("--profile", default="profiles/A0-legacy")
    ap.add_argument("--model-id", default="navvel-cfb-v1.0.0-dual-p1-s11")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    # one worker process per spec: keeps VC/Isaac teardown isolated and lets the
    # 2-way VRAM budget (13.6 GiB x 2) be the only concurrency limit.
    common = ["--num-envs", str(a.num_envs), "--steps", str(a.steps),
              "--outdir", a.outdir, "--profile", a.profile, "--model-id", a.model_id]
    queue, running, worst = list(a.specs), [], 0
    while queue or running:
        while queue and len(running) < a.parallel:
            spec = queue.pop(0)
            running.append((spec, subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--single", spec] + common)))
        spec, proc = running.pop(0)
        worst = max(worst, proc.wait())
    print(f"[acceptance] all done (worst exit={worst})")
    return worst


if __name__ == "__main__":
    if "--single" in sys.argv:          # worker: run exactly one spec, no pool
        argv = [x for x in sys.argv[1:] if x != "--single"]
        wap = argparse.ArgumentParser()
        wap.add_argument("spec")
        wap.add_argument("--num-envs", type=int, default=512)
        wap.add_argument("--steps", type=int, default=600)
        wap.add_argument("--outdir", default="/tmp/navvel_eval")
        wap.add_argument("--profile", default="profiles/A0-legacy")
        wap.add_argument("--model-id", default="navvel-cfb-v1.0.0-dual-p1-s11")
        wa = wap.parse_args(argv)
        sys.exit(run_one(wa.spec, wa)[0])
    sys.exit(main())
