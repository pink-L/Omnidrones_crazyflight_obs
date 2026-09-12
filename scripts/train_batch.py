#!/usr/bin/env python
"""NavVel training batch runner: N seeds x 1 profile, bounded concurrency.

Why a runner: every P0-P5 batch is "same profile, 3-5 seeds, same 6 non-task args".
Doing that by hand invites drift (and P0.2 already showed how expensive that is), so the
non-task args live in exactly one place here.

Concurrency default is 2, NOT 3: a 1024-env / 20M-frame run grows to ~11 GiB of VRAM at
the end of training (measured), so three of them exceed 32 GiB and the third dies with
CUDA OOM at the very last step of the run (see plan 0.5.7 pitfall 5).

Usage (from OmniDrones/):
  python scripts/train_batch.py --profile profiles/A1a --group NavVel-P1-A1a \
      --project env_design_geo11_p1geom --tag a1a --seeds 11 12 13 --parallel 2

Prints one `[train_batch] <seed> exit=<rc> ckpt=<path> sha256=<hash>` line per run so the
result can be pasted into the registry/lineage.
"""
import argparse
import glob
import hashlib
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("LZ_PYTHON", "/home/hybrid/miniconda3/envs/lz_env/bin/python")
# warm start shared by every batch since P0.1 (geo8 20M) -> batches are single-variable
DEFAULT_INIT = ("/home/lz/lzspace/drones/OmniDrones/scripts/wandb/"
                "run-20260909_180042-r02a809m/files/checkpoint_19693568.pt")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_final(before, run_name):
    """Newest run dir created after `before` whose config.yaml mentions `run_name`.

    NOTE: match the bare run name, not 'run_name: <name>' - wandb's dumped config.yaml
    does not use that exact key spelling, and an exact-match version silently returned
    nothing (v1: three A1a runs reported 'wandb=?' and 'ckpt=NONE' despite finishing).
    """
    newest, newest_t = None, 0.0
    for d in glob.glob(os.path.join(HERE, "wandb", "run-*")):
        cfg = os.path.join(d, "files", "config.yaml")
        if not os.path.isfile(cfg) or os.path.getmtime(d) < before:
            continue
        try:
            with open(cfg, errors="ignore") as fh:
                if run_name not in fh.read():
                    continue
        except OSError:
            continue
        if os.path.getmtime(d) > newest_t:
            newest, newest_t = d, os.path.getmtime(d)
    return newest


def gpu_pids():
    """PIDs currently holding a GPU context (empty list if nvidia-smi is unavailable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [int(t) for t in out.split() if t.strip().isdigit()]


def clear_gpu(why):
    """Kill anything still holding the GPU before/after a run.

    WHY THIS EXISTS (2026-09-12, cost ~2 h): an Isaac process that crashes does NOT
    always exit - it can survive with a `setproctitle`-renamed cmdline (e.g.
    "NavVel-ppo/09-12_22-29"), so `pkill -f train.py` misses it, yet it keeps its GPU
    memory and the next run segfaults too. Always kill by the PID nvidia-smi reports.
    """
    pids = gpu_pids()
    if pids:
        print(f"[train_batch] {why}: killing {len(pids)} process(es) still holding the "
              f"GPU: {pids}", flush=True)
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(6)
    return gpu_pids()


def run_one(seed, a):
    run_name = f"{a.tag}-{seed}-final"
    log = os.path.join(a.logdir, f"train_{a.tag}_s{seed}.log")
    left = clear_gpu(f"pre-launch(seed {seed})")
    if left:
        print(f"[train_batch] WARNING seed={seed}: GPU still busy after cleanup: {left}")
    cmd = [
        PYTHON, "train.py", f"task={a.profile}", "algo=ppo", "headless=true",
        "wandb.mode=online", "wandb.entity=fly-hust",
        f"wandb.project={a.project}", f"wandb.group={a.group}",
        f"wandb.run_name={run_name}",
        f"seed={seed}", f"algo.entropy_coef={a.entropy_coef}",
        f"total_frames={a.total_frames}", f"save_interval={a.save_interval}",
        "+render_eval=false", f"+init_ckpt={a.init_ckpt}",
    ]
    t0 = time.time()
    print(f"[train_batch] start seed={seed} run_name={run_name} -> {log}", flush=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env)
    ckpt = ""
    d = find_final(t0 - 5, run_name)
    if d:
        p = os.path.join(d, "files", "checkpoint_final.pt")
        ckpt = p if os.path.isfile(p) else ""
    # [2026-09-12] SUCCESS IS DEFINED BY THE CHECKPOINT, NOT BY THE EXIT CODE.
    #   On this machine Isaac also segfaults during teardown of a *successful* run, which
    #   makes the process exit with -11 while the 20M-frame training finished and the
    #   checkpoint was written. Judging by 'crashreporter lines present' or 'rc != 0'
    #   wrongly marked completed runs as failures (and led to a long wild-goose chase).
    ok = bool(ckpt)
    if ok and rc != 0:
        print(f"[train_batch] note seed={seed}: rc={rc} but checkpoint exists "
              f"=> teardown crash, result OK", flush=True)
    print(f"[train_batch] {seed} exit={rc} ok={ok} elapsed={int(time.time() - t0)}s "
          f"wandb={os.path.basename(d) if d else '?'} "
          f"ckpt={ckpt or 'NONE'} sha256={sha256(ckpt) if ckpt else '-'}", flush=True)
    left = clear_gpu(f"post-run(seed {seed})")
    if left:
        print(f"[train_batch] WARNING seed={seed}: GPU not released by the child: {left}",
              flush=True)
    if not ok:
        for pat in ("out of memory", "Traceback"):
            if os.path.isfile(log):
                with open(log, errors="ignore") as fh:
                    txt = fh.read()
                if pat in txt:
                    print(f"[train_batch] !! seed={seed} log contains {pat!r}", flush=True)
    return (0 if ok else rc), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="e.g. profiles/A1a")
    ap.add_argument("--group", required=True, help="wandb group")
    ap.add_argument("--project", required=True, help="wandb project")
    ap.add_argument("--tag", required=True, help="run_name prefix, e.g. cfb-dual-chiA-a1a")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--parallel", type=int, default=2,
                    help="simultaneous runs; keep <=2 (see the module docstring)")
    ap.add_argument("--entropy-coef", default="0.05")
    ap.add_argument("--total-frames", default="20000000")
    ap.add_argument("--save-interval", default="100")
    ap.add_argument("--init-ckpt", default=DEFAULT_INIT)
    ap.add_argument("--logdir", default="/tmp/navvel_train")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)
    if not os.path.isfile(a.init_ckpt):
        raise SystemExit(f"--init-ckpt not found: {a.init_ckpt}")

    queue, running, worst = list(a.seeds), [], 0
    # forward the parent's options EXPLICITLY: passing sys.argv through would also
    # forward --seeds, which the --child parser does not accept -> argparse exits 2.
    child_common = ["--profile", a.profile, "--group", a.group, "--project", a.project,
                    "--tag", a.tag, "--parallel", str(a.parallel),
                    "--entropy-coef", a.entropy_coef, "--total-frames", a.total_frames,
                    "--save-interval", a.save_interval, "--init-ckpt", a.init_ckpt,
                    "--logdir", a.logdir]
    while queue or running:
        while queue and len(running) < a.parallel:
            seed = queue.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "--child", str(seed)] + \
                child_common
            running.append((seed, subprocess.Popen(cmd)))
        seed, proc = running.pop(0)
        worst = max(worst, proc.wait())
    print(f"[train_batch] all done (worst exit={worst})")
    return worst


if __name__ == "__main__":
    if "--child" in sys.argv:           # worker: run exactly one seed, no pool
        argv = [x for x in sys.argv[1:] if x != "--child"]
        seed = int(argv[0])
        sys.argv = [sys.argv[0]] + argv[1:]
        ap = argparse.ArgumentParser()
        ap.add_argument("--profile", required=True)
        ap.add_argument("--group", required=True)
        ap.add_argument("--project", required=True)
        ap.add_argument("--tag", required=True)
        ap.add_argument("--parallel", type=int, default=2)
        ap.add_argument("--entropy-coef", default="0.05")
        ap.add_argument("--total-frames", default="20000000")
        ap.add_argument("--save-interval", default="100")
        ap.add_argument("--init-ckpt", default=DEFAULT_INIT)
        ap.add_argument("--logdir", default="/tmp/navvel_train")
        rc, _ = run_one(seed, ap.parse_args())
        sys.exit(rc)
    sys.exit(main())
