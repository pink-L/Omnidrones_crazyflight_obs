"""G20 diagnostic: why does `h_min^train` disagree with `min_clearance - cbf_extra`?

Background (plan §0.5.9③ / gap G20). The two quantities SHOULD be identical:
`clearances()` returns `||p - p_o|| - r_safe`, the filter is fed `r_cbf = r_safe + cbf_extra`,
so for the same obstacle set and the same drone position at the same step,

    h_filter = min_i(||p - p_o_i|| - r_cbf_i) = min_i(clearance_i) - cbf_extra

P0.1 (A0-legacy) obeys this. P0.2 (chiA) obeys it for ON but NOT for OFF:
`min_clearance_global_min - cbf_extra = 0.0459 - 0.05 = -0.0041` while the reported
`h_min^train = 0.0000`. Since h_min is a min over a superset of what the last-episode
accumulator sees, h_min must be <= that value - so one of the two is measuring something else.

This script runs a SHORT rollout with the filter in shadow + record_diag mode and records,
every step and every env:

    h_diag    the filter's own logged h_min            (diag_log[-1, :, 2])
    h_info    h recomputed from the filter's INPUT     (info["obstacle_cbf"] = [pos | r_cbf])
    dmin_all  min_i clearances(pos)                    (the min_clearance ruler)
    ep_min    the per-env accumulator at that moment
    pos       drone position

If `h_diag != h_info` the filter's logging is wrong. If `h_info != dmin_all - cbf_extra` then
the two rulers/obstacle sets are not the same as assumed. Whichever env/step is the global
argmin of `dmin_all` tells us whether that step was ever seen by the filter at all (the
"callback/step offset" hypothesis).

Usage (from OmniDrones/scripts):
    python cbf_hmin_diag.py task=profiles/A algo=ppo headless=true wandb.mode=disabled \
        task.env.num_envs=64 +checkpoint=<ckpt.pt> +diag_steps=120 +set_seed=11 \
        +out=/tmp/navvel_g20/diag_off.npz +runtime_filter=false
"""
import json

import hydra
import torch
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.learning import ALGOS
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def _stats(name, x):
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return f"{name}=<empty>"
    return f"{name}: min={float(x.min()):+.6f} p05={float(torch.quantile(x, .05)):+.6f}"


@hydra.main(config_path=".", config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        OmegaConf.set_struct(cfg.task, False)
    except Exception:
        pass

    steps = int(cfg.get("diag_steps", 120))
    out_path = str(cfg.get("out", "/tmp/navvel_g20/diag.npz"))
    runtime_filter = bool(cfg.get("runtime_filter", False))
    _sd = cfg.get("set_seed", None)
    if _sd is not None:
        import random as _rand
        _sd = int(_sd)
        torch.manual_seed(_sd)
        _rand.seed(_sd)
    print(f"[g20] steps={steps} num_envs={cfg.task.env.num_envs} "
          f"runtime_filter={runtime_filter} set_seed={_sd}")

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    cbf_filter = None
    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec,
                                          ("agents", "observation_central")))
    at = cfg.task.get("action_transform", None)
    if at is not None and at.startswith("multidiscrete"):
        transforms.append(FromMultiDiscreteAction(nbins=int(at.split(":")[1])))
    elif at is not None and at.startswith("discrete"):
        transforms.append(FromDiscreteAction(nbins=int(at.split(":")[1])))
    elif at == "velocity":
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
        base_env.low_level_controller = controller
        vl = cfg.task.get("vel_limit", {})
        transforms.append(VelController(controller, max_vel=vl.get("max_vel", None),
                                        max_yaw_rate=vl.get("max_yaw_rate", None)))
        from omni_drones.utils.cbf import build_cbf_filter
        cbf_filter = build_cbf_filter(cfg, shadow=not runtime_filter, record_diag=True)
        if cbf_filter is None:
            raise SystemExit("[g20] build_cbf_filter returned None (cbf.mode=none?)")
        transforms.append(cbf_filter)
    else:
        raise NotImplementedError(at)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec,
        device=base_env.device)
    policy.load_state_dict(torch.load(cfg.checkpoint, map_location=cfg.sim.device))
    print(f"[g20] loaded {cfg.checkpoint}")

    extra = float(base_env.cbf_extra)
    print(f"[g20] cbf_extra={extra}  r_safe={base_env.obstacles.r_safe[0][:4].tolist()}  "
          f"radius={base_env.obstacles.radius[0][:4].tolist()}  "
          f"active={int(base_env.obstacles.active[0].sum())}/{base_env.obstacles.pos.shape[1]}")

    rec = {k: [] for k in ("h_info", "dmin_all", "ep_min", "n_diag", "done")}
    n_diag_seen = [0]

    def cb(e, *a, **kw):
        b = base_env
        pos = b.drone_state[..., :3].detach()
        obs = b.obstacles
        clr = obs.clearances(pos)                              # (N,M) inf=inactive
        dmin_all = clr.min(dim=-1).values                      # (N,)
        # --- the filter's own input, re-read from the tensordict it consumed ---
        # pos is (N,1,3) and p_oc is (N,M,3): no unsqueeze here, otherwise the two
        # broadcast into (N,N,M) and every env is compared against every other env's
        # obstacles (that bug produced a fake "env-side skew" in the first version).
        oc = b.info["obstacle_cbf"]                            # (N,1,M,4) = [x,y,z,r_cbf]
        p_oc, r_oc = oc[..., 0, :, :3], oc[..., 0, :, 3]        # (N,M,3), (N,M)
        d_oc = (pos - p_oc).norm(dim=-1)                        # (N,M)
        h_info = torch.where(r_oc > 0, d_oc - r_oc,
                             torch.full_like(d_oc, float("inf"))).min(dim=-1).values
        # the filter's own series is read ONCE after the rollout from diag_tensor()
        # (appending it per step inside the callback is not shape-safe: the accumulate
        # wrapper can hand the transform a step-batched tensordict).
        n_diag_seen[0] = len(cbf_filter.diag_log)
        rec["h_info"].append(h_info.cpu())
        rec["dmin_all"].append(dmin_all.cpu())
        rec["ep_min"].append(obs.ep_min_clearance.squeeze(-1).detach().cpu())
        rec["n_diag"].append(torch.tensor(len(cbf_filter.diag_log)))
        _dn = getattr(b, "done", None)
        rec["done"].append((torch.as_tensor(_dn).reshape(-1).cpu() if _dn is not None
                            else torch.zeros_like(dmin_all, dtype=torch.bool).cpu()))

    td = env.reset()
    cbf_filter.reset_diag()
    with set_exploration_type(ExplorationType.MODE):
        env.rollout(max_steps=steps, policy=policy, tensordict=td, auto_reset=False,
                    break_when_any_done=False, callback=cb)

    # The filter's series: _inv_call runs once per (step, env), so diag_log holds
    # T*N entries and diag_tensor() must be flattened with reshape(-1, 4) exactly as
    # eval_ckpt.py does. Compare DISTRIBUTIONS (min/p05) rather than cells: the
    # per-(step,env) ordering inside a step is not part of the contract.
    dt = cbf_filter.diag_tensor()
    if dt is None:
        raise SystemExit("[g20] filter recorded no diagnostics")
    dg = dt.reshape(-1, 4)
    ncb = len(rec["dmin_all"])
    print(f"\n[g20] callback invocations={ncb}, filter diag entries={int(dg.shape[0])} "
          f"(= {int(dg.shape[0]) // max(ncb, 1)} per callback)")
    stack = {k: torch.stack(v).float() for k, v in rec.items()}
    stack["h_all"] = dg[:, 2].float()
    stack["corr_all"] = dg[:, 0].float()
    stack["intervened_all"] = dg[:, 1].float()
    stack["h_diag"] = dg[:, 2].float().reshape(-1, 1)     # distribution-only series
    T, N = stack["dmin_all"].shape
    print(f"[g20] callback-recorded T={T} N={N}; filter entries={int(dg.shape[0])}")

    print("\n================ G20 三方差表 ================")
    print("  " + _stats("h_diag                                ", stack["h_diag"]))
    print("  " + _stats("h_info   (from filter input)          ", stack["h_info"]))
    print("  " + _stats("dmin_all (clearances)                ", stack["dmin_all"]))
    print("  " + _stats("dmin_all - cbf_extra                  ", stack["dmin_all"] - extra))
    print("  " + _stats("ep_min   (accumulator)                ", stack["ep_min"]))

    hf_all = stack["h_all"][torch.isfinite(stack["h_all"])]
    df_all = (stack["dmin_all"] - extra)[torch.isfinite(stack["dmin_all"])]
    print("\n  --- 判据：分布级一致（不看逐格顺序） ---")
    print(f"  min(h_diag) - min(dmin_all - cbf_extra) = "
          f"{float(hf_all.min()):+.6f} - {float(df_all.min()):+.6f} = "
          f"{float(hf_all.min() - df_all.min()):+.2e}")
    print(f"  p05(h_diag) - p05(dmin_all - cbf_extra) = "
          f"{float(torch.quantile(hf_all, .05)):+.6f} - {float(torch.quantile(df_all, .05)):+.6f} = "
          f"{float(torch.quantile(hf_all, .05) - torch.quantile(df_all, .05)):+.2e}")

    # env-side series vs the filter's, now that both are (T,N): does the identity hold
    # cell by cell, and if not, is the violation AT a step where the env terminated
    # (i.e. the collision is detected on the post-step position, which the filter - which
    # runs before the physics step - never gets to see)?
    cell = (stack["h_info"] - (stack["dmin_all"] - extra))
    fin = torch.isfinite(cell)
    dn = stack["done"].bool()
    viol = (cell < -1e-5)[fin] if fin.any() else torch.zeros(0, dtype=torch.bool)
    viol_cell = (cell < -1e-5) & fin
    print(f"\n  cell-level |h_info - (dmin_all - cbf_extra)| : "
          f"max={float(cell[fin].abs().max()) if fin.any() else float('nan'):.3e} "
          f"(0 => identical rulers/sets, same step)")
    print(f"  cells where the env-side value is BELOW by >1e-5 : "
          f"{int(viol_cell.sum())}/{int(fin.sum())}")
    print(f"  env done flags in the window                     : {int(dn.sum())}/{dn.numel()}")
    if viol_cell.any():
        print(f"    of those cells, at a done step: {int((viol_cell & dn).sum())}"
              f" | at the step right AFTER a done step: "
              f"{int((viol_cell & torch.roll(dn, 1, 0)).sum())}")
        t0 = int(torch.nonzero(viol_cell.any(dim=1))[0])
        print(f"    first offending step t={t0}: dmin_all={float(stack['dmin_all'][t0].min()):+.6f} "
              f"h_info={float(stack['h_info'][t0].min()):+.6f} "
              f"done={int(dn[t0].sum())}")

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({k: v for k, v in stack.items()} |
               {"meta": json.dumps({"steps": T, "num_envs": N, "cbf_extra": extra,
                                    "runtime_filter": runtime_filter, "seed": _sd,
                                    "checkpoint": str(cfg.checkpoint)})}, out_path)
    print(f"\n[g20] dumped -> {out_path}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    main()
