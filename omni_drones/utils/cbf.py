# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
[M2-3, 2026-09-05] First-order CBF (control-barrier-function) safety layer for
NavVel's *velocity-command* action layer (see m2_plan.md §5 + navvel_rl_interface.md §2/§6).

Math (static obstacles, v_o = 0; speed commands -> low-level Lee has lag, so the
CBF decision radius carries an extra braking allowance on top of the geometric one):

    h_i(p)   = ||p - p_oi|| - r_si^cbf          (signed distance to the CBF ball)
    n_i      = (p - p_oi) / ||p - p_oi||
    dot h_i  = n_i^T v
    CBF constraint:   dot h_i + alpha h_i >= 0  <=>  n_i^T v >= -alpha h_i

    r_si^cbf = r_si + margin + (v_max^2 / (2 a_max))     [r_si = r_drone + r_oi + inflation]
                                                         (braking / inner-loop allowance)

* ``filter_velocity``  projects a nominal speed command into the intersection of the
  active half-spaces (iterative closed-form: each pass fixes the most-violated one),
  which is exactly CBF-RL's discrete closed-form filter for many obstacles.
* ``cbf_violation``     returns the total *pre-filter* constraint violation used by the
  reward core (soft-CBF / reward_only / hybrid): sum_i min(0, n_i^T v_nom + alpha h_i)
  (+ optional extra when h_i < 0, i.e. already inside the CBF ball).

All geometry is pure torch over env-frame positions, unit-testable on CPU (no isaac).
Ground-truth obstacle inputs (pos / r_safe / active) come from the env's ObstacleManager
via the tensordict ``("info", "obstacle_cbf")`` channel, so CBF/obs/collision/reward all
share one source of truth. The CBF radius is deliberately NOT fed to the policy obs.
"""

import torch

try:  # torchrl availability guards import when only pure functions are needed
    from torchrl.envs.transforms import Transform
    _TORCHRL = True
except Exception:  # pragma: no cover - pure-torch tests can run without torchrl
    Transform = object
    _TORCHRL = False


# --------------------------------------------------------------------------- radius
def safety_radius_extra(drone_radius, inflation, margin, v_max, a_max,
                        use_brake_term=True):
    """Clearance added ON TOP of the geometric decision radius r_s = r_drone+r_o+infl.

    Returns ``margin + v_max**2/(2 a_max)`` (m) when braking is enabled, else ``margin``.
    """
    extra = float(margin)
    if use_brake_term and a_max is not None and float(a_max) > 0:
        extra += float(v_max) ** 2 / (2.0 * float(a_max))
    return float(extra)


def cbf_safety_radius(r_o, drone_radius, inflation, margin, v_max, a_max,
                      use_brake_term=True):
    """Per-obstacle CBF ball radius (m): geometric r_s + safety/braking extra."""
    r_s = float(drone_radius) + float(r_o) + float(inflation)
    return r_s + safety_radius_extra(drone_radius, inflation, margin, v_max, a_max,
                                     use_brake_term)


# --------------------------------------------------------------- pure torch kernels
def _gradients(pos, p_obs):
    """Distance, outward unit normal and inputs for one obstacle set.

    pos (...,3), p_obs (...,K,3) -> dvec (...,K,3), dist (...,K), n (...,K,3).
    ``n`` is the OUTWARD normal (obstacle center -> drone), so flying straight at an
    obstacle gives n.v < 0 and the CBF constraint reads  n.v + alpha*h >= 0.
    """
    dvec = pos.unsqueeze(-2) - p_obs                    # center -> drone (outward)
    dist = torch.norm(dvec, dim=-1)                     # (...,K)
    n = dvec / (dist.unsqueeze(-1) + 1e-6)              # (...,K,3) unit outward normal
    return dvec, dist, n


def filter_velocity(pos, v_nom, p_obs, r_safe, active, alpha, iterations=3):
    """Iterative closed-form projection of ``v_nom`` into the CBF-safe half-spaces.

    Args:
        pos (...,3): drone env-frame position.
        v_nom (...,3): nominal (policy) velocity command.
        p_obs (...,K,3): obstacle centers (env frame). Inactive slots may be anything
                         (they are masked out by ``active``).
        r_safe (...,K): CBF ball radius per slot (inactive -> <=0).
        active (...,K): bool mask (same source as obs/geometry: radius>0).
        alpha: CBF convergence rate.
        iterations: number of "fix the most-violated constraint" passes.

    Returns:
        v_safe (...,3): projected command (no worse than the current one).
        fix_norm (...,): total projected-away speed magnitude (stats/diagnostic), >=0.
    """
    v = v_nom.clone()
    fix_norm = torch.zeros(pos.shape[:-1], dtype=v.dtype, device=v.device)
    for _ in range(int(iterations)):
        _, dist, n = _gradients(pos, p_obs)
        h = dist - r_safe                                        # (...,K)
        g = (n * v.unsqueeze(-2)).sum(-1) + alpha * h            # n.v + alpha h
        # violation only on active slots (inactive -> +inf so never selected)
        g_act = torch.where(active, g, torch.full_like(g, float("inf")))
        delta = torch.clamp(-g_act, min=0.0)                     # (...,K) >=0
        viol = delta.max(dim=-1).values                          # (...,)
        need = (viol > 0).any()
        if not need:
            break
        # push back along the outward normal of the MOST-violated constraint only:
        #   v *= v - (-g) n = v + (n.v + alpha h) n  (closed form, one obstacle)
        best = (delta == viol.unsqueeze(-1)) & active            # one-hot (...,K)
        n_sel = (n * best.unsqueeze(-1)).sum(dim=-2)             # (...,3)
        v = v + viol.unsqueeze(-1) * n_sel
        fix_norm = fix_norm + viol
    return v, fix_norm


def cbf_violation(pos, v_nom, p_obs, r_safe, active, alpha, penalty_intrude=True):
    """Total pre-filter CBF violation for the reward core (<=0).

    sum_i min(0, n_i^T v_nom + alpha h_i)  over active slots; optionally adds an extra
    min(0, h_i) term per slot that is already inside its CBF ball (h_i < 0), so "being
    inside" is penalized regardless of the current command direction.

    Returns (...,) <= 0 (more negative = more unsafe).
    """
    _, dist, n = _gradients(pos, p_obs)
    h = dist - r_safe                                            # (...,K)
    g = (n * v_nom.unsqueeze(-2)).sum(-1) + alpha * h            # n.v_nom + alpha h
    viol = torch.clamp(g, max=0.0)                               # (...,K) <=0
    viol = torch.where(active, viol, torch.zeros_like(viol))
    total = viol.sum(dim=-1)
    if penalty_intrude:
        intr = torch.clamp(h, max=0.0)
        intr = torch.where(active, intr, torch.zeros_like(intr))
        total = total + intr.sum(dim=-1)
    return total


# --------------------------------------------------------------- torchrl Transform
class CBFVelocityFilter(Transform):
    """Velocity-domain CBF filter placed BEFORE ``VelController`` in the action chain.

    Reads the policy's raw 4-dim command (3-vel + yaw), projects the linear part onto
    the CBF half-spaces and writes the result back to ``action_key`` (yaw untouched).
    It also records the PRE-filter command into ``("info", "policy_action")`` so the env
    reward core (reward_only / hybrid) can penalize exactly what the policy produced
    (CBF-RL soft-CBF). Requires env-frame geometry per env per slot in
    ``("info", "obstacle_cbf")`` of shape (..., K, 4) = [p_oi(3), r_si^cbf(1)],
    inactive slots exactly zero.

    Placement note: torchrl ``Compose`` runs inv (action) transforms in *reverse* add
    order, so this transform must be appended AFTER ``VelController`` in the chain list.
    """

    def __init__(
        self,
        action_key=("agents", "action"),
        alpha=1.0,
        iterations=3,
        filter_grad="detach",        # detach | through (through keeps graph; unused in env)
        do_filter=True,              # False -> record v_nom only (reward_only mode)
    ):
        if not _TORCHRL:
            raise RuntimeError("CBFVelocityFilter requires torchrl")
        super().__init__(in_keys=[], in_keys_inv=[("info", "drone_state"),
                                                  ("info", "obstacle_cbf")])
        self.action_key = action_key
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.filter_grad = filter_grad
        self.do_filter = bool(do_filter)

    def _inv_call(self, tensordict):
        drone_state = tensordict[("info", "drone_state")]      # (...,13) pos is [:3]
        obs_cbf = tensordict[("info", "obstacle_cbf")]         # (...,K,4)
        action = tensordict[self.action_key]                   # (...,4) raw speed cmd

        pos = drone_state[..., :3]
        p_obs = obs_cbf[..., :3]                               # (...,K,3)
        r_cbf = obs_cbf[..., 3]                                # (...,K)
        active = r_cbf > 0                                     # radius>0 source of truth

        if self.filter_grad == "through":
            v_nom = action[..., :3]
        else:
            v_nom = action[..., :3].detach()
            p_obs = p_obs.detach()
            r_cbf = r_cbf.detach()

        # record the pre-filter command for the reward core (full 4-dim, incl. yaw)
        tensordict.set(("info", "policy_action"), action.clone())

        if self.do_filter and active.any():
            v_safe, _ = filter_velocity(pos, v_nom, p_obs, r_cbf, active,
                                        self.alpha, self.iterations)
            action = action.clone()
            action[..., :3] = v_safe
            tensordict.set(self.action_key, action)
        return tensordict


# ------------------------------------------------------------------ config plumbing
def extract_cbf_params(cfg):
    """Pull the NavVel ``cbf:`` section + geometry/limits into a flat dict.

    ``cfg`` is the hydra config object (task defaults mounted under ``cfg.task``).
    Tolerates a missing section entirely (returns None -> caller disables CBF).
    """
    try:
        from omegaconf import OmegaConf
    except Exception:
        OmegaConf = None

    task = cfg.task if hasattr(cfg, "task") else cfg
    cbf = task.get("cbf", None)
    if cbf is None:
        return None
    if OmegaConf is not None and OmegaConf.is_config(cbf):
        cbf = OmegaConf.to_container(cbf, resolve=True)
    cbf = dict(cbf) if isinstance(cbf, dict) else {}

    oc = task.get("obstacle", None)
    if OmegaConf is not None and oc is not None and OmegaConf.is_config(oc):
        oc = OmegaConf.to_container(oc, resolve=True)
    oc = oc or {}

    vl = task.get("vel_limit", None)
    if OmegaConf is not None and vl is not None and OmegaConf.is_config(vl):
        vl = OmegaConf.to_container(vl, resolve=True)
    vl = vl or {}

    mode = str(cbf.get("mode", "none"))
    v_max = cbf.get("max_vel", None) or vl.get("max_vel", 1.8)
    return {
        "mode": mode,
        "alpha": float(cbf.get("alpha", 1.0)),
        "r_safety_margin": float(cbf.get("r_safety_margin", 0.1)),
        "use_brake_term": bool(cbf.get("use_brake_term", True)),
        "a_max": float(cbf.get("a_max", 2.0)),
        "v_max": float(v_max),
        "drone_radius": float(oc.get("drone_radius", 0.15)),
        "inflation": float(oc.get("inflation", 0.05)),
        "reward_weight": float(cbf.get("reward_weight", 0.5)),
        "penalty_intrude": bool(cbf.get("penalty_intrude", True)),
        "filter_grad": str(cbf.get("filter_grad", "detach")),
        "iterations": int(cbf.get("filter_iterations", 3)),
    }


def build_cbf_filter(cfg, action_key=("agents", "action")):
    """Return a CBFVelocityFilter for the env action chain, or None if not needed.

    filter_only / hybrid -> filter + record v_nom.
    reward_only           -> record v_nom only (filter off; env applies the reward core).
    none                  -> None (env behaves exactly like naive).
    """
    p = extract_cbf_params(cfg)
    if p is None or p["mode"] == "none":
        return None
    return CBFVelocityFilter(
        action_key=action_key,
        alpha=p["alpha"],
        iterations=p["iterations"],
        filter_grad=p["filter_grad"],
        do_filter=(p["mode"] in ("filter_only", "hybrid")),
    )
