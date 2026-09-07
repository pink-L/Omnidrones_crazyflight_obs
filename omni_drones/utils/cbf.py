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
[M2-3, 2026-09-05] NavVel *速度指令* 动作层的一阶 CBF（控制屏障函数，control-barrier-function）
安全层（参见 m2_plan.md §5 与 navvel_rl_interface.md §2/§6）。

数学描述（静态障碍物，v_o = 0；速度指令 -> 底层 Lee 控制器存在滞后，因此 CBF 的
决策半径在几何半径之上还需额外计入一段制动余量）：

    h_i(p)   = ||p - p_oi|| - r_si^cbf          （到 CBF 球的带符号距离）
    n_i      = (p - p_oi) / ||p - p_oi||
    dot h_i  = n_i^T v
    CBF 约束： dot h_i + alpha h_i >= 0  <=>  n_i^T v >= -alpha h_i

    r_si^cbf = r_si + margin + (v_max^2 / (2 a_max))     [r_si = r_drone + r_oi + inflation]
                                                         （制动 / 内环余量）

* ``filter_velocity``  将标称速度指令投影到各激活半平面的交集中（迭代闭式求解：
  每轮只修正违约最严重的那一个约束），这正是 CBF-RL 针对多障碍物的离散闭式滤波器。
* ``cbf_violation``    返回供奖励核心使用的 *滤波前* 总约束违例量
  （soft-CBF / reward_only / hybrid）：sum_i min(0, n_i^T v_nom + alpha h_i)
  （当 h_i < 0，即已进入 CBF 球内部时，可选择额外叠加惩罚项）。

所有几何运算均为基于环境坐标系位置的纯 torch 计算，可在 CPU 上做单元测试（无需 isaac）。
地面真值障碍物输入（pos / r_safe / active）通过 tensordict 的 ``("info", "obstacle_cbf")``
通道来自环境中的 ObstacleManager，因此 CBF / 观测 / 碰撞 / 奖励共用同一数据源。
CBF 半径刻意不提供给策略观测。
"""

import torch

try:  # torchrl 可用性守卫：仅需纯函数时避免导入失败
    from torchrl.envs.transforms import Transform
    _TORCHRL = True
except Exception:  # pragma: no cover - 纯 torch 测试可不依赖 torchrl
    Transform = object
    _TORCHRL = False


# --------------------------------------------------------------------------- 半径
def safety_radius_extra(drone_radius, inflation, margin, v_max, a_max,
                        use_brake_term=True):
    """在几何决策半径 r_s = r_drone+r_o+infl 之上额外增加的净空。

    当启用制动项时返回 ``margin + v_max**2/(2 a_max)``（米），否则仅返回 ``margin``。
    """
    extra = float(margin)
    if use_brake_term and a_max is not None and float(a_max) > 0:
        extra += float(v_max) ** 2 / (2.0 * float(a_max))
    return float(extra)


def cbf_safety_radius(r_o, drone_radius, inflation, margin, v_max, a_max,
                      use_brake_term=True):
    """单个障碍物的 CBF 球半径（米）：几何半径 r_s + 安全/制动余量。"""
    r_s = float(drone_radius) + float(r_o) + float(inflation)
    return r_s + safety_radius_extra(drone_radius, inflation, margin, v_max, a_max,
                                     use_brake_term)


def safety_obs_channels(dmin, extra, norm, add_clearance, add_cbf_margin):
    """[New2, 2026-09-07] obs 结构级 internalize：CBF 边界余量 / min_clearance 标量通道。

    new2_plan.md §2：把「CBF filter 触发边界」作为显式状态喂给策略（reward-shaping
    已证不可及 → obs 结构级改造）。两把"尺子"都是纯几何量（训练/部署同源、sim2real 只需
    障碍位置可复算、不依赖 filter 在线），故不会像「filter 介入量」那样在撤 filter 部署时
    产生分布偏移（navvel_rl_interface.md §6 口径变更预告）。

    Args:
        dmin (N,1): 活动障碍上的最小表面净空 min_i(||p-p_oi||-r_si)
                    （无活动障碍 = +inf，见 ObstacleManager.min_clearance）。
        extra (float): CBF 决策半径附加余量 r_si^cbf = r_si + extra（= cbf_extra；
                      无 CBF / naive(mode=none) 时传 0 → cbf_margin 退化为 min_clearance）。
        norm (float): 归一化尺度 (m)；默认用 danger_radius(0.6) 量级。
        add_clearance / add_cbf_margin: 是否输出对应通道（对应 obs_safety:
                       clearances | cbf_margin | both）。

    Returns: list[(N,1,1)]，按 [clearance?, cbf_margin?] 顺序（可直接 torch.cat 进 obs 尾；
             obs 拼装为 3D (N,1,f)，标量须 unsqueeze(1) 对齐，见 nav_vel._compute_state_and_obs）。

    通道值 = clamp(d / norm, -1, 1)（both 尺子同一 dmin 的仿射: clearance=dmin,
             cbf_margin=h=dmin-extra）。无活动障碍 dmin=+inf → 通道 = +1（完全安全）。
             cbf_margin 的 0 穿越点 = filter 介入边界（h<0 ⇔ 已入 CBF 决策球）。
             ⚠️ clamp 下限刻意保留「负余量=危险」信息到 -1，不用 relu 剪掉。
    """
    d = dmin / float(norm)                             # (N,1)，inf -> clamp 到 +1
    channels = []
    if add_clearance:
        channels.append(d.clamp(-1.0, 1.0).unsqueeze(1))       # (N,1,1)
    if add_cbf_margin:
        h = (dmin - float(extra)) / float(norm)
        channels.append(h.clamp(-1.0, 1.0).unsqueeze(1))       # (N,1,1)
    return channels


def h_boundary_penalty(dmin, extra, buffer, weight):
    """[New2/E1-v2, 2026-09-07] CBF 边界余量罚（CPU 可测，nav_vel reward core 调用）。

    h = dmin - extra（CBF 边界余量, 0 穿越点 = filter 介入边界; extra=cbf_extra）。
    pen = weight * relu(buffer - h) = weight * relu(buffer + extra - dmin)
      - buffer=0 → pen = weight * relu(-h)（[E1] 原版: 只在 h<0 = 已进 filter 决策区才罚;
        fire 少且 filter 兜底几乎不让 h 深负 → 梯度稀）。
      - buffer>0 → 在接近 filter 边界前 buffer 就开始罚（类 CBF 版 near_slowdown/soft wall,
        提前给"守边界"梯度, 治 E1 罚不着的机制）。软墙位置 = dmin < extra + buffer。
    无活动障碍 dmin=inf → h=inf → relu(...)=0（无罚）。返回与 dmin 同形状 tensor。
    """
    h = dmin - float(extra)
    return float(weight) * torch.relu(float(buffer) - h)


# --------------------------------------------------------------- 纯 torch 核函数
def _gradients(pos, p_obs):
    """计算到一组障碍物的距离、外法向单位向量及相关输入。

    pos (...,3), p_obs (...,K,3) -> dvec (...,K,3), dist (...,K), n (...,K,3)。
    ``n`` 为向外（由障碍物中心指向无人机）的单位法向量，因此正对障碍物飞行时
    n.v < 0，CBF 约束写为  n.v + alpha*h >= 0。
    """
    dvec = pos.unsqueeze(-2) - p_obs                    # 中心 -> 无人机（向外）
    dist = torch.norm(dvec, dim=-1)                     # (...,K)
    n = dvec / (dist.unsqueeze(-1) + 1e-6)              # (...,K,3) 单位外法向量
    return dvec, dist, n


def filter_velocity(pos, v_nom, p_obs, r_safe, active, alpha, iterations=3):
    """把 ``v_nom`` 以迭代闭式方式投影到 CBF 安全半平面的交集内。

    Args:
        pos (...,3): 无人机在环境坐标系下的位置。
        v_nom (...,3): 标称（策略输出的）速度指令。
        p_obs (...,K,3): 障碍物中心（环境坐标系）。未激活的槽位可以是任意值
                         （由 ``active`` 掩码屏蔽）。
        r_safe (...,K): 每个槽位的 CBF 球半径（未激活 -> <=0）。
        active (...,K): 布尔掩码（与观测/几何同源：半径>0）。
        alpha: CBF 收敛速率。
        iterations: "修正违约最严重的约束" 的迭代轮数。

    Returns:
        v_safe (...,3): 投影后的指令（不会比当前指令更差）。
        fix_norm (...,): 被投影掉的合计速度大小（统计/诊断用），>=0。
    """
    v = v_nom.clone()
    fix_norm = torch.zeros(pos.shape[:-1], dtype=v.dtype, device=v.device)
    for _ in range(int(iterations)):
        _, dist, n = _gradients(pos, p_obs)
        h = dist - r_safe                                        # (...,K)
        g = (n * v.unsqueeze(-2)).sum(-1) + alpha * h            # n.v + alpha h
        # 只统计激活槽位上的违例（未激活 -> +inf，永远不会被选中）
        g_act = torch.where(active, g, torch.full_like(g, float("inf")))
        delta = torch.clamp(-g_act, min=0.0)                     # (...,K) 违例量 >=0
        viol = delta.max(dim=-1).values                          # (...,) 各样本最大违例
        need = (viol > 0).any()
        if not need:
            break
        # 只沿违约最严重约束的外法向方向回推：
        #   v *= v - (-g) n = v + (n.v + alpha h) n  （单个障碍物的闭式解）
        best = (delta == viol.unsqueeze(-1)) & active            # one-hot 掩码 (...,K)
        n_sel = (n * best.unsqueeze(-1)).sum(dim=-2)             # (...,3) 选中的法向量
        v = v + viol.unsqueeze(-1) * n_sel
        fix_norm = fix_norm + viol
    return v, fix_norm


def cbf_violation(pos, v_nom, p_obs, r_safe, active, alpha, penalty_intrude=True):
    """供奖励核心使用的滤波前 CBF 总违例量（<=0）。

    对所有激活槽位求和 sum_i min(0, n_i^T v_nom + alpha h_i)；可选地对每个已处于
    CBF 球内部（h_i < 0）的槽位再额外叠加一项 min(0, h_i)，从而无论当前指令方向
    如何，都会对"已侵入球内"进行惩罚。

    Returns (...,) 取值 <= 0（越小表示越不安全）。
    """
    _, dist, n = _gradients(pos, p_obs)
    h = dist - r_safe                                            # (...,K)
    g = (n * v_nom.unsqueeze(-2)).sum(-1) + alpha * h            # n.v_nom + alpha h
    viol = torch.clamp(g, max=0.0)                               # (...,K) 违例 <=0
    viol = torch.where(active, viol, torch.zeros_like(viol))
    total = viol.sum(dim=-1)
    if penalty_intrude:
        intr = torch.clamp(h, max=0.0)
        intr = torch.where(active, intr, torch.zeros_like(intr))
        total = total + intr.sum(dim=-1)
    return total


# --------------------------------------------------------------- torchrl 变换（Transform）
class CBFVelocityFilter(Transform):
    """置于动作链中 ``VelController`` 之前的速域 CBF 滤波器。

    读取策略输出的原始 4 维指令（3 维速度 + yaw），把其中的平动部分投影到 CBF
    半平面内，并把结果写回 ``action_key``（yaw 保持不变）。同时把滤波前的指令
    记录到 ``("info", "policy_action")``，使环境奖励核心（reward_only / hybrid）能够
    恰好惩罚策略的实际输出（CBF-RL soft-CBF）。需要 ``("info", "obstacle_cbf")``
    提供每个环境、每个槽位的环境坐标系几何信息，形状为 (..., K, 4) =
    [p_oi(3), r_si^cbf(1)]，未激活槽位必须为全零。

    放置说明：torchrl 的 ``Compose`` 会按 *逆* 添加顺序执行 inv（动作侧）变换，
    因此本变换必须在动作链表中追加在 ``VelController`` 之后。
    """

    def __init__(
        self,
        action_key=("agents", "action"),
        alpha=1.0,
        iterations=3,
        filter_grad="detach",        # detach | through（through 保留计算图；环境中未使用）
        do_filter=True,              # False -> 只记录 v_nom 不做滤波（reward_only 模式）
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
        drone_state = tensordict[("info", "drone_state")]      # (...,13) 位置在前 [:3]
        obs_cbf = tensordict[("info", "obstacle_cbf")]         # (...,K,4)
        action = tensordict[self.action_key]                   # (...,4) 原始速度指令

        pos = drone_state[..., :3]
        p_obs = obs_cbf[..., :3]                               # (...,K,3)
        r_cbf = obs_cbf[..., 3]                                # (...,K)
        active = r_cbf > 0                                     # 以半径>0 作为激活判据

        if self.filter_grad == "through":
            v_nom = action[..., :3]
        else:
            v_nom = action[..., :3].detach()
            p_obs = p_obs.detach()
            r_cbf = r_cbf.detach()

        # 记录滤波前的指令供奖励核心使用（完整的 4 维，含 yaw）
        tensordict.set(("info", "policy_action"), action.clone())

        if self.do_filter and active.any():
            v_safe, _ = filter_velocity(pos, v_nom, p_obs, r_cbf, active,
                                        self.alpha, self.iterations)
            action = action.clone()
            action[..., :3] = v_safe
            tensordict.set(self.action_key, action)
        return tensordict


# ------------------------------------------------------------------ 配置接线（config plumbing）
def extract_cbf_params(cfg):
    """把 NavVel 的 ``cbf:`` 配置段 + 几何/限速参数抽取为扁平的字典。

    ``cfg`` 为 hydra 配置对象（task 默认参数挂在 ``cfg.task`` 下）。
    完全容忍缺少该配置段的情况（返回 None -> 调用方禁用 CBF）。
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
    """为环境动作链返回一个 CBFVelocityFilter；若不需要则返回 None。

    filter_only / hybrid -> 做滤波 + 记录 v_nom。
    reward_only          -> 只记录 v_nom（不滤波；由环境施加奖励核心）。
    none                 -> None（环境行为与 naive 完全一致）。
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


class CmdGaussNoise(Transform):
    """[M3-B 2026-09-06] 训练侧速度指令高斯噪声（domain randomization）。

    给动作的平动速度分量 (..., :3) 注入零均值高斯噪声（std = sigma m/s），yaw 不动。
    作用链位置：CBF 投影之后、VelController 之前 —— 滤波器先保证安全，再叠加执行层
    误差，等价于论文的动力学噪声 d ~ N(0, 20% v_max)（q 更新时位置积分带误差）。
    reward core 仍罚策略自己的滤波前指令（policy_action），不罚本噪声 → 策略学会
    “安全指令会被执行误差打偏 → 留裕量”，从而 internalize 对动力学不确定的鲁棒。

    放置：Compose 的 inv（动作侧）按“逆添加序”执行。若想让执行序为
    CBF filter -> 本噪声 -> VelController，则本变换必须在动作链中追加在
    VelController 之后、且 cbf_filter 之前（即 add 顺序 VelController, this, CBF）。
    """

    def __init__(self, action_key=("agents", "action"), sigma=0.0):
        if not _TORCHRL:
            raise RuntimeError("CmdGaussNoise requires torchrl")
        super().__init__([], in_keys_inv=[action_key])
        self.action_key = action_key
        self.sigma = float(sigma)

    def _inv_call(self, tensordict):
        if self.sigma > 0:
            action = tensordict.get(self.action_key)
            if action.shape[-1] >= 3:
                noise = torch.randn_like(action[..., :3]) * self.sigma
                action = action.clone()
                action[..., :3] = action[..., :3] + noise
                tensordict.set(self.action_key, action)
        return tensordict
