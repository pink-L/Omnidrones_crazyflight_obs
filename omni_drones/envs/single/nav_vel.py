# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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


import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones.utils.torchrl.compat import CompositeSpec, UnboundedContinuousTensorSpec
from omni_drones.envs.utils import create_obstacle
from omni_drones.envs.single.nav_vel_obstacles import ObstacleManager, scene_slot_count
from omni_drones.utils.nav_curriculum import ObstacleCurriculum
from omni_drones.utils.cbf import (
    cbf_violation,
    filter_velocity,
    h_boundary_penalty,
    safety_obs_channels,
    safety_radius_extra,
)


def _to_plain_dict(obj):
    """Hydra DictConfig -> plain python dict (None-safe)."""
    if obj is None:
        return None
    from omegaconf import OmegaConf
    return OmegaConf.to_container(obj, resolve=True)


class NavVel(IsaacEnv):
    r"""
    [M1, 2026-09-04] Waypoint-navigation with a VELOCITY action layer (CBF1-ready).

    Each episode the drone starts at a random pose and must fly to a random
    3D target waypoint and hold there (this is the "goal reaching" primitive
    that CBF-based safety filtering will later be attached to).

    The policy outputs a velocity command `[vx, vy, vz, yaw]` (world frame),
    which is converted to rotor commands by a low-level controller through the
    `action_transform: velocity` (VelController -> LeePositionController) in
    train.py / play.py. This env only manages task logic (targets, rewards,
    termination) and provides the `info` keys consumed by controller Transforms.

    ## Observation (all in the drone's env frame -> translation invariant)

    - `rpos` (3): target position - drone position.
    - `drone_state` (state_dim - 3): drone state except position.
    - `rheading` (3): target heading - drone heading.
    - `time_encoding` (optional, 4).

    ## Reward

        r = r_pos + r_pos*(r_up + r_spin) + r_effort + r_smooth + r_arrive

    where `r_pos = 1/(1+(k*d)^2)`, `d = ||rpos||` and `r_arrive` is a sparse
    bonus given once when the drone stays within `arrive_radius` for
    `arrive_hold_steps` consecutive steps.

    ## Episode End

    Terminated when the drone crashes (z too low), leaves the workspace
    (`bound_xy`/`bound_z`), produces NaN, or (optionally) right after a first
    successful arrival (`success_terminate`). Truncated at max_episode_length.
    """

    def __init__(self, cfg, headless):
        # rewards / obs options
        self.reward_effort_weight = cfg.task.get("reward_effort_weight", 0.1)
        self.reward_action_smoothness_weight = cfg.task.get("reward_action_smoothness_weight", 0.0)
        self.reward_distance_scale = cfg.task.get("reward_distance_scale", 1.6)
        self.time_encoding = cfg.task.get("time_encoding", True)
        self.randomization = cfg.task.get("randomization", {})
        # [NR-1 2026-09-06] reward scheme: legacy(默认=Hover 常驻基座版, 逐位不变) |
        #   navgoal(论文式纯 motion: progress 势差 γ建议1.0; 实证=from-scratch 死锁+warm 无增益,
        #   保留作对照) | r1([New1 2026-09-06] motion-gated 接近引导: 接近势只在朝目标运动时给正,
        #   悬停=0、远离=0、越近越强 -> 保留稠密引导又无悬停刷分平台; 实证 from-scratch 死锁
        #   (接近势核 k=1.6 远端≈0→远端无引导), 保留作对照) | f1([New1/F1 2026-09-06] 无核飞行主导:
        #   纯 ① relu(v·u_g)/vmax + 到达, 无距离核远端=近端同强, 治 r1"起飞吸引力不够"; 见 new_reward.md §F1)。
        #   方案文档 drones/new_reward.md / new1_plan.md。navgoal/r1/f1 才附加 first_arrival_step/mean_action_diff stat。
        self.reward_scheme = str(cfg.task.get("reward_scheme", "legacy"))
        # [NR-1] 到达"时间 bonus" w_t: +w_t*(1 - t_arr/T), 越快越多; 0=仅 flat base (navgoal/r1/f1)
        self.arrive_time_bonus = float(cfg.task.get("arrive_time_bonus", 0.0))
        # [New1/R1] motion-gated 接近引导权重 (r1 主稠密项): w_g * relu(v·u_g)/vmax * 1/(1+(k·d)²)
        self.reward_gate_weight = float(cfg.task.get("reward_gate_weight", 0.0))
        # [New1/F1] 无核飞行主导权重 (f1 主项): w_f * relu(v·u_g)/vmax; 0=关 (new_reward.md §F1)
        self.reward_fly_weight = float(cfg.task.get("reward_fly_weight", 0.0))
        # [2026-09-08 f1-圈内引导] k_in: 进 arrive_radius 圈内把 fly 替换为
        #   r_zone = w_f + k_in*(1 - d/R) (R=arrive_radius): 边界=w_f(连续无悬崖)、中心=w_f+k_in、
        #   随 d↓ 单调增 -> 治确定性"贴圈不进/进圈不停"; 默认 0=关(逐位兼容)
        self.reward_zone_weight = float(cfg.task.get("reward_zone_weight", 0.0))
        # [New1/F1-②] 可选 per-step 时间成本 (仅 f1): 悬停/loiter 持续亏, 治"时间短/完成"; 默认 0=关
        self.reward_time_cost = float(cfg.task.get("reward_time_cost", 0.0))
        vl_cfg = cfg.task.get("vel_limit", None) or {}
        self.max_vel = float(vl_cfg.get("max_vel", 1.8))

        # [M1] waypoint / arrival / termination
        tpr = torch.as_tensor(cfg.task.target_pos_range, dtype=torch.float32)   # (2, 3)
        ipr = torch.as_tensor(cfg.task.init_pos_range, dtype=torch.float32)     # (2, 3)
        self.target_pos_range = tpr
        self.init_pos_range = ipr
        self.min_init_target_dist = cfg.task.get("min_init_target_dist", 1.5)
        self.arrive_radius = cfg.task.get("arrive_radius", 0.5)
        self.arrive_hold_steps = cfg.task.get("arrive_hold_steps", 100)
        self.arrive_bonus = cfg.task.get("arrive_bonus", 5.0)
        self.success_terminate = cfg.task.get("success_terminate", False)
        self.bound_xy = cfg.task.get("bound_xy", 5.0)
        self.z_min = cfg.task.get("z_min", 0.15)
        self.z_max = cfg.task.get("z_max", 4.5)
        # [env_design 2026-09-07] 6×6×3 箱式房间越界(可选): arena_bound=[bx,by] 为 x/y 半宽,
        #   |x|>bx 或 |y|>by 即 OOB(与圆形 bound_xy 是"或"关系); None/缺省 = 不用箱式(向后兼容)。
        #   本设计 = 6×6 房间边界 -> [3.0,3.0] (飞出房间 = 失败, 配合 soft_respawn=false 硬终止)。
        _ab = cfg.task.get("arena_bound", None)
        if _ab is not None:
            self.arena_bound_x = float(_ab[0])
            self.arena_bound_y = float(_ab[1])
        else:
            self.arena_bound_x = None
            self.arena_bound_y = None
        # [M1 2026-09-04] survival / soft-respawn (diagnosed: drones keep falling
        # before they can learn to hover/navigate, see plan §10.5 + 200M-run record)
        self.soft_respawn = cfg.task.get("soft_respawn", True)
        self.survival_penalty_weight = cfg.task.get("survival_penalty_weight", 0.0)
        self.z_ref = cfg.task.get("z_ref", 1.0)
        # [M1 2026-09-04 / guide §1.2+§6] stage-0 convergence fixes (all optional)
        self.reward_timeout_penalty = cfg.task.get("reward_timeout_penalty", 0.0)
        self.reward_oob_penalty = cfg.task.get("reward_oob_penalty", 0.0)
        self.reward_crash_penalty = cfg.task.get("reward_crash_penalty", 0.0)
        self.reward_early_death_weight = cfg.task.get("reward_early_death_weight", 0.0)
        self.early_death_threshold_steps = cfg.task.get("early_death_threshold_steps", 300)
        # [Arena1-B 2026-09-07] 每次 soft-respawn(坠地 crash/出界 oob/碰撞超限) 一次性惩罚:
        #   治"多命蒙到到达"。训练 soft_respawn=true 多命下, 策略可撞了重生反复试, 600 步窗内
        #   曾到达(episode_any_arrival)即可把窗口 success_rate 抬到 ~0.7, 但严格单命验收
        #   (soft_respawn=false, 一次飞行失败即失败) 一次都飞不过去(arrival<1%)。
        #   给"每丢一条命"扣一次罚 -> 失误(撞/坠/出界)有即时成本, 逼策略学"单次无失误穿越"
        #   而非依赖复活。默认 0=关(逐位兼容)。⚠️ 量级勿大(>~15 易致 return 大幅震荡, M1
        #   早死反比惩罚 return≈-110 的前车之鉴); 建议 2~10 起步。
        self.reward_respawn_penalty = float(cfg.task.get("reward_respawn_penalty", 0.0))
        self.reward_pbrs_weight = cfg.task.get("reward_pbrs_weight", 0.0)
        self.pbrs_gamma = cfg.task.get("pbrs_gamma", 0.995)

        # ---- [M2 2026-09-04] obstacle / curriculum config ----------------
        # parsed BEFORE super().__init__ because _design_scene/_set_specs run inside it
        self._obstacle_cfg = _to_plain_dict(cfg.task.get("obstacle", None))
        self._curriculum_cfg = _to_plain_dict(cfg.task.get("curriculum", None))
        self._has_obstacles = bool(self._obstacle_cfg and self._obstacle_cfg.get("max_slots", 0))
        if self._has_obstacles:
            oc = self._obstacle_cfg
            self.K = int(oc["max_slots"])          # obs 槽位窗口 K（obs 维度 = 30 + 4K）
            # [M2 2026-09-05 obs-window] num_scene M = 场景物理障碍数（缓冲区/prim 数）。
            #   M > K -> obs 每步实时取最近 K 个（滑动窗口, obs 恒 62 维）；M <= K 保持固定槽旧语义。
            # [2026-09-08 pillar C] n_pillars>0 时 M = 柱层球数(n_pillar*pillar_layers) + 自由球数
            #   (混合布局, 忽略 num_scene)。
            # [P1 A2/A3 2026-09-12] 随机化路径（n_pillars_range / pillar_layers_range）下，
            #   ObstacleManager 用 **上界** n_max*L_max 预留槽位 ⇒ 这里必须同样用上界，否则
            #   `obstacle_views`（prim 数）与采到的障碍数不一致，reset 时 set_world_poses
            #   会 shape mismatch（1024×24 vs 1024×16）→ Isaac 再把它放大成 SIGSEGV。
            #   公式现收敛到 ObstacleManager 侧的 `scene_slot_count` **单一来源**，
            #   并在 manager 构造后断言一致。
            self.M = scene_slot_count(oc)
            _ow = oc.get("obs_window")
            self.obs_window = bool(self.M > self.K) if _ow is None else bool(_ow)
            self.obstacle_phys_radius = float(max(oc.get("radius_choices", [0.30])))
            self.obstacle_collision_margin = float(oc.get("collision_margin", 0.05))
            self.obstacle_danger_radius = float(oc.get("danger_radius", 0.6))
            self.obstacle_max_collisions = int(oc.get("max_collisions", 2))
            self.reward_obs_log_weight = float(oc.get("reward_obs_log_weight", 1.5))
            self.reward_obs_log_scale = float(oc.get("reward_obs_log_scale", 0.3))
            # [P3 2026-09-16] 障碍 log 距离项符号（用户裁决 = 应为惩罚）：
            #   legacy  = `reward -= wλΣφ`（φ=ln(d/D)<0 ⇒ 危险区内实为**正奖励**；
            #             A0–A4 冻结口径的历史实现，逐位保留）
            #   penalty = `reward += wλΣφ`（危险区内为负 = 惩罚，与 kaiwu 原型/设计文档一致）
            self.reward_obs_log_mode = str(oc.get("reward_obs_log_mode", "legacy"))
            if self.reward_obs_log_mode not in ("legacy", "penalty"):
                raise ValueError(f"unknown reward_obs_log_mode: "
                                 f"{self.reward_obs_log_mode!r} (legacy|penalty)")
            self.reward_collision_edge = float(oc.get("reward_collision_edge", 2.0))
            self.reward_near_slowdown_weight = float(oc.get("reward_near_slowdown_weight", 0.5))
            self.obstacle_reward_early_death_weight = float(oc.get("reward_early_death_weight", 0.0))
            self.obstacle_early_death_threshold = float(oc.get("early_death_threshold_steps", 300))
            self.obstacle_obs_dist_norm = float(oc.get("obs_dist_norm", 5.0))
            # [New2 2026-09-07] obs 结构级 internalize (new2_plan.md §2): 把 CBF filter
            #   触发边界作为显式状态喂给策略（reward-shaping 已证不可及 → 换 obs 结构手段）。
            #   none=不加(62 维, 与 New1 pg6q5ji5 逐位兼容); clearance/cbf_margin/both 在
            #   障碍块后追加 1/1/2 维标量通道(旧 ckpt 因维度变化不可 warm 续, 同 30→62 先例)。
            self.obs_safety = str(oc.get("obs_safety", "none"))  # none|clearance|cbf_margin|both
            if self.obs_safety not in ("none", "clearance", "cbf_margin", "both"):
                raise ValueError(
                    "task.obstacle.obs_safety must be one of none|clearance|cbf_margin|both, "
                    f"got {self.obs_safety!r}")
            # 归一化尺度: 默认 danger_radius(0.6) 量级; 显式 0 -> 沿用 danger_radius
            self.obs_safety_norm = float(oc.get("obs_safety_norm", 0.0)) \
                or self.obstacle_danger_radius
            self.obs_safety_add_clearance = self.obs_safety in ("clearance", "both")
            self.obs_safety_add_cbf_margin = self.obs_safety in ("cbf_margin", "both")
            self.obs_safety_dim = int(self.obs_safety_add_clearance) \
                + int(self.obs_safety_add_cbf_margin)
            # [M2-3] CBF1 config (velocity-layer safety layer; see m2_plan.md §5).
            #   cbf_extra = margin + brake allowance added on top of the geometric r_s.
            #   mode: none | filter_only | reward_only | hybrid (transform in train/play/
            #   eval_ckpt; reward core is applied below in _compute_reward_and_done).
            self._cbf_cfg = _to_plain_dict(cfg.task.get("cbf", None))
            ccbf = self._cbf_cfg or {}
            self.cbf_mode = str(ccbf.get("mode", "none"))
            self.cbf_use_filter = self.cbf_mode in ("filter_only", "hybrid")
            self.cbf_use_reward_core = self.cbf_mode in ("reward_only", "hybrid")
            self.cbf_alpha = float(ccbf.get("alpha", 1.0))
            self.cbf_reward_weight = float(ccbf.get("reward_weight", 0.5))
            self.cbf_penalty_intrude = bool(ccbf.get("penalty_intrude", True))
            self.cbf_penalty_src = str(ccbf.get("penalty_src", "nominal"))
            self.cbf_iterations = int(ccbf.get("filter_iterations", 3))
            self.cbf_correction_sigma = float(ccbf.get("correction_sigma", 0.5))
            # [M3-A 2026-09-06] dual reward core 的 w2 (gaussian-correction 项权重);
            #   w1 = reward_weight (nominal-viol 项)。默认 0 → 其余三模式逐位不变。
            self.cbf_correction_weight = float(ccbf.get("correction_weight", 0.0))
            # [New2/E1 2026-09-07] CBF 边界余量罚权重 w_h(见 reward core): 罚 w_h*relu(-h),
            #   h=dmin-cbf_extra(0 穿越=filter 介入边界) → 守边界梯度; 默认 0=关(逐位不变)。
            self.cbf_h_penalty_weight = float(ccbf.get("h_penalty_weight", 0.0))
            # [New2/E1-v2] 罚提前量 buffer: 罚 = w_h*relu(buffer - h) → buffer>0 时在接近
            #   filter 边界前就开始罚(E1 只罚 h<0, fire 少; buffer 给提前梯度)。默认 0 = E1 原版。
            self.cbf_h_penalty_buffer = float(ccbf.get("h_penalty_buffer", 0.0))
            self.cbf_extra = None
            if self.cbf_mode != "none":
                vl = _to_plain_dict(cfg.task.get("vel_limit", None)) or {}
                v_max = ccbf.get("max_vel", None) or float(vl.get("max_vel", 1.8))
                self.cbf_extra = safety_radius_extra(
                    float(oc.get("drone_radius", 0.15)),
                    float(oc.get("inflation", 0.05)),
                    float(ccbf.get("r_safety_margin", 0.1)),
                    float(v_max),
                    float(ccbf.get("a_max", 2.0)),
                    bool(ccbf.get("use_brake_term", True)),
                )
        else:
            self.K = 0
            self.cbf_mode = "none"
            self.cbf_use_filter = False
            self.cbf_use_reward_core = False
            self.cbf_alpha = 1.0
            self.cbf_reward_weight = 0.0
            self.cbf_penalty_intrude = False
            self.cbf_penalty_src = "nominal"
            self.cbf_iterations = 3
            self.cbf_correction_sigma = 0.5
            self.cbf_correction_weight = 0.0
            self.cbf_extra = None
            self.cbf_h_penalty_weight = 0.0
            self.cbf_h_penalty_buffer = 0.0
            self.obs_safety = "none"
            self.obs_safety_norm = 0.6
            self.obs_safety_add_clearance = False
            self.obs_safety_add_cbf_margin = False
            self.obs_safety_dim = 0

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        # [B1 2026-09-09] controller_sync_dr: when the action_transform=velocity script
        #   mounts its low-level Lee controller here (base_env.low_level_controller), each
        #   reset re-syncs it to the per-env RANDOMIZED real mass/KF after drone._reset_idx.
        #   This mimics a real Crazyflie low-level controller that self-calibrates to its
        #   true params; used as a B1 control to test whether the geo7 DR drop is caused by
        #   the nominal-controller mismatch (sim artifact) rather than policy fragility.
        #   Default off -> nominal low-level controller (identical to pre-B1 behavior).
        self.controller_sync_dr = bool(cfg.task.get("controller_sync_dr", False))
        self.low_level_controller = None
        self._b1_synced = False

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()

        # ---- [M2] obstacle prim view + pure-logic manager + curriculum scheduler ----
        if self._has_obstacles:
            self.obstacle_views = RigidPrimView(
                "/World/envs/env_*/obstacle_*",
                reset_xform_properties=False,
                shape=[self.num_envs, self.M],
            )
            self.obstacle_views.initialize()
            self.obstacles = ObstacleManager(self._obstacle_cfg, self.num_envs, self.device)
            # [P1 2026-09-12] 两处独立算 M（env 算 prim 数、manager 算槽位）→ 必须一致。
            #   不一致的表现是 reset 时 set_world_poses shape mismatch，再被 Isaac 的
            #   关闭路径放大成 SIGSEGV（A2 首次开训就是这样丢了两条 20M run）。
            #   这里排错一次，让这类不一致当场报错。
            if int(self.obstacles.M) != int(self.M):
                raise RuntimeError(
                    f"obstacle slot count mismatch: env M={self.M} vs manager M="
                    f"{self.obstacles.M} (check n_pillars/_range and pillar_layers/_range "
                    f"in the profile - both sides must use the same upper bound)")

            cc = self._curriculum_cfg or {}
            # [New2/E3] margin gate: 提升还需窗口滚动 margin_ok_rate >= margin_frac
            #   (margin_ok = 整窗表面净空 >= margin_clearance, 默认 0.1 ≈ brake-off cbf_extra
            #   → 等价 h>=0 = 从未进 CBF filter 介入区 = internalize 语义)。默认关。
            self.curriculum_margin_gate = bool(cc.get("margin_gate", False))
            self.curriculum_margin_frac = float(cc.get("margin_frac", 0.5))
            self.curriculum_margin_clearance = float(cc.get("margin_clearance", 0.1))
            self.curriculum = ObstacleCurriculum(
                levels=cc.get("levels", [0, 2, 4, 8]),
                initial_level=int(cc.get("initial_level", 0)),
                gate_window=int(cc.get("gate_window_episodes", 3000)),
                success_threshold=float(cc.get("success_rate_threshold", 0.8)),
                collision_threshold=float(cc.get("collision_rate_threshold", 0.05)),
                min_frames=float(cc.get("min_frames_between_promote", 2_000_000)),
                allow_demote=bool(cc.get("allow_demote", False)),
                margin_gate=self.curriculum_margin_gate,
                margin_frac=self.curriculum_margin_frac,
                device=self.device,
            )
            self.curriculum_enabled = bool(cc.get("enabled", True))
            self.curriculum_levels = list(cc.get("levels", [0, 2, 4, 8]))
            self.level_idx = self.curriculum.level_idx
            # per-life / per-episode(600-step window) collision bookkeeping
            self.ep_collision_edges = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
            self.life_collision_edges = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
            self.prev_in_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        else:
            self.obstacle_views = None
            self.obstacles = None
            self.curriculum = None
            self.curriculum_enabled = False
            self.curriculum_levels = []
            self.level_idx = 0

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(ipr[0].to(self.device), ipr[1].to(self.device))
        self.target_pos_dist = D.Uniform(tpr[0].to(self.device), tpr[1].to(self.device))
        # [Arena1 2026-09-07] 固定起终点(可选): fixed_init/fixed_target 非 None 时 init/target
        #   退化为常量(随机 yaw 保留), 任务 = 固定穿越障碍场. 布局净空/重生逻辑不变(见
        #   _sample_init_pos/_sample_target_pos). cfg.task.fixed_init / fixed_target。
        _fi = cfg.task.get("fixed_init", None)
        _ft = cfg.task.get("fixed_target", None)
        self.fixed_init = (torch.as_tensor(_fi, dtype=torch.float32, device=self.device)
                           if _fi is not None else None)
        self.fixed_target = (torch.as_tensor(_ft, dtype=torch.float32, device=self.device)
                             if _ft is not None else None)
        # [env_design 2026-09-07] 起终点"任务采样器": uniform(全范围随机, 旧默认/向后兼容) |
        #   edge(6×6×3 NavRL 风格: start 恒在 x 左带 [-outer,-inner], target 恒在右带
        #   [inner,outer], y 全覆盖障碍区, z 随机 [edge_z_lo,edge_z_hi])。fixed_* 非 None 时
        #   一律优先(fixed 常量, 与 sampler 无关, eval 用)。
        self.episode_sampler = str(cfg.task.get("episode_sampler", "uniform"))
        self.edge_inner = float(cfg.task.get("edge_inner", 2.6))   # 6×6 内缩 0.4 -> 2.6
        self.edge_outer = float(cfg.task.get("edge_outer", 3.0))   # 6×6 半宽 3.0
        _ey = cfg.task.get("edge_y_range", [-3.0, 3.0])            # y 全覆盖(障碍区)
        _ez = cfg.task.get("edge_z_range", [0.4, 2.4])             # z 随机(> z_min 即不算坠地)
        self.edge_y_lo, self.edge_y_hi = float(_ey[0]), float(_ey[1])
        self.edge_z_lo, self.edge_z_hi = float(_ez[0]), float(_ez[1])
        # small initial tilt + random yaw
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 2.0], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 2.0], device=self.device) * torch.pi
        )

        # per-env target (env frame)
        self.target_pos = self._sample_target_pos(self.num_envs)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8

        # [M1] arrival bookkeeping
        self.arrive_timer = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.arrival_triggered = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        # [M1 2026-09-04] per-life / per-episode trackers (failure penalties + PBRS)
        self.life_steps = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.episode_any_arrival = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.prev_goal_dist = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        # [SimpleFlight migration 2026-09-04]: buffers consumed by controller Transforms
        self.prev_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.policy_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)
        # [env_design 2026-09-07] 每窗终止原因(诊断, 供 eval_ckpt 归因; 单命 soft_respawn=false 才有意义):
        #   0=进行中 1=crash(坠地/NaN) 2=oob(出界/z>z_max) 3=collide 4=truncated未到达(超时) 5=truncated已到达
        self.term_cause = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)

    # [Arena1 2026-09-07 / env_design 2026-09-07] 起/终点采样优先级:
    #   fixed_* 非 None -> 常量(全部 env 同点, eval 用);
    #   episode_sampler=="edge" -> x 边带强制对侧(训练用, NavRL 语义: start 左带 x<0 ->
    #     target 右带 x>0, 必穿越场地中央; y 全覆盖障碍区防"贴边走逃逸不避障");
    #   否则 -> 全范围随机 Uniform(旧默认, 向后兼容)。
    def _sample_edge_pos(self, n, left: bool):
        """edge 模式采样 (n,1,3): x ∈ ±[edge_inner, edge_outer) 带, y ∈ [y_lo,y_hi], z ∈ [z_lo,z_hi]."""
        half = self.edge_inner + (self.edge_outer - self.edge_inner) * torch.rand(
            n, 1, device=self.device)                              # [inner, outer)
        x = -half if left else half
        y = self.edge_y_lo + (self.edge_y_hi - self.edge_y_lo) * torch.rand(n, 1, device=self.device)
        z = self.edge_z_lo + (self.edge_z_hi - self.edge_z_lo) * torch.rand(n, 1, device=self.device)
        return torch.stack([x, y, z], dim=-1)                       # (n,1,3)

    def _sample_init_pos(self, n):
        """(n,1,3) init poses: fixed_init 常量 | edge 左带(训练) | 随机 Uniform."""
        if self.fixed_init is not None:
            return self.fixed_init.reshape(1, 1, 3).expand(n, 1, 3).clone()
        if self.episode_sampler == "edge":
            return self._sample_edge_pos(n, left=True)
        return self.init_pos_dist.sample((n, 1))

    def _sample_target_pos(self, n):
        """(n,1,3) targets: fixed_target 常量 | edge 右带(训练) | 随机 Uniform."""
        if self.fixed_target is not None:
            return self.fixed_target.reshape(1, 1, 3).expand(n, 1, 3).clone()
        if self.episode_sampler == "edge":
            return self._sample_edge_pos(n, left=False)
        return self.target_pos_dist.sample((n, 1))

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        # target visual prim (template; copied to every env by IsaacEnv)
        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 1.5),
        )
        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(),
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])[0]

        # [M2 2026-09-04] K kinematic sphere obstacles (template under env_0, GridCloner
        # copies them into every env). Physical radius fixed to the max logical tier (0.4)
        # so no per-reset USD radius writes are needed; logical r_o (obs/reward/collision)
        # is <= physical, thus the geometric decision radius r_s >= physical ball keeps the
        # geometric detection consistent (triggers at/earlier than physical contact).
        if self._has_obstacles:
            for i in range(self.M):
                create_obstacle(
                    f"/World/envs/env_0/obstacle_{i}", "Sphere",
                    translation=(0.0, 0.0, -100.0),
                    attributes={"radius": self.obstacle_phys_radius},
                )
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        # [M2] append K obstacle slots (rpos_xyz(3) + radius(1) per slot): 30 -> 62
        if self._has_obstacles:
            self.obstacle_obs_dim = 4 * self.K
            observation_dim += self.obstacle_obs_dim
            # [New2] 安全量通道(CBF 边界余量 h / min_clearance)追加在障碍块后: 62 -> 63/64
            observation_dim += self.obs_safety_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_dict = {
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "arrival": UnboundedContinuousTensorSpec(1),   # [2026-09-08] 曾到达∧保持hold_steps步(episode_any_arrival 持存); 非瞬时圈内EMA
            "vel_norm": UnboundedContinuousTensorSpec(1),  # [M1] EMA of speed
            # [M2] obstacle stats (EpisodeStats samples them at the episode end step)
            "collision": UnboundedContinuousTensorSpec(1),  # EMA frac. of steps in contact
            "collision_episodes": UnboundedContinuousTensorSpec(1),  # window had >=1 edge
            "min_clearance": UnboundedContinuousTensorSpec(1),  # EMA min surface clearance
            "success_rate": UnboundedContinuousTensorSpec(1),  # [2026-09-08] window success = (曾到达∧保持50步) & 0碰撞 (episode_any_arrival 已含保持50步; done 事件率)
            "curriculum_level": UnboundedContinuousTensorSpec(1),  # active obstacle count
            "cbf_violation": UnboundedContinuousTensorSpec(1),  # [M2-3] CBF reward-core violation (>=0)
            # [P3 2026-09-16] term_* 分项仪器：每步奖励贡献的 EMA（含符号）。
            #   此前没有分项统计 -> 障碍 log 项符号错误在整个 M2/P0/P1 期间不可见。
            #   只记录、不参与任何门槛；train 的 stats logger 会自动带上这些键。
            "term_obs_log": UnboundedContinuousTensorSpec(1),      # ⑦a 障碍 log 项贡献（含符号）
            "term_near_slow": UnboundedContinuousTensorSpec(1),    # ⑦c 近障减速贡献（<=0）
            "term_collision_edge": UnboundedContinuousTensorSpec(1),  # ⑦b 碰撞边沿贡献（<=0）
            "term_cbf_viol": UnboundedContinuousTensorSpec(1),     # ⑧ w1·viol 贡献（<=0）
            "term_cbf_corr": UnboundedContinuousTensorSpec(1),     # ⑧ -w2·(1-e) 贡献（<=0）
            "term_h_pen": UnboundedContinuousTensorSpec(1),        # h_penalty 贡献（<=0）
            "p_filter": UnboundedContinuousTensorSpec(1),          # [P3] 当步 filter 执行概率
        }
        # [NR-1] navgoal/r1/f1-only stats（legacy 保持 stats_spec 逐位不变）:
        #   first_arrival_step = 窗口内首次到达的 progress_buf 步数(0=从未到达, EpisodeStats
        #                        在 done 步采样 -> 反映整窗); mean_action_diff = 策略速度指令
        #                        差分 EMA(验收"动作平滑").
        if self.reward_scheme in ("navgoal", "r1", "f1"):
            stats_dict["first_arrival_step"] = UnboundedContinuousTensorSpec(1)
            stats_dict["mean_action_diff"] = UnboundedContinuousTensorSpec(1)
        stats_spec = CompositeSpec(stats_dict).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        # info keys consumed by controller Transforms
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "policy_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            # [M2-3] CBF ball channel for CBFVelocityFilter: (n, M, 4) = [p_oi(3), r_si^cbf(1)]
            #   M = num_scene (>= K)；CBF 必须对全部场景障碍安全（不只看 obs 窗口）。
            "obstacle_cbf": UnboundedContinuousTensorSpec(
                (self.drone.n, self.M, 4), device=self.device),
            # [P3 2026-09-16] p_filter / CBF 诊断通道。CBFVelocityFilter 仅在
            #   emit_info=True（train.py 训练入口）时写入；其余调用方不写 -> 恒 0。
            "p_filter": UnboundedContinuousTensorSpec((self.drone.n, 1), device=self.device),
            "cbf_corr": UnboundedContinuousTensorSpec((self.drone.n, 1), device=self.device),
            "cbf_intervened": UnboundedContinuousTensorSpec((self.drone.n, 1), device=self.device),
            "cbf_executed": UnboundedContinuousTensorSpec((self.drone.n, 1), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        # [M2] consume finished episode outcomes into the curriculum BEFORE clearing the
        # per-env window counters below. Only training updates the schedule; eval keeps the
        # level locked. success = arrived at least once AND zero collision edges (window).
        if (self._has_obstacles and self.curriculum is not None
                and self.training and self.curriculum_enabled):
            ran = self.progress_buf[env_ids] > 0
            if ran.any():
                ids = env_ids[ran]
                arrived = self.episode_any_arrival[ids].squeeze(-1)
                edges = self.ep_collision_edges[ids].squeeze(-1)
                success = arrived & (edges == 0)
                collided = edges > 0
                # [New2/E3] margin gate: margin_ok = 整窗几何表面净空 min >= margin_clearance
                #   (ep_min_clearance 仍是本窗结算值; commit_layout 在下方才重置为 inf)。
                #   无活动障碍(inf)视为 OK。cbf_extra=None(naive) 时默认仍用几何净空门槛。
                margin_ok = None
                if self.curriculum_margin_gate:
                    mc = self.obstacles.ep_min_clearance[ids].squeeze(-1)   # (n,)
                    margin_ok = (mc >= self.curriculum_margin_clearance) \
                        | (~torch.isfinite(mc))
                self.curriculum.update(
                    success, collided, margin_ok=margin_ok,
                    add_frames=float(ran.sum().item()) * float(self.max_episode_length))
                self.level_idx = self.curriculum.level_idx

        self.drone._reset_idx(env_ids, self.training)
        self._sync_low_level_controller()

        n = len(env_ids)
        # --- initial pose (env frame; fixed_init -> 常量, Arena1) ---
        pos = self._sample_init_pos(n)
        rpy = self.init_rpy_dist.sample((n, 1)).to(self.device)
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        # --- target waypoint (env frame); fixed_target -> 常量; 仅随机模式做太近重采 ---
        target = self._sample_target_pos(n)
        if self.fixed_init is None and self.fixed_target is None:
            for _ in range(3):
                d = torch.norm(target - pos, dim=-1, keepdim=True)      # (n,1,1)
                too_close = d < self.min_init_target_dist
                if not too_close.any():
                    break
                resample = self._sample_target_pos(n)
                target = torch.where(too_close, resample, target)

        self.target_pos[env_ids] = target

        target_rpy = self.target_rpy_dist.sample((n, 1)).to(self.device)
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)

        # target visual in world frame: env offset + target (env frame)
        self.target_vis.set_world_poses(
            positions=target + self.envs_positions[env_ids].unsqueeze(1),
            orientations=target_rot,
            env_indices=env_ids
        )

        # ---- [M2] sample fresh obstacle layout for this window (env frame) & move prims ----
        if self._has_obstacles:
            n_active = self.curriculum_levels[self.level_idx] if self.curriculum_levels else 0
            pos_l, rad_l, act_l = self.obstacles.sample_layout(pos, target, n_active)
            self.obstacles.commit_layout(env_ids, pos_l, rad_l, act_l)
            wpos = self.obstacles.world_pose_tensor(env_ids, self.envs_positions)
            self.obstacle_views.set_world_poses(positions=wpos, env_indices=env_ids)
            # reset per-life / per-window collision bookkeeping
            self.ep_collision_edges[env_ids] = 0
            self.life_collision_edges[env_ids] = 0
            self.prev_in_collision[env_ids] = False

        # reset bookkeeping
        self.arrive_timer[env_ids] = 0
        self.arrival_triggered[env_ids] = False
        self.episode_any_arrival[env_ids] = False
        self.life_steps[env_ids] = 0
        # PBRS baseline = initial distance (per-life tracking starts clean)
        self.prev_goal_dist[env_ids] = torch.norm(target - pos, dim=-1)
        self.stats[env_ids] = 0.
        self.term_cause[env_ids] = 0.
        if self._has_obstacles:
            self.stats["curriculum_level"][env_ids] = (
                self.curriculum_levels[self.level_idx] if self.curriculum_levels else 0.0)

        # init prev_action to a hover thrust cmd (for controller Transforms)
        self.info[env_ids] = 0.
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info["prev_action"][env_ids, :, 3] = cmd_init.mean(-1)
        self.prev_actions[env_ids] = self.info["prev_action"][env_ids].clone()

    def _respawn(self, env_ids: torch.Tensor):
        # [M1 2026-09-04] soft reset on crash/out-of-bound/NaN: teleport to a fresh
        # random init pose but KEEP the current target & episode progress. This lets
        # the episode run up to max_episode_length (arrival reachable) while the
        # agent keeps getting "one more life" toward the same goal.
        n = len(env_ids)
        if n == 0:
            return
        # same ordering as _reset_idx: reset the drone/articulation view FIRST so the
        # physics buffers are consistent before we teleport mid-episode.
        self.drone._reset_idx(env_ids, self.training)
        self._sync_low_level_controller()
        pos = self._sample_init_pos(n)
        # [M2] obstacles of this window are static: rejection-sample the respawn pose so
        # a fresh life never starts inside/next to an obstacle (layout itself is kept).
        if self._has_obstacles and self.obstacles is not None:
            for _ in range(12):
                clr = self.obstacles.clearances_for(env_ids, pos)       # (n,K)
                dmin = clr.min(dim=-1, keepdim=True).values             # (n,1)
                bad = (torch.isfinite(dmin) & (dmin < self.obstacle_collision_margin + 0.05))
                bad = bad.squeeze(-1)                                   # (n,)
                if not bad.any():
                    break
                bidx = bad.nonzero().squeeze(-1)
                if bidx.numel() == 0:
                    break
                pos[bidx] = self._sample_init_pos(bidx.numel())
        rpy = self.init_rpy_dist.sample((n, 1)).to(self.device)
        rot = euler_to_quaternion(rpy)
        poses = pos + self.envs_positions[env_ids].unsqueeze(1)
        self.drone.set_world_poses(
            poses, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.arrive_timer[env_ids] = 0
        self.arrival_triggered[env_ids] = False
        # rebase per-life trackers: the teleport must not look like "goal progress"
        self.life_steps[env_ids] = 0
        self.prev_goal_dist[env_ids] = torch.norm(self.target_pos[env_ids] - pos, dim=-1)
        # [M2] fresh life resets obstacle contact trackers too
        if self._has_obstacles:
            self.life_collision_edges[env_ids] = 0
            self.prev_in_collision[env_ids] = False

    def _sync_low_level_controller(self):
        """[B1 2026-09-09] push the drone's current per-env RANDOMIZED dynamics into the
        low-level controller that the velocity action-transform uses. Called right after
        drone._reset_idx() (episode reset and mid-episode respawn), so the low-level
        controller acts like a real Crazyflie firmware that self-calibrates to its true
        mass/thrust. No-op unless the script mounted a controller AND controller_sync_dr
        is on (default off -> exactly the pre-B1 nominal-controller behavior).
        """
        if not self.controller_sync_dr or self.low_level_controller is None:
            return
        ctl = self.low_level_controller
        if not hasattr(ctl, "sync_randomized"):
            return
        n_rot = self.drone.num_rotors
        mass = self.drone.masses.reshape(-1, 1)        # (N, 1)
        kf = self.drone.KF.reshape(-1, n_rot)          # (N, num_rotors)
        ctl.sync_randomized(mass=mass, kf=kf)
        # [B1 debug] print once so we can confirm randomization actually reached the
        #   low-level controller (spread over envs > 0 only when DR eval is on).
        if not self._b1_synced:
            self._b1_synced = True
            print(f"[NavVel B1] low-level controller synced: mass "
                  f"mean={mass.mean().item():.5f} std={mass.std().item():.5f} "
                  f"[{mass.min().item():.5f},{mass.max().item():.5f}] | "
                  f"KF mean={kf.mean().item():.6f} (ctrl={type(ctl).__name__})", flush=True)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # capture what controller Transform wrote (prev_action always,
        # policy_action only written by PIDRateController)
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        if ("info", "policy_action") in tensordict.keys(True, True):
            self.info["policy_action"] = tensordict[("info", "policy_action")]
        # [P3 2026-09-16] p_filter / CBF 诊断通道（仅 train 入口 emit_info=True 时存在）
        if ("info", "p_filter") in tensordict.keys(True, True):
            self.info["p_filter"] = tensordict[("info", "p_filter")]
        if ("info", "cbf_corr") in tensordict.keys(True, True):
            self.info["cbf_corr"] = tensordict[("info", "cbf_corr")]
        if ("info", "cbf_intervened") in tensordict.keys(True, True):
            self.info["cbf_intervened"] = tensordict[("info", "cbf_intervened")]
        self.prev_actions = self.info["prev_action"].clone()
        # [NR-1] cache previous policy velocity-cmd for action-layer smoothness.
        #   info.policy_action 由 CBFVelocityFilter 写 = 滤波前策略 4D 指令 (含 yaw)。
        #   非 CBF 臂(无滤波)不写 -> policy_action 恒 0 -> adiff≈0, 平滑项自然失效(可接受)。
        if not hasattr(self, "policy_actions") or self.policy_actions is None:
            self.policy_actions = self.info["policy_action"].clone()
        self.prev_policy_actions = self.policy_actions
        self.policy_actions = self.info["policy_action"].clone()
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()   # env frame by default
        self.info["drone_state"][:] = self.drone_state[..., :13]

        # relative position & heading (translation invariant)
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        obs = [self.rpos, self.drone_state[..., 3:], self.rheading]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        # [M2] obstacle block appended last (first 30 dims stay identical to M1)
        if self._has_obstacles:
            drone_pos = self.drone_state[..., :3]
            self._obs_block = self.obstacles.build_obs(drone_pos)
            self._obs_dmin = self.obstacles.min_clearance(drone_pos)   # (N,1)
            self.obstacles.update_min_clearance(drone_pos)
            obs.append(self._obs_block)
            # [New2 2026-09-07] obs 结构级 internalize: CBF 边界余量 h / min_clearance 标量
            # 通道 (new2_plan.md §2)。h = min_i(||p-p_oi||-r_si^cbf) = dmin - cbf_extra
            # (cbf_extra 为标量; 无 CBF/naive -> 0, 退化为 min_clearance)。纯几何量训练/部署
            # 同源、不依赖 filter 在线; cbf_margin 的 0 穿越点 = CBF filter 介入边界 → 策略
            # 在 obs 层学会"看到 h 收紧就提前减速/绕行"(介入前预测性特征, 治 New1 事后罚稀释)。
            # 通道须 unsqueeze(1) 成 (N,1,1) 与 cat(dim=-1) 的 3D obs 对齐(标量别保持 (N,1))。
            if self.obs_safety_dim > 0:
                obs.extend(safety_obs_channels(
                    self._obs_dmin,
                    0.0 if self.cbf_extra is None else float(self.cbf_extra),
                    self.obs_safety_norm,
                    self.obs_safety_add_clearance,
                    self.obs_safety_add_cbf_margin,
                ))
            # [M2-3] per-slot CBF ball (center + CBF radius) fed to CBFVelocityFilter
            # before VelController each step. Inactive slots zeroed (pos=0, r_cbf=0) so
            # the filter's "r_cbf>0" activation excludes them (fix 2026-09-07).
            if self.cbf_extra is not None:
                actm = self.obstacles.active.float()                  # (N,K)
                pos_use = self.obstacles.pos * actm.unsqueeze(-1)     # inactive -> 0
                rcbf = (self.obstacles.r_safe + self.cbf_extra) * actm
                self.info["obstacle_cbf"][:] = torch.cat(
                    [pos_use, rcbf.unsqueeze(-1)], dim=-1).unsqueeze(1)
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "info": self.info,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        pos_error = torch.norm(self.rpos, dim=-1)                       # (num_envs,1)
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)
        distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)

        reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        # --- [M1] arrival detection: stay within radius for hold_steps ---
        inside = pos_error < self.arrive_radius
        self.arrive_timer = torch.where(
            inside, self.arrive_timer + 1, torch.zeros_like(self.arrive_timer)
        )
        just_arrived = (self.arrive_timer >= self.arrive_hold_steps) & (~self.arrival_triggered)
        self.arrival_triggered |= just_arrived
        self.episode_any_arrival |= just_arrived    # whole-episode flag (kept across respawns)
        reward_arrival = self.arrive_bonus * just_arrived.float()
        if self.reward_scheme in ("navgoal", "r1", "f1"):
            # [NR-1] time-scaled arrival: 越快越多 (+w_t*(1 - t_arr/T)), T=max_episode_length
            if self.arrive_time_bonus > 0:
                t_arr = self.progress_buf.to(pos_error.dtype).unsqueeze(-1)      # (N,1)
                frac = (1.0 - t_arr / self.max_episode_length).clamp(min=0.0)
                reward_arrival = reward_arrival + self.arrive_time_bonus * frac * just_arrived.float()
            # [NR-1] window first-arrival step (0 = never arrived); 验收"时间短"用.
            fa = torch.where(just_arrived.bool(),
                             self.progress_buf.to(pos_error.dtype).unsqueeze(-1),
                             self.stats["first_arrival_step"])
            self.stats["first_arrival_step"][:] = fa

        # [M1 / guide §1.2-3] PBRS goal-progress (optional, off by default):
        #   w * (d_{t-1} - gamma * d_t), rebased after spawn/respawn inside _respawn/_reset_idx.
        if self.reward_pbrs_weight > 0:
            reward_pbrs = self.reward_pbrs_weight * (
                self.prev_goal_dist - self.pbrs_gamma * pos_error
            )
        else:
            reward_pbrs = torch.zeros_like(pos_error)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        if self.reward_scheme == "navgoal":
            # [NR-1] motion-based base (论文式): progress 势差(悬停=0; γ 由 CLI pbrs_gamma,
            #   建议 1.0) + time-scaled arrival。移除 Hover 常驻 pose/up/spin/effort 正基座
            #   (new_reward.md §3)。⑥⑦ 障碍/CBF 项随后叠加。
            reward = reward_pbrs + reward_arrival
        elif self.reward_scheme == "r1":
            # [New1/R1] motion-gated 接近引导: 悬停(v≈0)=0、远离/切向=0、只有"正朝目标运动"
            #   才按接近势给正且越近越强 -> 保留 legacy pose 的稠密引导(可 from-scratch 学)但
            #   不重建"悬停也刷正"平台 (new1_plan.md R1)。w_g 可大(不构成平台)。
            #   r = w_g * [relu(v·u_g)/v_max]_≤1 * 1/(1+(k·||rpos||)²) + arrival(time-scaled)
            if self.reward_gate_weight > 0:
                # 单位化必须用 rpos 自身 norm(keepdim) (N,1,1) 除；不能除 pos_error (N,1)
                # 否则 (N,1,3)/(N,1) 会广播出 (N,N,3)。且对 3D (N,1,3) 做 reduce 时不要
                # keepdim（否则得 (N,1,1)，与 2D reward/stats 相乘会灾难广播 (N,N,1)）。
                u_goal = self.rpos / self.rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (N,1,3)
                v_lin = self.drone_state[..., 7:10]                          # (N,1,3) 平动速度
                appro = torch.relu((v_lin * u_goal).sum(dim=-1))             # (N,1) 朝目标速率(2D)
                appro = (appro / self.max_vel).clamp(max=1.0)
                pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * pos_error))
                reward = self.reward_gate_weight * appro * pose + reward_arrival
            else:
                reward = reward_arrival
        elif self.reward_scheme == "f1":
            # [New1/F1] 无核飞行主导 (用户 2026-09-06 反馈"起飞吸引力不够"): r1 失败因接近势核
            #   1/(1+(k·d)²) 在 spawn 距离 d≈3.6 处把信号压到 ~3% 峰值 -> 远端=无引导。
            #   f1 去掉距离核 -> ① 纯运动项无衰减(远端=近端同强)且是速度指令直接函数(action-local),
            #   悬停/远离=0、正朝目标运动每步按 relu(v·u_g)/vmax 给正。
            #   r = w_f * [relu(v·u_g)/v_max]_≤1 + arrival(time-scaled); 平滑④/⑥⑦随后叠加。
            #   形状坑同 r1: u_goal 用 rpos.norm(dim=-1,keepdim=True) 除(N,1,1); reduce 无 keepdim
            #   得 2D (N,1) 与 reward/stats 对齐。
            if self.reward_fly_weight > 0:
                u_goal = self.rpos / self.rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (N,1,3)
                v_lin = self.drone_state[..., 7:10]                          # (N,1,3) 平动速度
                fly = torch.relu((v_lin * u_goal).sum(dim=-1))               # (N,1) 朝目标速率(2D)
                fly = (fly / self.max_vel).clamp(max=1.0)
                if self.reward_zone_weight > 0:
                    # [2026-09-08] 圈内距离引导(d<arrive_radius): 替换 fly 项为
                    #   r_zone = w_f + k_in*(1 - d/R), 边界 d=R⁻ -> w_f(与圈外前进上限连续,
                    #   无进圈悬崖), 中心 d→0 -> w_f+k_in, 随 d↓ 严格单调增;
                    #   到达保持(hold_steps)后另 +arrive_bonus(大稀疏) 并 success_terminate 结算。
                    inside = pos_error < self.arrive_radius                 # (N,1) bool
                    frac = 1.0 - (pos_error / self.arrive_radius).clamp(max=1.0)   # (N,1) 0..1
                    zone = self.reward_fly_weight + self.reward_zone_weight * frac   # (N,1)
                    base = torch.where(inside, zone, self.reward_fly_weight * fly)
                    reward = base + reward_arrival
                else:
                    reward = self.reward_fly_weight * fly + reward_arrival
            else:
                reward = reward_arrival
            # [2026-09-08 PBRS-in-f1] 距离势差加入 f1: 给 μ(mean) 一个全程"净接近才得分"梯度
            #   (悬停/远离=0 → 无悬停平台; 远端近端同效 → 无历史远端死区), 治"确定性 mean 学不会
            #   精确入圈/停稳保持" (M1 legacy+PBRS 曾 eval_ckpt 确定性 0.99; 现 f1 一直 reward_pbrs_weight=0)。
            #   仅 reward_pbrs_weight>0 时生效(默认 0=逐位不变); prev_goal_dist 已在 reset/respawn rebase。
            if self.reward_pbrs_weight > 0:
                reward = reward + reward_pbrs
            # [F1-② 可选] per-step 时间成本(治 loiter/时间短/推完成): r -= c_t 每步;
            #   默认 0=关(仅 f1 生效, legacy/r1 不受影响)。reward 为 2D (N,1), 减标量广播安全。
            if self.reward_time_cost > 0:
                reward = reward - self.reward_time_cost
        else:
            # [legacy] Hover 常驻版（默认, 逐位不变）
            reward = (
                reward_pose
                + reward_pose * (reward_up + reward_spin)
                + reward_effort
                + reward_action_smoothness
                + reward_arrival
                + reward_pbrs
            )
        if self.reward_scheme in ("navgoal", "r1", "f1"):
            # [NR-1/R1] action-layer smoothness（速度指令差分, 负惩罚）:
            #   -w_s*||a_t - a_{t-1}||^2; a = policy_action(滤波前 4D 速度+yaw 指令).
            #   注意 pa/pp 为 3D (N,1,4) -> norm(dim=-1, keepdim=False) 得 2D (N,1)，
            #   与 reward/stats (N,1) 对齐（keepdim=True 会给 (N,1,1) 引发错误广播）。
            adiff = (self.policy_actions - self.prev_policy_actions).norm(dim=-1)
            if self.reward_action_smoothness_weight > 0:
                reward = reward - self.reward_action_smoothness_weight * adiff.square()
            # mean_action_diff EMA (显式 in-place, 避免 Tensor.lerp_ 的 broadcast 语义)
            m = self.stats["mean_action_diff"]
            m[:] = self.alpha * m + (1.0 - self.alpha) * adiff.detach().float()

        # [M1 2026-09-04] survival penalty: falling below z_ref (diagnosed free-fall).
        # r -= lambda * max(0, z_ref - z): mild, only bites while the drone is low.
        if self.survival_penalty_weight > 0:
            z = self.drone_state[..., 2]
            reward = reward - self.survival_penalty_weight * torch.clamp(
                self.z_ref - z, min=0.0
            )

        # ---- [M2 2026-09-04] obstacle rewards / collision edge / respawn cause ----
        if self._has_obstacles:
            drone_pos = self.drone_state[..., :3]
            clr = self.obstacles.clearances(drone_pos)              # (N,K), inf=inactive
            dmin = clr.min(dim=-1, keepdim=True).values             # (N,1)
            finite = torch.isfinite(dmin)
            in_col = (dmin < self.obstacle_collision_margin) & finite
            new_edge = in_col & (~self.prev_in_collision)
            self.prev_in_collision = in_col.clone()                 # respawn resets to False
            self.life_collision_edges += new_edge.long()
            self.ep_collision_edges += new_edge.long()

            # 1) obstacle log-distance penalty over active slots in the danger zone
            if self.reward_obs_log_weight > 0:
                D = self.obstacle_danger_radius
                d = clr
                phi = torch.zeros_like(d)
                band = (d > 0) & (d <= D)                           # 0 < d <= D
                phi = torch.where(band, torch.log(d.clamp_min(1e-3) / D), phi)
                neg = d <= 0                                        # inside the ball
                phi = torch.where(neg, torch.log(torch.tensor(1e-3 / D, device=self.device))
                                  + 100.0 * d, phi)
                _log_term = self.reward_obs_log_weight * self.reward_obs_log_scale \
                    * phi.sum(dim=-1, keepdim=True)
                # [P3 2026-09-16 用户裁决] φ = ln(d/D) ≤ 0（危险区内）⇒ `+φ` 即"越近越负"的惩罚。
                if self.reward_obs_log_mode == "penalty":
                    _log_contrib = _log_term
                else:
                    # legacy：与 A0–A4 冻结口径逐位一致（危险区内实为 +奖励，历史符号错误）
                    _log_contrib = -_log_term
                reward = reward + _log_contrib
                self.stats["term_obs_log"].lerp_(_log_contrib.detach(), (1 - self.alpha))

            # 2) one-shot collision-edge penalty (only on new contacts)
            _edge_contrib = -self.reward_collision_edge * new_edge.float()
            reward = reward + _edge_contrib
            self.stats["term_collision_edge"].lerp_(_edge_contrib.detach(), (1 - self.alpha))

            # 3) near-obstacle slowdown penalty (optional, default on small)
            if self.reward_near_slowdown_weight > 0:
                v_lin = self.drone_state[..., 7:10]                 # (N,1,3) lin velocity
                active_any = self.obstacles.active.any(-1, keepdim=True)   # (N,1)
                _, min_idx = clr.min(dim=-1)                        # (N,)
                pc = self.obstacles.pos.gather(
                    1, min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)
                ).squeeze(1)                                        # (N,3) nearest center
                u = torch.nn.functional.normalize(
                    pc - drone_pos.squeeze(1), dim=-1)              # drone -> obstacle
                v_par = torch.relu((v_lin.squeeze(1) * u).sum(-1, keepdim=True))
                slow = self.reward_near_slowdown_weight * v_par * torch.clamp(
                    1.0 - dmin / self.obstacle_danger_radius, min=0.0)
                slow = slow * (finite & active_any & (dmin < self.obstacle_danger_radius))
                reward = reward - slow
                self.stats["term_near_slow"].lerp_((-slow).detach(), (1 - self.alpha))

            # 4) a life that accumulates max_collisions contacts soft-respawns (kept as a
            #    separate cause, outside crash/oob, so the drone gets another life)
            self.collide_exceed = self.life_collision_edges >= self.obstacle_max_collisions
            if self.obstacle_reward_early_death_weight > 0:
                scale = (self.obstacle_early_death_threshold
                         / self.life_steps.clamp(min=1)).clamp(1.0, 10.0)
                reward = reward - self.obstacle_reward_early_death_weight \
                    * scale * (new_edge & self.collide_exceed).float()

            # [M2-3] CBF1 reward core + violation stat. Uses the PRE-filter policy command
            # (v_nom recorded by CBFVelocityFilter into info.policy_action), so the policy
            # learns that the penalized command is its own output (soft-CBF, CBF-RL).
            if self.cbf_extra is not None:
                # [fix 2026-09-07] vnom 限幅到 ±v_max(与 VelController 施加一致; 无界高斯采样可给
                #   ±1000 → viol 爆炸); act 用 active(勿用 r_safe>0: inactive r_safe=0.2>0 会把
                #   16 个 (0,0,0) 哨兵当激活)。
                vnom = self.policy_actions[..., :3].clamp(-self.max_vel, self.max_vel)
                rcbf = self.obstacles.r_safe + self.cbf_extra          # (N,K)
                act = self.obstacles.active                            # (N,K)
                viol = cbf_violation(
                    drone_pos, vnom,
                    self.obstacles.pos.unsqueeze(1), rcbf.unsqueeze(1),
                    act.unsqueeze(1),
                    self.cbf_alpha, self.cbf_penalty_intrude)          # (N,1) <= 0
                self.stats["cbf_violation"].lerp_(-viol, (1 - self.alpha))
                # [P3 2026-09-16] 分项仪器初值（下面各分支填实际贡献）
                _term_viol = torch.zeros_like(viol)
                _term_corr = torch.zeros_like(viol)
                _term_hpen = torch.zeros_like(viol)
                # viol <= 0 (more negative = more unsafe): ADD it (weighted) so the unsafe
                # command is PENALIZED. Do NOT subtract (that would reward violations --
                # bug found 2026-09-05: return ballooned to ~1e4 while policy learned to
                # deliberately aim at obstacles; the filter kept physics safe so collision
                # stayed ~0 but the "penalty" was actually a bonus).
                if self.cbf_reward_weight > 0 and self.cbf_use_reward_core:
                    # [M2 2026-09-05 / M3-A 2026-09-06] penalty_src 选择"罚什么"(只对
                    # 有滤波的臂有区分意义; 默认 nominal 与旧行为逐位一致, 不动基线):
                    #   nominal    = 罚原始指令 v_nom 的 CBF 违反量 (soft-CBF 原版;
                    #               hybrid λ0.05 长训后期 arrival 停滞的机制: 保守策略
                    #               学会"远离被罚区"而非"安全穿过").
                    #   correction = 罚滤波器"实际纠偏量" ||v_filtered − v_nom||(线性):
                    #               滤波没拦你(安全通过)就不罚, 拦得越多罚越多 → 逼策略
                    #               学"让滤波无话可说"的安全近穿, 消除保守绕行偏置。
                    #   gaussian  = correction 的论文式平滑高斯核 (CBF-RL Table II
                    #               r_cbf 第二项): pen = λ·(1 − exp(−corr²/σ²)) ∈ [0, λ),
                    #               0 纠偏不罚、小纠偏几乎无感、大纠偏饱和到 λ —— 比线性
                    #               λ·corr 稳(大纠偏不无限放大), σ 标定"判为显著纠偏"的
                    #               尺度(默认 0.5 ≈ 0.28·v_max)。缓解强 shaping 下
                    #               "correction 罚大拽回 hybrid" 的过罚问题。
                    #   dual      = [M3-A] CBF-RL 论文式"两项相加" r_cbf (Eq.22+23):
                    #               w1·viol(v_nom) + w2·(1 − exp(−corr²/σ²)), nominal viol
                    #               项与 gaussian correction 项同时启用 → 训练中滤波让
                    #               策略 internalize 安全, 部署可免 runtime filter(Dual 无
                    #               rt.filter 92.7% vs Filter-only 38.7%, 论文 Table I)。
                    #               w1 = reward_weight, w2 = correction_weight(默认 0)。
                    #   要求 mode=filter_only/hybrid (纠偏来自滤波本身); reward_only
                    #   (do_filter=False, 无纠偏) 自动回退 nominal。
                    if self.cbf_penalty_src == "dual":
                        # [M3-A] viol 项 w1·viol(任何模式都可用, vnom 即策略输出)
                        _term_viol = self.cbf_reward_weight * viol
                        reward = reward + _term_viol
                        # [M3-A] correction 项 w2·(1−exp(−corr²/σ²)): 需真实滤波才有
                        # v_safe≠vnom; 无滤波时无纠偏 → dual 退化为仅 w1·viol 项。
                        if self.cbf_use_filter and self.cbf_correction_weight > 0:
                            v_safe, _ = filter_velocity(
                                drone_pos, vnom,
                                self.obstacles.pos.unsqueeze(1), rcbf.unsqueeze(1),
                                act.unsqueeze(1), self.cbf_alpha, self.cbf_iterations)
                            corr = (v_safe - vnom).norm(dim=-1)      # (N,1) >= 0
                            sig2 = self.cbf_correction_sigma ** 2
                            pen = self.cbf_correction_weight * (1.0 - torch.exp(-(corr ** 2) / sig2))
                            _term_corr = -pen
                            reward = reward + _term_corr
                    elif self.cbf_penalty_src in ("correction", "gaussian") and self.cbf_use_filter:
                        v_safe, _ = filter_velocity(
                            drone_pos, vnom,
                            self.obstacles.pos.unsqueeze(1), rcbf.unsqueeze(1),
                            act.unsqueeze(1), self.cbf_alpha, self.cbf_iterations)
                        corr = (v_safe - vnom).norm(dim=-1)          # (N,1) >= 0
                        if self.cbf_penalty_src == "gaussian":
                            sig2 = self.cbf_correction_sigma ** 2
                            pen = self.cbf_reward_weight * (1.0 - torch.exp(-(corr ** 2) / sig2))
                            _term_corr = -pen
                            reward = reward + _term_corr
                        else:
                            _term_corr = -self.cbf_reward_weight * corr
                            reward = reward + _term_corr
                    else:
                        _term_viol = self.cbf_reward_weight * viol
                        reward = reward + _term_viol

                # [New2/E1 2026-09-07] CBF 边界余量罚(独立于 penalty_src, 任何有 CBF 的模式可用):
                #   h = dmin - cbf_extra(0 穿越点 = filter 介入边界)。罚 w_h*relu(buffer-h) → 策略
                #   保持 h>=buffer 的梯度。buffer=0 即 E1 原版 relu(-h)(只罚已进 filter 决策区, fire
                #   少); buffer>0 = E1-v2 提前在接近边界前罚(类 CBF 版 near_slowdown/soft wall)。
                #   配合 obs_safety=cbf_margin 通道给"守边界"直接信号(new2_plan.md §6 E1/E1-v2)。
                #   dmin 为几何表面净空(obstacle 块上方已算; 无活动障碍=inf → 罚=0)。
                #   默认 w_h=0 = 逐位不变。
                if self.cbf_h_penalty_weight > 0:
                    _term_hpen = -h_boundary_penalty(
                        dmin, float(self.cbf_extra),
                        self.cbf_h_penalty_buffer, self.cbf_h_penalty_weight)
                    reward = reward + _term_hpen
                # [P3 2026-09-16] 分项仪器落账（EMA；只记录，供标定与符号核查）
                self.stats["term_cbf_viol"].lerp_(_term_viol.detach(), (1 - self.alpha))
                self.stats["term_cbf_corr"].lerp_(_term_corr.detach(), (1 - self.alpha))
                self.stats["term_h_pen"].lerp_(_term_hpen.detach(), (1 - self.alpha))

        # --- per-life / termination bookkeeping (spatial bounds, env frame) ---
        self.life_steps += 1
        xyz = self.drone_state[..., :3]
        out_of_xy = torch.norm(xyz[..., :2], dim=-1) > self.bound_xy
        # [env_design 2026-09-07] 箱式房间越界(arena_bound 给 x/y 半宽, 6×6×3): 飞出房间算失败
        if self.arena_bound_x is not None:
            abs_xy = xyz[..., :2].abs()
            out_of_xy = out_of_xy | (abs_xy[..., 0] > self.arena_bound_x) \
                | (abs_xy[..., 1] > self.arena_bound_y)
        crash = (xyz[..., 2] < self.z_min) | torch.isnan(self.drone_state).any(-1)
        oob = out_of_xy | (xyz[..., 2] > self.z_max)
        misbehave = crash | oob

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # [M1 / guide §6.2+§6.4] one-shot failure penalties (stage-0 convergence fix):
        #   crash / oob flat penalties + optional early-death amplifier (kaiwu style).
        if (self.reward_crash_penalty > 0 or self.reward_oob_penalty > 0
                or self.reward_early_death_weight > 0):
            base = torch.zeros_like(reward)
            if self.reward_crash_penalty > 0:
                base = base - self.reward_crash_penalty * crash.float()
            if self.reward_oob_penalty > 0:
                base = base - self.reward_oob_penalty * oob.float()
            if self.reward_early_death_weight > 0:
                died = crash | oob
                scale = (self.early_death_threshold_steps / self.life_steps.clamp(min=1)).clamp(1.0, 10.0)
                base = base - self.reward_early_death_weight * scale * died.float()
            base = base * (~truncated).float()   # don't double-count with timeout on the last step
            reward = reward + base

        # rebase PBRS baseline to this step's distance (respawned envs overridden below)
        self.prev_goal_dist[:] = pos_error

        # [M1 / guide §6.4] timeout penalty: episode ended (600) without ever arriving.
        if self.reward_timeout_penalty > 0:
            reward = reward - self.reward_timeout_penalty * (
                truncated & (~self.episode_any_arrival)
            ).float()

        # [M1 2026-09-04] soft-respawn: instead of terminating on crash/out-of-bound,
        # give the drone "another life" toward the SAME target (see _respawn). Episodes
        # then last up to max_episode_length so arrival becomes reachable and long-horizon
        # signal exists (diagnosed: drones fell long before any learning could happen).
        # [M2] collision-exceed is another (soft) respawn cause when obstacles are on
        collide_death = torch.zeros_like(misbehave)
        if self._has_obstacles:
            collide_death = self.collide_exceed & (~truncated)
        if self.soft_respawn:
            terminated = torch.zeros_like(misbehave)
            # keep masks 1-D for the boolean op to avoid (N,N) broadcast
            respawn = ((misbehave | collide_death).squeeze(-1) & ~truncated.squeeze(-1))
            # [Arena1-B 2026-09-07] 每次 soft-respawn 一次性惩罚(丢一条命 -> 扣一次):
            #   挂在 respawn 判定之后、_respawn 之前, 只作用于本步真正要重生的 env。
            #   reward 为 2D (N,1), respawn 为 1D (N,) -> unsqueeze 广播。默认 0=关。
            if self.reward_respawn_penalty > 0:
                reward = reward - self.reward_respawn_penalty * respawn.unsqueeze(-1).float()
            ids = respawn.nonzero().squeeze(-1)
            if ids.numel() > 0:
                self._respawn(ids)
        else:
            terminated = misbehave | collide_death
        if self.success_terminate:
            terminated = terminated | just_arrived

        # [env_design 2026-09-07] 记录本步结束 env 的终止原因(仅单命; eval auto_reset=false 时跨步保留):
        #   1=crash 2=oob 3=collide 4=truncated未到(timeout) 5=truncated已到. 供 eval_ckpt 归因诊断.
        if not self.soft_respawn:
            done_cause = (terminated | truncated).squeeze(-1)
            cause = torch.zeros_like(misbehave, dtype=torch.long)
            cause = torch.where(crash, torch.ones_like(cause), cause)                       # 1 crash
            cause = torch.where(oob & ~crash, torch.full_like(cause, 2), cause)             # 2 oob
            cause = torch.where(collide_death & ~misbehave, torch.full_like(cause, 3), cause)  # 3 collide
            cause = torch.where(truncated & ~self.episode_any_arrival.bool(),
                                torch.full_like(cause, 4), cause)                          # 4 timeout
            cause = torch.where(truncated & self.episode_any_arrival.bool(),
                                torch.full_like(cause, 5), cause)                          # 5 arrived@end
            self.term_cause = torch.where(done_cause.unsqueeze(-1), cause, self.term_cause)

        # stats (EMA)
        # [P3 2026-09-16] p_filter 当步值（train 入口 emit_info=True 时由 filter 写入；
        #   未挂 filter 的臂恒 0 = "不执行滤波"）
        self.stats["p_filter"].lerp_(
            self.info["p_filter"].detach().float().reshape(self.num_envs, 1),
            (1 - self.alpha))
        self.stats["pos_error"].lerp_(pos_error, (1 - self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1 - self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
        # [2026-09-08 语义变更] arrival = "曾到达∧保持 arrive_hold_steps 步"(episode_any_arrival 持存,
        #   reset 清 0), 不再是"瞬时在圈内 EMA"(后者 train ~0.99 假象, 与确定性 eval 严重不符)。
        #   EpisodeStats 在 done 步采样 -> train log 的 stats.arrival = 已完成 episode 的真实到达率。
        self.stats["arrival"][:] = self.episode_any_arrival.float()
        self.stats["vel_norm"].lerp_(torch.norm(self.drone.vel[..., :3], dim=-1), (1 - self.alpha))
        if self._has_obstacles:
            # sampled by EpisodeStats at the window-end step -> reflects whole window
            self.stats["collision"].lerp_(in_col.float(), (1 - self.alpha))
            self.stats["collision_episodes"][:] = (self.ep_collision_edges > 0).float()
            cap = self.obstacle_obs_dist_norm
            clr_cap = dmin.clamp(max=cap)
            clr_cap = torch.where(torch.isfinite(dmin), clr_cap,
                                  torch.full_like(clr_cap, cap))
            self.stats["min_clearance"].lerp_(clr_cap, (1 - self.alpha))
            self.stats["success_rate"][:] = (
                self.episode_any_arrival & (self.ep_collision_edges == 0)).float()
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
