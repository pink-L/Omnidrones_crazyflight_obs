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
from omni_drones.envs.single.nav_vel_obstacles import ObstacleManager
from omni_drones.utils.nav_curriculum import ObstacleCurriculum
from omni_drones.utils.cbf import (
    cbf_violation,
    filter_velocity,
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
            self.M = int(max(int(oc.get("num_scene") or self.K), self.K))
            _ow = oc.get("obs_window")
            self.obs_window = bool(self.M > self.K) if _ow is None else bool(_ow)
            self.obstacle_phys_radius = float(max(oc.get("radius_choices", [0.30])))
            self.obstacle_collision_margin = float(oc.get("collision_margin", 0.05))
            self.obstacle_danger_radius = float(oc.get("danger_radius", 0.6))
            self.obstacle_max_collisions = int(oc.get("max_collisions", 2))
            self.reward_obs_log_weight = float(oc.get("reward_obs_log_weight", 1.5))
            self.reward_obs_log_scale = float(oc.get("reward_obs_log_scale", 0.3))
            self.reward_collision_edge = float(oc.get("reward_collision_edge", 2.0))
            self.reward_near_slowdown_weight = float(oc.get("reward_near_slowdown_weight", 0.5))
            self.obstacle_reward_early_death_weight = float(oc.get("reward_early_death_weight", 0.0))
            self.obstacle_early_death_threshold = float(oc.get("early_death_threshold_steps", 300))
            self.obstacle_obs_dist_norm = float(oc.get("obs_dist_norm", 5.0))
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

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

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

            cc = self._curriculum_cfg or {}
            self.curriculum = ObstacleCurriculum(
                levels=cc.get("levels", [0, 2, 4, 8]),
                initial_level=int(cc.get("initial_level", 0)),
                gate_window=int(cc.get("gate_window_episodes", 3000)),
                success_threshold=float(cc.get("success_rate_threshold", 0.8)),
                collision_threshold=float(cc.get("collision_rate_threshold", 0.05)),
                min_frames=float(cc.get("min_frames_between_promote", 2_000_000)),
                allow_demote=bool(cc.get("allow_demote", False)),
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
        self.target_pos = self.target_pos_dist.sample((self.num_envs, 1)).to(self.device)
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
            "arrival": UnboundedContinuousTensorSpec(1),   # [M1] EMA of within-radius ratio
            "vel_norm": UnboundedContinuousTensorSpec(1),  # [M1] EMA of speed
            # [M2] obstacle stats (EpisodeStats samples them at the episode end step)
            "collision": UnboundedContinuousTensorSpec(1),  # EMA frac. of steps in contact
            "collision_episodes": UnboundedContinuousTensorSpec(1),  # window had >=1 edge
            "min_clearance": UnboundedContinuousTensorSpec(1),  # EMA min surface clearance
            "success_rate": UnboundedContinuousTensorSpec(1),  # window success (arr&0 edge)
            "curriculum_level": UnboundedContinuousTensorSpec(1),  # active obstacle count
            "cbf_violation": UnboundedContinuousTensorSpec(1),  # [M2-3] CBF reward-core violation (>=0)
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
                self.curriculum.update(
                    success, collided,
                    add_frames=float(ran.sum().item()) * float(self.max_episode_length))
                self.level_idx = self.curriculum.level_idx

        self.drone._reset_idx(env_ids, self.training)

        n = len(env_ids)
        # --- random initial pose (env frame) ---
        pos = self.init_pos_dist.sample((n, 1)).to(self.device)
        rpy = self.init_rpy_dist.sample((n, 1)).to(self.device)
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        # --- random target waypoint (env frame), re-sample if too close to init ---
        target = self.target_pos_dist.sample((n, 1)).to(self.device)
        for _ in range(3):
            d = torch.norm(target - pos, dim=-1, keepdim=True)          # (n,1,1)
            too_close = d < self.min_init_target_dist
            if not too_close.any():
                break
            resample = self.target_pos_dist.sample((n, 1)).to(self.device)
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
        pos = self.init_pos_dist.sample((n, 1)).to(self.device)
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
                pos[bidx] = self.init_pos_dist.sample((bidx.numel(), 1)).to(self.device)
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

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # capture what controller Transform wrote (prev_action always,
        # policy_action only written by PIDRateController)
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        if ("info", "policy_action") in tensordict.keys(True, True):
            self.info["policy_action"] = tensordict[("info", "policy_action")]
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
            # [M2-3] per-slot CBF ball (center + CBF radius) fed to CBFVelocityFilter
            # before VelController each step. Inactive slots stay 0 (r_safe=0).
            if self.cbf_extra is not None:
                rcbf = self.obstacles.r_safe + self.cbf_extra          # (N,K)
                self.info["obstacle_cbf"][:] = torch.cat(
                    [self.obstacles.pos, rcbf.unsqueeze(-1)], dim=-1).unsqueeze(1)
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
                reward = self.reward_fly_weight * fly + reward_arrival
            else:
                reward = reward_arrival
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
                reward = reward - self.reward_obs_log_weight * self.reward_obs_log_scale \
                    * phi.sum(dim=-1, keepdim=True)

            # 2) one-shot collision-edge penalty (only on new contacts)
            reward = reward - self.reward_collision_edge * new_edge.float()

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
                vnom = self.policy_actions[..., :3]                    # (N,1,3) pre-filter
                rcbf = self.obstacles.r_safe + self.cbf_extra          # (N,K)
                act = self.obstacles.r_safe > 0                        # (N,K)
                viol = cbf_violation(
                    drone_pos, vnom,
                    self.obstacles.pos.unsqueeze(1), rcbf.unsqueeze(1),
                    act.unsqueeze(1),
                    self.cbf_alpha, self.cbf_penalty_intrude)          # (N,1) <= 0
                self.stats["cbf_violation"].lerp_(-viol, (1 - self.alpha))
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
                        reward = reward + self.cbf_reward_weight * viol
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
                            reward = reward - pen
                    elif self.cbf_penalty_src in ("correction", "gaussian") and self.cbf_use_filter:
                        v_safe, _ = filter_velocity(
                            drone_pos, vnom,
                            self.obstacles.pos.unsqueeze(1), rcbf.unsqueeze(1),
                            act.unsqueeze(1), self.cbf_alpha, self.cbf_iterations)
                        corr = (v_safe - vnom).norm(dim=-1)          # (N,1) >= 0
                        if self.cbf_penalty_src == "gaussian":
                            sig2 = self.cbf_correction_sigma ** 2
                            pen = self.cbf_reward_weight * (1.0 - torch.exp(-(corr ** 2) / sig2))
                            reward = reward - pen
                        else:
                            reward = reward - self.cbf_reward_weight * corr
                    else:
                        reward = reward + self.cbf_reward_weight * viol

        # --- per-life / termination bookkeeping (spatial bounds, env frame) ---
        self.life_steps += 1
        xyz = self.drone_state[..., :3]
        out_of_xy = torch.norm(xyz[..., :2], dim=-1) > self.bound_xy
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
            ids = respawn.nonzero().squeeze(-1)
            if ids.numel() > 0:
                self._respawn(ids)
        else:
            terminated = misbehave | collide_death
        if self.success_terminate:
            terminated = terminated | just_arrived

        # stats (EMA)
        self.stats["pos_error"].lerp_(pos_error, (1 - self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1 - self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
        self.stats["arrival"].lerp_(inside.float(), (1 - self.alpha))
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
