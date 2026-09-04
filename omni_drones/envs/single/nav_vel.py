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
from omni_drones.views import ArticulationView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from omni_drones.utils.torchrl.compat import CompositeSpec, UnboundedContinuousTensorSpec


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

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
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
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

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

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "arrival": UnboundedContinuousTensorSpec(1),   # [M1] EMA of within-radius ratio
            "vel_norm": UnboundedContinuousTensorSpec(1),  # [M1] EMA of speed
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        # info keys consumed by controller Transforms
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "policy_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
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

        # reset bookkeeping
        self.arrive_timer[env_ids] = 0
        self.arrival_triggered[env_ids] = False
        self.stats[env_ids] = 0.

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
        rpy = self.init_rpy_dist.sample((n, 1)).to(self.device)
        rot = euler_to_quaternion(rpy)
        poses = pos + self.envs_positions[env_ids].unsqueeze(1)
        self.drone.set_world_poses(
            poses, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.arrive_timer[env_ids] = 0
        self.arrival_triggered[env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # capture what controller Transform wrote (prev_action always,
        # policy_action only written by PIDRateController)
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        if ("info", "policy_action") in tensordict.keys(True, True):
            self.info["policy_action"] = tensordict[("info", "policy_action")]
        self.prev_actions = self.info["prev_action"].clone()
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
        reward_arrival = self.arrive_bonus * just_arrived.float()

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
            + reward_arrival
        )

        # [M1 2026-09-04] survival penalty: falling below z_ref (diagnosed free-fall).
        # r -= lambda * max(0, z_ref - z): mild, only bites while the drone is low.
        if self.survival_penalty_weight > 0:
            z = self.drone_state[..., 2]
            reward = reward - self.survival_penalty_weight * torch.clamp(
                self.z_ref - z, min=0.0
            )

        # --- [M1] termination: spatial bounds (env frame), NOT fixed-point distance ---
        xyz = self.drone_state[..., :3]
        out_of_xy = torch.norm(xyz[..., :2], dim=-1) > self.bound_xy
        out_of_z = (xyz[..., 2] < self.z_min) | (xyz[..., 2] > self.z_max)
        hasnan = torch.isnan(self.drone_state).any(-1)
        misbehave = out_of_xy | out_of_z | hasnan

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # [M1 2026-09-04] soft-respawn: instead of terminating on crash/out-of-bound,
        # give the drone "another life" toward the SAME target (see _respawn). Episodes
        # then last up to max_episode_length so arrival becomes reachable and long-horizon
        # signal exists (diagnosed: drones fell long before any learning could happen).
        if self.soft_respawn:
            terminated = torch.zeros_like(misbehave)
            # keep masks 1-D for the boolean op to avoid (N,N) broadcast
            respawn = (misbehave.squeeze(-1) & ~truncated.squeeze(-1))
            ids = respawn.nonzero().squeeze(-1)
            if ids.numel() > 0:
                self._respawn(ids)
        else:
            terminated = misbehave
        if self.success_terminate:
            terminated = terminated | just_arrived

        # stats (EMA)
        self.stats["pos_error"].lerp_(pos_error, (1 - self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1 - self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1 - self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1 - self.alpha))
        self.stats["arrival"].lerp_(inside.float(), (1 - self.alpha))
        self.stats["vel_norm"].lerp_(torch.norm(self.drone.vel[..., :3], dim=-1), (1 - self.alpha))
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
