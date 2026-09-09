# GUI 实飞演示脚本 (headless=false): 随机环境逐轮切换 + 可选多 checkpoint 轮换模型。
#
# 用法 (OmniDrones/scripts 下, NX GUI 会话, conda activate lz_env, export DISPLAY):
#   单个模型(随机障碍+随机起终点, 每轮 reset 换一批新随机场景, 无限轮):
#     python -u demo_fly.py task=NavVel algo=ppo headless=false wandb.mode=disabled \
#       +eval_points=train task.env.num_envs=4 task.action_transform=velocity \
#       task.obstacle.num_scene=16 'task.obstacle.spawn_xy_range=[[-3.0,-3.0],[3.0,3.0]]' \
#       'task.curriculum.levels=[8]' task.curriculum.enabled=false \
#       task.cbf.mode=filter_only task.cbf.use_brake_term=false +runtime_filter=true \
#       +per_ep_steps=600 +checkpoint=<ckpt.pt>
#   固定起终点(专注看障碍随机变化, 不随机起终点):
#       把 +eval_points=train 换成 task.fixed_init=[-2.8,0,0.5] task.fixed_target=[2.8,0,1.0]
#   多个同配置 checkpoint 逐轮自动轮换模型(如同一 arm 的不同训练轮次):
#       +ckpts=[ckpt_a.pt,ckpt_b.pt]  (会忽略单数 +checkpoint)
#   不同 arm(naive/dual/reward_only/filter_only)因 cbf.mode/levels/num_scene 配置不同,
#   需各自一条命令跑(见仓库日志模板)。跑够想停 -> Ctrl+C。# 视角:
#   +cam_mode=topdown  俯瞰整条走廊(默认, eye/look 可用 +topdown_eye=[...] +topdown_look=[...] 调)
#   +cam_mode=follow   相机跟随无人机斜后方
#   +trail=false       关轨迹拖影; +trail_len=N  轨迹保留最近 N 物理步 (默认 1500)
#   (俯瞰看绕障全貌时建议 num_envs=1 且用固定起终点)#
# 每轮(reset)后 env 自动重采样: 起终点(edge/固定) + 障碍布局(sample_layout),
# 终端同步打印 ep# / 模型 / arrival / collision / mean|rpos|, GUI 窗口实时可见。
import os
import logging
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

logging.getLogger("omni").setLevel(logging.ERROR)


class _CamTrack:
    """rollout per-step callback:
      - trail: 维护 env0 无人机 world-pos 历史, 每 interval 步用 DebugDraw 重画轨迹拖影;
      - mode=follow: 每 interval 步相机跟随无人机(斜后方);
        mode=topdown: 相机由外层设成俯瞰, 这里只画轨迹不碰相机;
      - verbose>0: 每 verbose 步打印位置(headless 诊断用)。
    """
    def __init__(self, base_env, mode="topdown", trail=True, trail_len=1500,
                 interval=3, verbose=0):
        from collections import deque
        self.base_env = base_env
        self.mode = mode
        self.trail = bool(trail)
        self.interval = max(1, int(interval))
        self.verbose = int(verbose)
        self.hist = deque(maxlen=int(trail_len))
        self._dd = getattr(base_env, "debug_draw", None)
        self.i = 0
        self._warned = False

    def _drone_pose(self):
        try:
            if hasattr(self.base_env, "drone") and self.base_env.drone is not None:
                p = self.base_env.drone.pos[0, 0]           # world (3,) env0
                if p is not None and p.shape[-1] >= 3:
                    return p.detach().cpu()
        except Exception:
            pass
        try:
            s = self.base_env.drone_state[..., :3].float()  # env frame fallback
            return s[0, 0].detach().cpu()
        except Exception:
            return None

    def _draw_trail(self):
        if not self.trail or self._dd is None or len(self.hist) < 2:
            return
        try:
            pts = torch.stack(list(self.hist)).float().cpu()   # (n,3) world
            self._dd.clear()
            self._dd.plot(pts, size=3.0, color=(0.2, 0.65, 1.0, 1.0))
        except Exception as e:
            if not self._warned:
                print(f"[demo] trail draw failed: {e}", flush=True)
                self._warned = True

    def _set_follow_cam(self):
        p = self._drone_pose()
        if p is None:
            return
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
            eye = [float(p[0]), float(p[1]) - 2.6, float(p[2]) + 1.7]
            tgt = [float(p[0]) + 1.0, float(p[1]), float(p[2])]
            set_camera_view(eye=eye, target=tgt)
        except Exception as e:
            if not self._warned:
                print(f"[demo] follow cam unavailable: {e}", flush=True)
                self._warned = True

    def __call__(self, env, *args):
        p = self._drone_pose()
        if p is not None:
            self.hist.append(p)
            if self.verbose and self.i % self.verbose == 0:
                print(f"[cam] t={self.i:4d} pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})", flush=True)
        if self.i % self.interval == 0:
            if self.mode == "follow":
                self._set_follow_cam()
            self._draw_trail()
        self.i += 1


@hydra.main(config_path=".", config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    try:
        OmegaConf.set_struct(cfg.task, False)
    except Exception:
        pass

    # 起终点模式: train=随机对侧(edge, 泛化演示) | fixed=固定穿越点(专注障碍随机)
    _ep = str(cfg.get("eval_points", "train"))
    if _ep == "fixed":
        print("[demo_fly] eval_points=fixed -> 固定起终点 (障碍布局仍每轮随机)")
    else:
        cfg.task.fixed_init = None
        cfg.task.fixed_target = None
        cfg.task.episode_sampler = "edge"
        print("[demo_fly] eval_points=train -> 随机起终点(edge 对侧) + 每轮随机障碍")

    # 模型来源: 多 ckpt 轮换优先, 否则单个 checkpoint
    ckpts = [str(c) for c in cfg.get("ckpts", [])]
    single = cfg.get("checkpoint", None)
    if not ckpts and single is not None:
        ckpts = [str(single)]
    assert ckpts, "must pass +checkpoint=<ckpt.pt> or +ckpts=[a.pt,b.pt,...]"
    tags = [os.path.basename(c) for c in ckpts]
    print(f"[demo_fly] {len(ckpts)} model(s): {tags}  (逐轮轮换)" if len(ckpts) > 1
          else f"[demo_fly] model: {tags[0]}")

    _rf = bool(cfg.get("runtime_filter", True))
    print(f"[demo_fly] runtime_filter={_rf} -> "
          + ("CBF velocity filter ON (deploy)" if _rf else "CBF filter DISABLED (internalize check)"))

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            vl = cfg.task.get("vel_limit", {})
            transforms.append(VelController(
                controller,
                max_vel=vl.get("max_vel", None),
                max_yaw_rate=vl.get("max_yaw_rate", None),
            ))
            from omni_drones.utils.cbf import build_cbf_filter
            if bool(cfg.get("runtime_filter", True)):
                cbf_filter = build_cbf_filter(cfg)
                if cbf_filter is not None:
                    transforms.append(cbf_filter)
                else:
                    print("[demo_fly] no CBF filter built (mode=none or config)")
            else:
                print("[demo_fly] runtime_filter=false -> CBF velocity filter DISABLED")
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transforms.append(PIDRateController(controller))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec, env.reward_spec,
        device=base_env.device,
    )

    env.eval()
    base_env.eval()

    # [2026-09-08] 播放倍速: GUI 下默认每个物理步都渲染 -> 墙钟被渲染拖慢(近似慢放)。
    # +render_every=N 让每 N 个物理步才渲染一帧: N 越大播放越快。
    #   估算: dt=0.01s, 渲染 1 帧墙钟约 0.3~1s -> 要 1x 正常速需 N≈30~100。
    #   N=15 起试 (约 0.15s sim/帧, 明显加速); N=100 约正常速; N>300 近乎快进。
    render_every = int(cfg.get("render_every", 0))      # 0 = 每步渲染(原默认)
    if render_every > 0 and not bool(cfg.headless):
        _rc = {"n": 0}
        _sub = max(1, int(base_env.substeps))
        def _should_render(substep):
            if substep != _sub - 1:            # 只在最后一个 substep 决定
                return False
            _rc["n"] += 1
            return (_rc["n"] % render_every) == 0
        base_env.enable_render(_should_render)
        print(f"[demo_fly] render_every={render_every} -> 每 {render_every} 物理步渲染 1 帧 (加速播放)",
              flush=True)

    # 每轮(episode) 步数上限, 默认 = max_episode_length(1000):
    #   - 无人机坠/出界/碰撞(done)  -> 该轮立即结束 -> reset 换新随机场景
    #   - 到达后保持 -> episode 到 timeout(1000 步) 才 done -> 届时换新场景
    # 每轮结束即 env.reset() 重采样障碍+起终点, 避免无人机停在某个终态干等长轮次。
    per_ep = int(cfg.get("per_ep_steps", int(base_env.max_episode_length)))
    break_on_done = bool(cfg.get("break_on_done", True))
    n_env = base_env.num_envs
    print(f"[demo_fly] {n_env} envs, <= {per_ep} steps/episode (break_on_done={break_on_done}), obstacle level="
          f"{base_env.level_idx} "
          f"({base_env.curriculum_levels[base_env.level_idx] if base_env.curriculum_levels else 0} obs), "
          f"GUI window open. Ctrl+C to stop.")

    ep = 0
    _last_ck = object()                       # 哨兵: 保证首轮一定 load
    try:
        while True:
            _idx = ep % len(ckpts)
            tag = tags[_idx]
            if ckpts[_idx] != _last_ck:
                policy.load_state_dict(torch.load(ckpts[_idx], map_location=cfg.sim.device))
                print(f">>> episode {ep}: loaded model -> {tag}", flush=True)
                _last_ck = ckpts[_idx]
            td = env.reset()                          # 新随机障碍 + 起终点
            cam_mode = str(cfg.get("cam_mode", "topdown"))
            use_trail = bool(cfg.get("trail", True))
            if cam_mode == "topdown":
                try:
                    from omni.isaac.core.utils.viewports import set_camera_view
                    eye = [float(v) for v in cfg.get("topdown_eye", [0.2, -7.0, 6.2])]
                    look = [float(v) for v in cfg.get("topdown_look", [0.0, 0.0, 0.6])]
                    set_camera_view(eye=eye, target=look)
                except Exception as e:
                    print(f"[demo] topdown camera failed: {e}", flush=True)
            cam = _CamTrack(base_env,
                            mode=cam_mode,
                            trail=use_trail,
                            trail_len=int(cfg.get("trail_len", 1500)),
                            interval=max(int(cfg.get("cam_interval", 3)),
                                         render_every if render_every > 0 else 3),
                            verbose=int(cfg.get("verbose_progress", 0)))
            with set_exploration_type(ExplorationType.MODE):
                env.rollout(max_steps=per_ep, policy=policy, tensordict=td,
                            auto_reset=False,
                            break_when_any_done=break_on_done,
                            callback=cam)

            # 本轮摘要
            arr = base_env.arrival_triggered.float().squeeze(-1) if hasattr(base_env, "arrival_triggered") else None
            coll = base_env.ep_collision_edges.float().squeeze(-1) if hasattr(base_env, "ep_collision_edges") else None
            rpos = torch.norm(base_env.rpos.float(), dim=-1).squeeze(-1) if hasattr(base_env, "rpos") else None
            line = f"[ep {ep:3d}] {tag:24s}"
            if arr is not None:
                line += f"  arrival={arr.sum().item():2d}/{n_env}"
            if coll is not None:
                line += f"  collided={coll.gt(0).sum().item():2d}/{n_env}"
            if rpos is not None:
                line += f"  mean|rpos|={rpos.mean().item():.3f}"
            print(line, flush=True)
            ep += 1
    except KeyboardInterrupt:
        print("\n[demo_fly] stopped by user.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
