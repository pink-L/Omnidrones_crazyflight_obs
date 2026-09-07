# MIT License
#
# Copyright (c) 2026
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
[M2, 2026-09-04] Self-driven obstacle curriculum scheduler (env-internal, pure logic).

NavVel obstacle course: fixed obstacle-count stages ``levels = [0, 2, 4, 8]``.
All envs share a single global *level*; the scheduler consumes *episode* outcomes
(a whole 600-step window, since NavVel soft-respawns and only resets on truncation)
and promotes the level once the rolling success rate clears the gate for enough
environment-frames.

Why env-internal / pure torch (see navvel_obstacle_migration_guide.md §7.1):
  - The OmniDrones fork PPO loop (torchrl SyncDataCollector) never touches
    env/task params; keeping the state machine in the env means train.py/play.py/
    eval_ckpt.py need zero changes and ``level`` is just reported through stats.
  - Pure logic (no isaac import) -> unit-testable on CPU in milliseconds.

Success semantics (m2_plan.md §4.3 / §9-6, soft-respawn aware):
  success(episode) = reached arrival at least once during the window
                     AND zero collision edges over the whole window (across respawns).
  collided(episode) = one or more collision edges over the window.
"""

import torch


class ObstacleCurriculum:
    """Rolling-window, success-rate gated stage scheduler.

    Usage (from ``nav_vel._reset_idx``, once per episode/reset boundary):
        if self.training and previous window actually ran:
            cur.update(success_mask)          # success_mask: (n,) bool episode outcomes
            cur.add_frames(n_completed_windows * episode_len)
            level_idx = cur.level_idx         # read back the (possibly advanced) level
    """

    def __init__(self, levels, initial_level, gate_window, success_threshold,
                 collision_threshold, min_frames, allow_demote, device="cuda:0",
                 margin_gate=False, margin_frac=0.5):
        self.levels = list(levels)
        self.level_idx = int(initial_level)
        if not (0 <= self.level_idx < len(self.levels)):
            raise ValueError(f"initial_level {initial_level} out of range {levels}")
        self.gate_window = int(gate_window)
        self.success_threshold = float(success_threshold)
        self.collision_threshold = float(collision_threshold)
        self.min_frames = float(min_frames)
        self.allow_demote = bool(allow_demote)
        # [New2/E3 2026-09-07] margin gate (new2_plan.md §6 E3): 提升除了 success/collision
        # 门槛外, 还要求窗口滚动 margin_ok_rate >= margin_frac。margin_ok = 该窗口全程保持
        # 表面净空 >= margin_clearance(默认 0.1 = brake-off cbf_extra) → 等价于 h>=0 = 从未
        # 进入 CBF filter 介入区 = internalize 语义(策略必须守边界, 密度才升)。默认关(逐位兼容)。
        self.margin_gate = bool(margin_gate)
        self.margin_frac = float(margin_frac)

        # rolling window of episode outcomes (filled oldest-first circular buffer)
        self._window = torch.zeros(self.gate_window, dtype=torch.bool, device=device)
        self._window_edges = torch.zeros(self.gate_window, dtype=torch.float32, device=device)
        self._window_margin = torch.zeros(self.gate_window, dtype=torch.float32, device=device)
        self._n_filled = 0
        self._head = 0                      # index of next slot to overwrite
        self._successes = 0
        self._collided = 0
        self._margin_ok = 0
        self._frames = 0.0                  # env-steps since last promotion
        self._device = device

    # ------------------------------------------------------------------ accessors
    @property
    def level(self) -> int:
        """Number of obstacles for the current global level."""
        return int(self.levels[self.level_idx])

    @property
    def level_idx(self) -> int:
        return self._level_idx

    @level_idx.setter
    def level_idx(self, value):
        self._level_idx = int(value)

    @property
    def active_obstacles(self) -> int:
        return int(self.levels[self._level_idx])

    @property
    def frames(self) -> float:
        return self._frames

    @property
    def success_rate(self) -> float:
        """Success rate over the filled part of the rolling window (empty -> 0)."""
        if self._n_filled == 0:
            return 0.0
        return self._successes / self._n_filled

    @property
    def collision_rate(self) -> float:
        if self._n_filled == 0:
            return 1.0
        return self._collided / self._n_filled

    @property
    def margin_ok_rate(self) -> float:
        """[New2/E3] 窗口保持净空(margin_ok)比例; 空窗 -> 0(margin gate 下不提升)。"""
        if self._n_filled == 0:
            return 0.0
        return self._margin_ok / self._n_filled

    # ------------------------------------------------------------------ mutators
    def add_frames(self, n_env_steps: float):
        self._frames += float(n_env_steps)

    def update(self, success: torch.Tensor, collided: torch.Tensor = None,
               margin_ok: torch.Tensor = None, add_frames: float = 0.0):
        """Consume a batch of completed episodes and (optionally) advance the level.

        Args:
            success:  (n,) bool tensor, True if the episode window succeeded.
            collided: (n,) bool tensor (optional), True if the window had any collision.
            margin_ok: (n,) bool tensor (optional, [New2/E3] 仅 margin_gate=True 时必填):
                       True if the window kept surface clearance >= margin_clearance
                       (never entered the CBF filter intervention zone, h>=0).
            add_frames: env-steps these episodes contributed (len * n usually).
        """
        success = success.reshape(-1).to(self._device)
        n = success.numel()
        if n == 0:
            return
        if collided is None:
            collided = ~success
        else:
            collided = collided.reshape(-1).to(self._device)
        if self.margin_gate:
            if margin_ok is None:
                raise ValueError("margin_gate=True requires margin_ok tensor in update()")
            margin_ok = margin_ok.reshape(-1).to(self._device)
        else:
            margin_ok = None
        self.add_frames(add_frames)

        # write into circular buffer (supports n > window gracefully by folding)
        if n >= self.gate_window:
            # keep only the most recent gate_window outcomes
            success = success[-self.gate_window:]
            collided = collided[-self.gate_window:]
            if margin_ok is not None:
                margin_ok = margin_ok[-self.gate_window:]
            n = self.gate_window
        start = self._head
        end = start + n
        if end <= self.gate_window:
            self._window[start:end] = success
            self._window_edges[start:end] = collided.float()
            if margin_ok is not None:
                self._window_margin[start:end] = margin_ok.float()
        else:
            first = self.gate_window - start
            self._window[start:] = success[:first]
            self._window[:end - self.gate_window] = success[first:]
            self._window_edges[start:] = collided[:first].float()
            self._window_edges[:end - self.gate_window] = collided[first:].float()
            if margin_ok is not None:
                self._window_margin[start:] = margin_ok[:first].float()
                self._window_margin[:end - self.gate_window] = margin_ok[first:].float()

        self._head = end % self.gate_window
        self._n_filled = min(self.gate_window, self._n_filled + n)
        # recompute counts only over the filled span (cheap at this frequency)
        span = self._window[:self._n_filled]
        edge_span = self._window_edges[:self._n_filled]
        self._successes = int(span.sum().item())
        self._collided = int((edge_span > 0).sum().item())
        if margin_ok is not None:
            self._margin_ok = int(self._window_margin[:self._n_filled].sum().item())
        else:
            self._margin_ok = self._successes

        self._maybe_advance()

    def _maybe_advance(self):
        """Promote (or optionally demote) the global level."""
        max_idx = len(self.levels) - 1
        if self._n_filled == 0:
            return
        gate_ok = (self.success_rate >= self.success_threshold
                   and self.collision_rate <= self.collision_threshold)
        if self.margin_gate:
            gate_ok = gate_ok and self.margin_ok_rate >= self.margin_frac
        if (gate_ok
                and self._frames >= self.min_frames
                and self._level_idx < max_idx):
            self._level_idx += 1
            self._frames = 0.0
            self._window.zero_()
            self._window_edges.zero_()
            self._window_margin.zero_()
            self._n_filled = 0
            self._successes = 0
            self._collided = 0
            self._margin_ok = 0
        elif (self.allow_demote
              and self._frames >= 10 * self.min_frames
              and self.success_rate < 0.3 * self.success_threshold
              and self._level_idx > 0):
            self._level_idx -= 1
            self._frames = 0.0
            self._window.zero_()
            self._window_edges.zero_()
            self._window_margin.zero_()
            self._n_filled = 0
            self._successes = 0
            self._collided = 0
            self._margin_ok = 0

    def reset(self):
        self._window.zero_()
        self._window_edges.zero_()
        self._window_margin.zero_()
        self._n_filled = 0
        self._head = 0
        self._successes = 0
        self._collided = 0
        self._margin_ok = 0
        self._frames = 0.0
