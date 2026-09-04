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
                 collision_threshold, min_frames, allow_demote, device="cuda:0"):
        self.levels = list(levels)
        self.level_idx = int(initial_level)
        if not (0 <= self.level_idx < len(self.levels)):
            raise ValueError(f"initial_level {initial_level} out of range {levels}")
        self.gate_window = int(gate_window)
        self.success_threshold = float(success_threshold)
        self.collision_threshold = float(collision_threshold)
        self.min_frames = float(min_frames)
        self.allow_demote = bool(allow_demote)

        # rolling window of episode outcomes (filled oldest-first circular buffer)
        self._window = torch.zeros(self.gate_window, dtype=torch.bool, device=device)
        self._window_edges = torch.zeros(self.gate_window, dtype=torch.float32, device=device)
        self._n_filled = 0
        self._head = 0                      # index of next slot to overwrite
        self._successes = 0
        self._collided = 0
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

    # ------------------------------------------------------------------ mutators
    def add_frames(self, n_env_steps: float):
        self._frames += float(n_env_steps)

    def update(self, success: torch.Tensor, collided: torch.Tensor = None,
               add_frames: float = 0.0):
        """Consume a batch of completed episodes and (optionally) advance the level.

        Args:
            success:  (n,) bool tensor, True if the episode window succeeded.
            collided: (n,) bool tensor (optional), True if the window had any collision.
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
        self.add_frames(add_frames)

        # write into circular buffer (supports n > window gracefully by folding)
        if n >= self.gate_window:
            # keep only the most recent gate_window outcomes
            success = success[-self.gate_window:]
            collided = collided[-self.gate_window:]
            n = self.gate_window
        start = self._head
        end = start + n
        if end <= self.gate_window:
            self._window[start:end] = success
            self._window_edges[start:end] = collided.float()
        else:
            first = self.gate_window - start
            self._window[start:] = success[:first]
            self._window[:end - self.gate_window] = success[first:]
            self._window_edges[start:] = collided[:first].float()
            self._window_edges[:end - self.gate_window] = collided[first:].float()

        self._head = end % self.gate_window
        self._n_filled = min(self.gate_window, self._n_filled + n)
        # recompute counts only over the filled span (cheap at this frequency)
        span = self._window[:self._n_filled]
        edge_span = self._window_edges[:self._n_filled]
        self._successes = int(span.sum().item())
        self._collided = int((edge_span > 0).sum().item())

        self._maybe_advance()

    def _maybe_advance(self):
        """Promote (or optionally demote) the global level."""
        max_idx = len(self.levels) - 1
        if self._n_filled == 0:
            return
        if (self.success_rate >= self.success_threshold
                and self.collision_rate <= self.collision_threshold
                and self._frames >= self.min_frames
                and self._level_idx < max_idx):
            self._level_idx += 1
            self._frames = 0.0
            self._window.zero_()
            self._window_edges.zero_()
            self._n_filled = 0
            self._successes = 0
            self._collided = 0
        elif (self.allow_demote
              and self._frames >= 10 * self.min_frames
              and self.success_rate < 0.3 * self.success_threshold
              and self._level_idx > 0):
            self._level_idx -= 1
            self._frames = 0.0
            self._window.zero_()
            self._window_edges.zero_()
            self._n_filled = 0
            self._successes = 0
            self._collided = 0

    def reset(self):
        self._window.zero_()
        self._window_edges.zero_()
        self._n_filled = 0
        self._head = 0
        self._successes = 0
        self._collided = 0
        self._frames = 0.0
