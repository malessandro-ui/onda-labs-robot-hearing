"""Bounded, fail-closed, asynchronous Onda CPU reference adapter.

A single producer (Isaac Lab policy thread) submits immutable scene poses.
A single dedicated worker owns the existing IsaacLabHearingManager and *all*
acoustic state. No Isaac Lab/Kit calls occur in the worker. No audio windows
are silently spliced together: queue overflow forces a fresh acoustic epoch.

This is a best-effort Python thread implementation, NOT hard real time. The
GIL and CUDA->CPU pose copies can still affect control-loop scheduling.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import perf_counter_ns
from collections.abc import Mapping
from typing import Any

import numpy as np

from onda_hearing.simulators.isaac_lab_runtime import IsaacLabHearingManager
from onda_hearing.simulators.isaac_lab_wrapper import IsaacLabSceneReader


@dataclass(frozen=True)
class AsyncHearingConfig:
    queue_capacity: int = 2
    # No valid hearing result older than this many *simulation* milliseconds.
    max_staleness_ms: float = 250.0
    max_wall_staleness_ms: float = 750.0
    fail_closed: bool = True

    def __post_init__(self):
        if type(self.queue_capacity) is not int or not 1 <= self.queue_capacity <= 64:
            raise ValueError("queue_capacity must be integer in [1,64]")
        if not np.isfinite(self.max_staleness_ms) or self.max_staleness_ms <= 0:
            raise ValueError("max_staleness_ms must be positive and finite")
        if not np.isfinite(self.max_wall_staleness_ms) or self.max_wall_staleness_ms <= 0:
            raise ValueError("max_wall_staleness_ms must be positive and finite")
        if type(self.fail_closed) is not bool:
            raise TypeError("fail_closed must be bool")


@dataclass(frozen=True)
class _Task:
    epoch: int
    seq: int
    # Coordinates are staged and made immutable on producer thread.
    p: np.ndarray
    q: np.ndarray
    sources: dict[str, np.ndarray]
    done: tuple[int, ...]
    resync: bool


class AsyncHearingWorker:
    """Threaded manager with explicit overload behavior and telemetry.

    submit() is bounded (no waiting for worker computation), but CPU staging
    from a GPU can synchronize. read() is O(num_envs), never calls Onda.
    A gap (including overflow) resets all acoustic histories, preserving no
    false continuity in rendered PCM/timestamps.
    """

    def __init__(self, manager: IsaacLabHearingManager, config: AsyncHearingConfig | None = None):
        if not isinstance(manager, IsaacLabHearingManager):
            raise TypeError("manager must be IsaacLabHearingManager")
        self.manager = manager
        self.config = config or AsyncHearingConfig()
        self.num_envs = manager.num_envs
        self._input: Queue[_Task | None] = Queue(maxsize=self.config.queue_capacity)
        self._lock = Lock()
        self._epoch = 0
        self._seq = 0
        self._submitted = 0
        self._processed = 0
        self._dropped = 0
        self._resyncs = 0
        self._worker_errors = 0
        self._last_error: BaseException | None = None
        self._closed = False
        self._features = np.zeros((self.num_envs, 5), dtype=np.float32)
        self._published_seq = -1
        self._published_mono_ns = 0
        self._last_worker_ms = 0.0
        self._last_pose_staging_ms = 0.0
        self._max_age_steps = max(1, int(self.config.max_staleness_ms * 1e6 / manager.config.step_dt_ns))
        self._thread = Thread(target=self._run, name="onda-acoustic-worker", daemon=True)
        self._thread.start()

    @staticmethod
    def _cpu(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        # Caller owns GPU-to-CPU synchronization and corresponding cost.
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").numpy()
        x = np.array(value, dtype=np.float64, order="C", copy=True)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f"{name} must be finite array of shape {shape}")
        x.setflags(write=False)
        return x

    def _make_task(self, poses, seq: int, epoch: int, done: tuple[int, ...], resync: bool) -> _Task:
        p, q, sources = poses
        p = self._cpu(p, (self.num_envs, 3), "array_positions")
        q = self._cpu(q, (self.num_envs, 4), "array_quaternions_wxyz")
        if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-5):
            raise ValueError("nonunit array quaternion")
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError("sources must be a nonempty mapping")
        src = {k: self._cpu(v, (self.num_envs, 3), f"source:{k}") for k,v in sources.items()}
        return _Task(epoch, seq, p, q, src, done, resync)

    def _ensure_healthy(self):
        if self._closed:
            raise RuntimeError("Onda asynchronous worker already closed")
        if self._last_error is not None:
            raise RuntimeError("Onda asynchronous worker failed") from self._last_error

    def _flush(self):
        while True:
            try:
                self._input.get_nowait()
            except Empty:
                break

    def _invalidate_locked(self):
        self._features.fill(0)
        self._published_seq = -1
        self._published_mono_ns = 0

    def submit(self, poses, *, done_env_ids=(), reset_all: bool = False) -> bool:
        """Submit one post-physics pose batch. Returns False if overload forced a gap.

        `done_env_ids` must correspond to environments that have *already*
        auto-reset in the original Isaac Lab env.step(), not pending resets.
        """
        ids = tuple(int(i) for i in done_env_ids)
        if len(ids) != len(set(ids)) or any(not 0 <= i < self.num_envs for i in ids):
            raise ValueError("invalid/duplicate done environment ID")
        with self._lock:
            self._ensure_healthy()
            self._seq += 1
            seq = self._seq
            if reset_all:
                self._epoch += 1
                self._flush()
                self._invalidate_locked()
                self._resyncs += 1
            epoch = self._epoch
        # CPU copy is done outside the lock; no worker touches GPU tensors.
        stage_start = perf_counter_ns()
        item = self._make_task(poses, seq, epoch, ids, reset_all)
        stage_ms = (perf_counter_ns() - stage_start)/1e6
        with self._lock:
            self._ensure_healthy()
            self._submitted += 1
            self._last_pose_staging_ms = stage_ms
            if epoch != self._epoch:
                # Another reset in the meantime. Conservatively discard.
                self._dropped += 1
                return False
            try:
                self._input.put_nowait(item)
                return True
            except Full:
                # Never splice an old PCM history to a new timestamp.
                self._dropped += self._input.qsize() + 1
                self._epoch += 1
                self._resyncs += 1
                self._flush()
                self._invalidate_locked()
                # Newly accepted frame seeds the worker's fresh acoustic epoch.
                newest = _Task(self._epoch, seq, item.p, item.q, item.sources, (), True)
                self._input.put_nowait(newest)
                return False

    def _run(self):
        active_epoch = -1
        previous_seq = -1
        while True:
            item = self._input.get()
            if item is None:
                return
            with self._lock:
                if self._closed:
                    return
                if item.epoch != self._epoch:
                    continue
            start = perf_counter_ns()
            try:
                poses = (item.p, item.q, item.sources)
                if item.resync or item.epoch != active_epoch or item.seq != previous_seq + 1:
                    self.manager.reset()
                    self.manager.seed(*poses)
                    active_epoch = item.epoch
                else:
                    reset_ids = item.done
                    if reset_ids:
                        self.manager.reset(reset_ids)
                    active_ids = [i for i in range(self.num_envs) if i not in reset_ids]
                    # Step token deliberately omitted: this worker enforces order.
                    self.manager.advance(*poses, active_env_ids=active_ids)
                    if reset_ids:
                        self.manager.seed(*poses, env_ids=reset_ids)
                features = self.manager.read()
                if features.shape != (self.num_envs, 5) or not np.isfinite(features).all():
                    raise ValueError("invalid Onda acoustic output")
                previous_seq = item.seq
                elapsed_ms = (perf_counter_ns() - start) / 1e6
                with self._lock:
                    # Invalidated epochs are never allowed to publish stale data.
                    if item.epoch == self._epoch and not self._closed:
                        self._features = features.copy()
                        self._published_seq = item.seq
                        self._published_mono_ns = perf_counter_ns()
                        self._processed += 1
                        self._last_worker_ms = elapsed_ms
            except BaseException as exc:
                with self._lock:
                    self._last_error = exc
                    self._worker_errors += 1
                    self._invalidate_locked()
                return

    def read(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Nonblocking, cached, fail-closed view for the robot policy."""
        with self._lock:
            self._ensure_healthy()
            features = self._features.copy()
            lag_steps = max(0, self._seq - self._published_seq) if self._published_seq >= 0 else None
            wall_age_ms = ((perf_counter_ns() - self._published_mono_ns)/1e6
                           if self._published_mono_ns else None)
            stale = (lag_steps is None or lag_steps > self._max_age_steps or
                     wall_age_ms is None or wall_age_ms > self.config.max_wall_staleness_ms)
            if stale and self.config.fail_closed:
                features.fill(0)
            elif lag_steps:
                # Feature 4 is age in seconds. Add pending simulation time.
                features[:, 4] += lag_steps * self.manager.config.step_dt_ns / 1e9
            data = {
                "latest_submitted_step": self._seq,
                "latest_completed_step": self._published_seq,
                "lag_steps": lag_steps,
                "stale": stale,
                "dropped_frames": self._dropped,
                "resynchronizations": self._resyncs,
                "submitted_frames": self._submitted,
                "processed_frames": self._processed,
                "worker_errors": self._worker_errors,
                "worker_ms_last": self._last_worker_ms,
                "pose_staging_ms_last": self._last_pose_staging_ms,
                "published_wall_age_ms": wall_age_ms,
                "queue_depth": self._input.qsize(),
                "epoch": self._epoch,
            }
            return features, data

    def close(self, *, timeout_s: float = 5.0) -> bool:
        """Stop accepting work and signal exit; returns whether worker joined."""
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._lock:
            self._closed = True
            self._invalidate_locked()
            self._flush()
            self._input.put_nowait(None)
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()


class OndaAsyncIsaacLabEnvWrapper:
    """Gym-style wrapper; SDK never modifies the original policy, actions or rewards.

    Asynchronous observations are intentionally separate from the existing
    `policy` group. This wrapper DOES NOT cause an RL policy to use hearing by
    itself. `onda_hearing_metadata` stays out of policy observations.
    """

    def __init__(self, env, manager: IsaacLabHearingManager,
                 poses: IsaacLabSceneReader, *, config: AsyncHearingConfig | None = None):
        if int(env.num_envs) != manager.num_envs:
            raise ValueError("environment/manager batch mismatch")
        self.env = env
        self.poses = poses
        self.worker = AsyncHearingWorker(manager, config)
        self.last_hearing_metadata: dict[str, Any] = {}

    def _augment(self, obs):
        if not isinstance(obs, Mapping) or "onda_hearing" in obs:
            raise ValueError("expected observation mapping without onda_hearing")
        import torch
        features, info = self.worker.read()
        self.last_hearing_metadata = info
        out = dict(obs)
        out["onda_hearing"] = torch.as_tensor(features, dtype=torch.float32, device=self.env.device)
        return out

    def reset(self, *a, **kw):
        obs, extra = self.env.reset(*a, **kw)
        self.worker.submit(self.poses.read(self.env), reset_all=True)
        return self._augment(obs), extra

    def step(self, actions):
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        import torch
        done = (torch.as_tensor(terminated, device="cpu") |
                torch.as_tensor(truncated, device="cpu")).numpy()
        if done.shape != (self.env.num_envs,):
            raise ValueError("invalid termination mask shape")
        done_ids = tuple(int(i) for i in np.flatnonzero(done))
        self.worker.submit(self.poses.read(self.env), done_env_ids=done_ids)
        return self._augment(obs), reward, terminated, truncated, extras

    def close(self):
        if not self.worker.close():
            raise RuntimeError("Acoustic worker did not terminate; do not close Kit yet")
        return self.env.close()

    def __getattr__(self, key):
        return getattr(self.env, key)
