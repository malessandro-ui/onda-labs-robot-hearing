"""Isaac Lab -> Onda windowed PCM perception -> cached policy features.

No Isaac Lab, Kit, or torch import at module import time. This is a bounded
CPU reference backend: each simulation environment owns its own audio clock,
window buffer, selector, and policy encoder. An Isaac Lab application owns the
physics step, robot actions, resets, rewards, and scene sources.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from onda_hearing.policy import WindowedHearingPipeline, POLICY_FEATURES
from onda_hearing.runtime import SceneSnapshot


def _numpy(value, *, name: str) -> np.ndarray:
    """Copy from a batched torch/numpy input. Convert GPU arrays once per field."""
    if hasattr(value, 'detach'):
        value = value.detach().to('cpu').numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains non-finite values')
    return array


def _env_ids(env_ids, num_envs: int) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(num_envs))
    if isinstance(env_ids, slice):
        return tuple(range(num_envs)[env_ids])
    if hasattr(env_ids, 'detach'):
        env_ids = env_ids.detach().cpu().numpy()
    if isinstance(env_ids, np.ndarray):
        env_ids = env_ids.tolist()
    if not isinstance(env_ids, (tuple, list, range)):
        raise TypeError('env_ids must be a sequence, slice or tensor')
    values = tuple(env_ids)
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
           or not 0 <= i < num_envs for i in values):
        raise IndexError('invalid environment ID')
    if len(set(map(int, values))) != len(values):
        raise ValueError('duplicate environment ID')
    return tuple(map(int, values))


@dataclass(frozen=True, slots=True)
class IsaacLabAcousticConfig:
    num_envs: int
    step_dt_ns: int
    max_cpu_envs: int = 16

    def __post_init__(self):
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError('num_envs must be positive int')
        if type(self.step_dt_ns) is not int or not 0 < self.step_dt_ns <= 10**9:
            raise ValueError('step_dt_ns must be positive int <= 1 second')
        if type(self.max_cpu_envs) is not int or self.max_cpu_envs < 1:
            raise ValueError('max_cpu_envs must be positive int')
        if self.num_envs > self.max_cpu_envs:
            raise ValueError('CPU windowed backend capacity exceeded; use explicit profiling or GPU backend')


class IsaacLabHearingManager:
    """Batched scene input, independent environment clocks, read-only features.

    On reset, `reset(env_ids)` invalidates observations and clears each
    pipeline. The next `advance` initializes that environment at local t=0;
    audio begins on the following call. Alternatively use `seed` after the
    simulator has finished resetting physics, so the next advance produces audio.

    Call `advance` exactly once after each *policy* step (not each observation
    term evaluation). Individual episode clocks advance by step_dt_ns, never
    global simulation time, so partial resets never cause a timestamp rewind.
    """
    def __init__(self, config: IsaacLabAcousticConfig,
                 pipeline_factory: Callable[[int], WindowedHearingPipeline]):
        if not isinstance(config, IsaacLabAcousticConfig):
            raise TypeError('config must be IsaacLabAcousticConfig')
        if not callable(pipeline_factory):
            raise TypeError('pipeline_factory must be callable')
        self.config = config
        self.num_envs = config.num_envs
        self.pipelines = []
        seen = set()
        for i in range(self.num_envs):
            pipeline = pipeline_factory(i)
            if not isinstance(pipeline, WindowedHearingPipeline):
                raise TypeError('factory must return WindowedHearingPipeline')
            if id(pipeline) in seen:
                raise ValueError('each environment requires an independent pipeline')
            seen.add(id(pipeline))
            if i and (pipeline.runtime.array.channels != self.pipelines[0].runtime.array.channels
                      or pipeline.runtime.config.sample_rate_hz != self.pipelines[0].runtime.config.sample_rate_hz):
                raise ValueError('inconsistent audio configuration between environments')
            self.pipelines.append(pipeline)
        self._ticks = np.zeros(self.num_envs, dtype=np.int64)
        self._initialized = np.zeros(self.num_envs, dtype=bool)
        self._features = np.zeros((self.num_envs, len(POLICY_FEATURES)), dtype=np.float32)
        self._last_frames = [None] * self.num_envs
        self._step_token = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return POLICY_FEATURES

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._step_token = None
        for i in _env_ids(env_ids, self.num_envs):
            self.pipelines[i].reset()
            self._ticks[i] = 0
            self._initialized[i] = False
            self._features[i].fill(0)
            self._last_frames[i] = None

    def _validate_batch(self, array_positions, array_quaternions_wxyz, source_positions):
        p = _numpy(array_positions, name='array_positions')
        q = _numpy(array_quaternions_wxyz, name='array_quaternions_wxyz')
        if p.shape != (self.num_envs, 3) or q.shape != (self.num_envs, 4):
            raise ValueError('array positions must be [N,3], rotations [N,4] WXYZ')
        norms = np.linalg.norm(q, axis=1)
        if not np.allclose(norms, 1, atol=1e-5):
            raise ValueError('array quaternions must be unit WXYZ')
        if not isinstance(source_positions, Mapping):
            raise TypeError('source_positions must map source IDs to [N,3] world positions')
        source_names = set(source_positions)
        for pipeline in self.pipelines:
            if source_names != {src.source_id for src in pipeline.runtime.sources}:
                raise ValueError('scene sources must match each pipeline scheduled sources')
        sources = {}
        for key, value in source_positions.items():
            if not isinstance(key, str) or not key:
                raise ValueError('source names must be nonempty strings')
            source = _numpy(value, name=f'source:{key}')
            if source.shape != (self.num_envs, 3):
                raise ValueError(f'source:{key} must be [N,3]')
            sources[key] = source
        return p, q, sources

    @staticmethod
    def _snapshot(index, time_ns, p, q, sources):
        # Isaac Lab exposes scalar-first quaternion WXYZ. Onda uses XYZW.
        w, x, y, z = map(float, q[index])
        return SceneSnapshot(int(time_ns), tuple(map(float, p[index])),
                             (x, y, z, w),
                             {name: tuple(map(float, value[index]))
                              for name, value in sources.items()})

    def seed(self, array_positions, array_quaternions_wxyz, source_positions,
             env_ids=None):
        """Establish zero-time poses for reset environments after physics reset."""
        indices = _env_ids(env_ids, self.num_envs)
        p, q, sources = self._validate_batch(array_positions, array_quaternions_wxyz,
                                             source_positions)
        for i in indices:
            if self._initialized[i] or self._ticks[i] != 0:
                raise ValueError('seed requires a freshly reset environment')
            frame = self.pipelines[i].advance(self._snapshot(i, 0, p, q, sources))
            self._features[i] = frame.features
            self._last_frames[i] = frame
            self._initialized[i] = True

    def advance(self, array_positions, array_quaternions_wxyz, source_positions,
                *, active_env_ids=None, step_token: int | None = None):
        """Advance selected environments once, after physics, before policy reads.

        `step_token` may be a monotonically increasing global policy-step ID;
        duplicate values are rejected to prevent duplicate acoustic PCM.
        """
        if step_token is not None:
            if type(step_token) is not int or step_token < 0:
                raise ValueError('step_token must be a nonnegative int')
            if self._step_token is not None and step_token <= self._step_token:
                raise ValueError('duplicate/out-of-order global policy step')
        indices = _env_ids(active_env_ids, self.num_envs)
        p, q, sources = self._validate_batch(array_positions, array_quaternions_wxyz,
                                             source_positions)
        # Complete all input validation before changing any clock.
        snapshots = []
        for i in indices:
            ns = (int(self._ticks[i]) + (1 if self._initialized[i] else 0)) * self.config.step_dt_ns
            snapshots.append((i, self._snapshot(i, ns, p, q, sources)))
        for i, snapshot in snapshots:
            frame = self.pipelines[i].advance(snapshot)
            if self._initialized[i]:
                self._ticks[i] += 1
            else:
                self._initialized[i] = True
            self._features[i] = frame.features
            self._last_frames[i] = frame
        if step_token is not None:
            self._step_token = step_token
        return self.read()

    def read(self) -> np.ndarray:
        """Return an independent (N,5) float32 copy; never run perception."""
        return self._features.copy()

    def torch(self, *, device=None):
        import torch
        return torch.as_tensor(self.read(), dtype=torch.float32, device=device)

    def frames(self):
        """Last PolicyFrame for each environment, for diagnostics only."""
        return tuple(self._last_frames)


def onda_acoustic_policy_observation(env, *, manager_attribute='onda_hearing_manager'):
    """Isaac Lab ObservationTermCfg-compatible, intentionally READ-ONLY.

    Requires `env.onda_hearing_manager` to be attached before evaluation.
    Call `manager.advance` via your explicit post-physics hook, NOT here.
    """
    manager = getattr(env, manager_attribute)
    if not isinstance(manager, IsaacLabHearingManager):
        raise TypeError('invalid Onda hearing manager')
    if int(env.num_envs) != manager.num_envs:
        raise ValueError('Isaac Lab batch size does not match Onda')
    return manager.torch(device=env.device)


def onda_reset_hearing(env, env_ids=None, *, manager_attribute='onda_hearing_manager'):
    """Isaac Lab EventTermCfg-compatible reset callback.

    Use only when manager has been attached before the event executes.
    For existing tasks, prefer the explicit wrapper's reset handling.
    """
    manager = getattr(env, manager_attribute)
    manager.reset(env_ids)
