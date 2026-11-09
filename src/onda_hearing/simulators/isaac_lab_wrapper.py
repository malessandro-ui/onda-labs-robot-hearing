"""Optional integration for existing Isaac Lab vectorized RL environments.

The wrapper executes the ORIGINAL task, rewards, terminations and robot policy.
It appends a separate `onda_hearing` observation tensor AFTER the original env's
step. The caller can concatenate it with other observations in its policy.
This does not alter the original task's ObservationManager configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .isaac_lab_runtime import IsaacLabHearingManager


@dataclass(frozen=True, slots=True)
class IsaacLabPoseRef:
    asset: str
    body_index: int | None = None

    def __post_init__(self):
        if not isinstance(self.asset, str) or not self.asset:
            raise ValueError('asset must be a nonempty scene key')
        if self.body_index is not None and (type(self.body_index) is not int or self.body_index < 0):
            raise ValueError('body_index must be nonnegative')


class IsaacLabSceneReader:
    """Read root/body world poses from Isaac Lab scene asset.data tensors.

    `body_index` is for articulated body pose arrays; it is not a USD prim index.
    A microphone array mounted off the body origin should use a dedicated
    rigid-body frame or an upstream calibrated pose transform.
    """
    def __init__(self, *, array: IsaacLabPoseRef,
                 sources: Mapping[str, IsaacLabPoseRef]):
        if not isinstance(array, IsaacLabPoseRef):
            raise TypeError('array must be IsaacLabPoseRef')
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError('sources must be a nonempty mapping')
        if any(not isinstance(k, str) or not k or not isinstance(v, IsaacLabPoseRef)
               for k, v in sources.items()):
            raise ValueError('source IDs and pose refs are invalid')
        self.array = array
        self.sources = dict(sources)

    @staticmethod
    def _pose(scene, ref):
        try:
            asset = scene[ref.asset]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(f'missing Isaac Lab scene asset: {ref.asset}') from exc
        data = asset.data
        if ref.body_index is None:
            return data.root_pos_w, data.root_quat_w
        if not hasattr(data, 'body_pos_w') or not hasattr(data, 'body_quat_w'):
            raise ValueError(f'{ref.asset} has no body poses')
        if ref.body_index >= data.body_pos_w.shape[1]:
            raise ValueError(f'body_index out of range for {ref.asset}')
        return data.body_pos_w[:, ref.body_index], data.body_quat_w[:, ref.body_index]

    def read(self, env):
        pos, quat = self._pose(env.scene, self.array)
        sources = {key: self._pose(env.scene, ref)[0]
                   for key, ref in self.sources.items()}
        return pos, quat, sources


class OndaIsaacLabEnvWrapper:
    """Wrap an existing Isaac Lab task without modifying its policy or rewards.

    The task must return gym-style (obs, reward, terminated, truncated, extras)
    from step and (obs, info) from reset. The wrapper adds a *separate* key
    `onda_hearing` to the observation mapping. Existing `policy` tensors are
    preserved. Use a custom policy wrapper to concatenate/select features.

    For resets triggered INSIDE the Isaac Lab step, terminated/truncated envs
    are reset acoustically and seeded from their new poses, avoiding audio
    history leakage across episodes. Non-reset environments are advanced.
    """
    def __init__(self, env, manager: IsaacLabHearingManager,
                 poses: IsaacLabSceneReader, *, key: str = 'onda_hearing'):
        if not isinstance(manager, IsaacLabHearingManager):
            raise TypeError('manager must be IsaacLabHearingManager')
        if not isinstance(poses, IsaacLabSceneReader):
            raise TypeError('poses must be IsaacLabSceneReader')
        if int(env.num_envs) != manager.num_envs:
            raise ValueError('environment batch size mismatch')
        if not isinstance(key, str) or not key or key == 'policy':
            raise ValueError('use a distinct nonempty hearing observation key')
        self.env = env
        self.manager = manager
        self.poses = poses
        self.key = key
        self._step_id = 0
        # Allow the ObservationTermCfg hook to access the same manager if needed.
        self.env.onda_hearing_manager = manager

    def _augment(self, observations):
        if not isinstance(observations, Mapping):
            raise TypeError('expected dict-style Isaac Lab observation groups')
        if self.key in observations:
            raise ValueError(f'hearing observation key already exists: {self.key}')
        result = dict(observations)
        result[self.key] = self.manager.torch(device=self.env.device)
        return result

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError('expected reset() -> (observations, info)')
        obs, info = result
        # Reset all when resetting full env; partial reset is not assumed.
        self.manager.reset()
        self._step_id = 0
        self.manager.seed(*self.poses.read(self.env))
        return self._augment(obs), info

    def step(self, actions):
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError('expected step() -> obs, reward, terminated, truncated, extras')
        obs, reward, terminated, truncated, extras = result
        import numpy as np
        from .isaac_lab_runtime import _numpy
        term = _numpy(terminated, name='terminated').reshape(-1)
        trunc = _numpy(truncated, name='truncated').reshape(-1)
        if term.shape != (self.manager.num_envs,) or trunc.shape != (self.manager.num_envs,):
            raise ValueError('termination masks must have one value per environment')
        if not np.all(np.isin(term, [0, 1])) or not np.all(np.isin(trunc, [0, 1])):
            raise ValueError('termination masks must be boolean/0-1')
        resets = np.flatnonzero((term != 0) | (trunc != 0)).tolist()
        active = [i for i in range(self.manager.num_envs) if i not in set(resets)]
        poses = self.poses.read(self.env)
        # Isaac Lab typically auto-resets terminated envs before returning obs.
        # We discard old acoustics, then seed their post-reset pose at t=0.
        if resets:
            self.manager.reset(resets)
        self._step_id += 1
        self.manager.advance(*poses, active_env_ids=active, step_token=self._step_id)
        if resets:
            self.manager.seed(*poses, env_ids=resets)
        return self._augment(obs), reward, terminated, truncated, extras

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate standard task methods/attributes, not writable metadata.
        return getattr(self.env, name)
