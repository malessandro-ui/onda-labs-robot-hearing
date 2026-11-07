"""Independent, bounded state for vectorized robotics environments."""
from __future__ import annotations

import numpy as np
from .observations import AcousticEvidence, PolicyHearingConfig, PolicyHearingEncoder


class HearingObservationBank:
    def __init__(self, num_envs: int, config: PolicyHearingConfig | None = None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self._encoders = [PolicyHearingEncoder(config) for _ in range(num_envs)]

    def _check_id(self, env_id: int) -> int:
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise IndexError("invalid environment ID")
        return env_id

    def ingest(self, env_id: int, event: AcousticEvidence) -> None:
        self._encoders[self._check_id(env_id)].ingest(event)

    def ingest_hearing(self, env_id: int, observation) -> None:
        self._encoders[self._check_id(env_id)].ingest_hearing(observation)

    def ingest_loop(self, env_id: int, observation) -> None:
        self._encoders[self._check_id(env_id)].ingest_loop(observation)

    def reset(self, env_ids=None) -> None:
        ids = range(self.num_envs) if env_ids is None else env_ids
        for env_id in ids:
            self._encoders[self._check_id(int(env_id))].reset()

    def read(self, now_ns: int | list[int] | np.ndarray) -> np.ndarray:
        if isinstance(now_ns, (int, np.integer)):
            times = [int(now_ns)] * self.num_envs
        else:
            times = list(now_ns)
            if len(times) != self.num_envs:
                raise ValueError("expected one timestamp per environment")
        return np.stack([e.read(int(t)) for e, t in zip(self._encoders, times)], axis=0)

    def torch(self, now_ns, *, device=None):
        """Lazy torch import keeps core usable on MuJoCo-only machines."""
        import torch
        return torch.as_tensor(self.read(now_ns), dtype=torch.float32, device=device)
