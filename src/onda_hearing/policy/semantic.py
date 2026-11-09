"""Typed semantic event records and capability-aware, policy-facing encodings.

This module represents AUDIO-INFERRED semantic events. It does not infer them,
and it must never be populated from simulator source labels in evaluation.
Class scores are backend-specific evidence, not calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class SemanticStatus(str, Enum):
    UNSUPPORTED = 'unsupported'
    NO_DETECTION = 'no_detection'
    DETECTED = 'detected'


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    label: str
    score: float
    onset_ns: int
    last_observed_ns: int
    backend_id: str
    clock_id: str = 'simulation'
    source_track_id: str | None = None
    embedding: tuple[float, ...] | None = None
    offset_ns: int | None = None

    def __post_init__(self):
        for name in ('label', 'backend_id', 'clock_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be nonempty')
        if type(self.onset_ns) is not int or type(self.last_observed_ns) is not int or self.onset_ns < 0 or self.last_observed_ns < self.onset_ns:
            raise ValueError('invalid semantic timestamps')
        if self.offset_ns is not None and (type(self.offset_ns) is not int or self.offset_ns < self.onset_ns):
            raise ValueError('invalid semantic offset')
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score) or self.score < 0:
            raise ValueError('score must be finite, nonnegative backend evidence')
        if self.source_track_id is not None and (not isinstance(self.source_track_id, str) or not self.source_track_id):
            raise ValueError('invalid source track ID')
        if self.embedding is not None:
            values = tuple(float(v) for v in self.embedding)
            if not values or len(values) > 8192 or not all(math.isfinite(v) for v in values):
                raise ValueError('invalid embedding')
            object.__setattr__(self, 'embedding', values)

    @property
    def duration_s(self) -> float | None:
        return None if self.offset_ns is None else (self.offset_ns - self.onset_ns) / 1e9


@dataclass(frozen=True, slots=True)
class SemanticResult:
    """Distinguishes unavailable recognition from an enabled empty result."""
    status: SemanticStatus
    events: tuple[SemanticEvent, ...] = ()
    clock_id: str = 'simulation'
    backend_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, SemanticStatus):
            raise TypeError('status must be SemanticStatus')
        events = tuple(self.events)
        if any(not isinstance(x, SemanticEvent) or x.clock_id != self.clock_id for x in events):
            raise ValueError('semantic event clock/type mismatch')
        if (self.status is SemanticStatus.DETECTED) != bool(events):
            raise ValueError('detected status must match nonempty events')
        if self.status is SemanticStatus.UNSUPPORTED and self.backend_id is not None:
            raise ValueError('unsupported semantic capability has no active backend')
        if self.status is not SemanticStatus.UNSUPPORTED and (not isinstance(self.backend_id, str) or not self.backend_id):
            raise ValueError('enabled semantic capability requires backend_id')
        object.__setattr__(self, 'events', events)


@dataclass(frozen=True, slots=True)
class SemanticPolicyConfig:
    """Frozen vocabulary for deterministic, fixed-size model observations."""
    labels: tuple[str, ...]
    score_scale: float = 1.0

    def __post_init__(self):
        labels = tuple(self.labels)
        if not labels or any(not isinstance(s, str) or not s for s in labels) or len(set(labels)) != len(labels):
            raise ValueError('labels must be distinct nonempty strings')
        if not math.isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError('invalid score scale')
        object.__setattr__(self, 'labels', labels)


class SemanticPolicyEncoder:
    """Shape: [capability_enabled, detections_present, per-class evidence...].

    This is an observation transform, NOT a sound recognition backend. Unknown
    labels are ignored rather than silently assigned to a different class.
    """
    def __init__(self, config: SemanticPolicyConfig):
        self.config = config
        self._labels = {name: i for i, name in enumerate(config.labels)}

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ('semantic_enabled', 'semantic_detected') + tuple('class:' + x for x in self.config.labels)

    def encode(self, result: SemanticResult) -> np.ndarray:
        if not isinstance(result, SemanticResult):
            raise TypeError('result must be SemanticResult')
        values = np.zeros(2 + len(self.config.labels), dtype=np.float32)
        if result.status is SemanticStatus.UNSUPPORTED:
            return values
        values[0] = 1.0
        values[1] = float(bool(result.events))
        for event in result.events:
            idx = self._labels.get(event.label)
            if idx is not None:
                values[idx + 2] = max(values[idx + 2], min(event.score / self.config.score_scale, 1.0))
        return values
