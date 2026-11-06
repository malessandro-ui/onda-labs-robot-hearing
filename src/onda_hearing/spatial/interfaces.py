"""Separate capability contracts for spatial inference, tracking and separation.

The caller must not interpret a spatial peak as a physically distinct source.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, SpatialResult

@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Diagnostics for one candidate; values are not probabilities."""
    azimuth_rad: float
    srp_score: float
    local_prominence: float
    relative_strength: float
    supported_pairs: int
    total_pairs: int
    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.azimuth_rad,self.srp_score,self.local_prominence,self.relative_strength)):
            raise ValueError('nonfinite evidence')
        if not 0 <= self.supported_pairs <= self.total_pairs or self.total_pairs < 0:
            raise ValueError('invalid pair support')

@dataclass(frozen=True, slots=True)
class LocalizationFrame:
    result: SpatialResult
    evidence: tuple[SourceEvidence, ...]
    method: str
    certified_source_count: bool = False
    def __post_init__(self):
        if len(self.result.directions) != len(self.evidence):
            raise ValueError('evidence must correspond to directions')
        if self.certified_source_count:
            raise ValueError('No built-in Onda v0.5 localizer certifies independent source count')

@dataclass(frozen=True, slots=True)
class SeparatedAudio:
    """Estimated component. Never imply ideal or validated isolation."""
    source_id: str
    pcm: np.ndarray
    sample_rate_hz: int
    method: str
    quality_validated: bool = False
    def __post_init__(self):
        arr = np.array(self.pcm,dtype=np.float32,copy=True)
        if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all() or not self.source_id:
            raise ValueError('invalid separated signal')
        if self.sample_rate_hz <= 0: raise ValueError('invalid sample rate')
        arr.setflags(write=False)
        object.__setattr__(self,'pcm',arr)

@runtime_checkable
class DirectionEstimator(Protocol):
    def estimate(self, audio: AudioPacket, array: MicrophoneArray) -> LocalizationFrame: ...
    def reset(self) -> None: ...

@runtime_checkable
class MultiSourceLocalizer(DirectionEstimator, Protocol):
    @property
    def maximum_candidates(self) -> int: ...

@runtime_checkable
class SourceTracker(Protocol):
    def update(self, observations: LocalizationFrame, timestamp_s: float): ...
    def reset(self) -> None: ...

@runtime_checkable
class SourceSeparator(Protocol):
    def separate(self, audio: AudioPacket, array: MicrophoneArray, observations: LocalizationFrame) -> tuple[SeparatedAudio,...]: ...
