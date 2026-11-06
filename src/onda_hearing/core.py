"""Simulator-independent, strict sensor and event contracts."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def vec3(value: object, name: str = "vector") -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite length-3 vector")
    return a.copy()


@dataclass(frozen=True)
class MicrophoneArray:
    """Positions in meters, array-local XYZ. Rows follow PCM channel order."""
    positions_m: np.ndarray
    frame_id: str = "microphone_array"

    def __post_init__(self) -> None:
        a = np.array(self.positions_m, dtype=np.float64, copy=True)
        if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 1:
            raise ValueError("positions_m must be [channels,3] with >=1 channels")
        if not np.isfinite(a).all() or not self.frame_id:
            raise ValueError("Invalid geometry/frame")
        if a.shape[0] > 1 and np.unique(a, axis=0).shape[0] != a.shape[0]:
            raise ValueError("Duplicate microphone positions")
        a.setflags(write=False)
        object.__setattr__(self, "positions_m", a)

    @property
    def channels(self) -> int:
        return self.positions_m.shape[0]

    @property
    def geometry_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.positions_m - self.positions_m.mean(axis=0), tol=1e-8))

    def validate_azimuth(self) -> None:
        if self.channels < 3 or np.linalg.matrix_rank(self.positions_m[:, :2] - self.positions_m[:, :2].mean(axis=0)) < 2:
            raise ValueError("Planar azimuth needs >=3 noncollinear XY microphone positions; geometry ambiguous")


@dataclass(frozen=True)
class AudioBlock:
    """Frame-major float PCM [samples,channels]. t0 is capture time, seconds."""
    pcm: np.ndarray
    sample_rate_hz: int
    t0_s: float
    frame_id: str

    def __post_init__(self) -> None:
        a = np.array(self.pcm, dtype=np.float32, copy=True)
        if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
            raise ValueError("pcm must be finite nonempty [samples,channels]")
        if self.sample_rate_hz <= 0 or not math.isfinite(self.t0_s) or not self.frame_id:
            raise ValueError("Invalid sample rate, timestamp, or frame")
        a.setflags(write=False)
        object.__setattr__(self, "pcm", a)

    @property
    def t1_s(self) -> float:
        return self.t0_s + len(self.pcm) / self.sample_rate_hz


@dataclass(frozen=True)
class Direction:
    """Azimuth radians in XY from +X toward +Y, relative to frame_id."""
    azimuth_rad: float
    score: float
    frame_id: str


@dataclass(frozen=True)
class SourceTrack:
    track_id: int
    azimuth_rad: float
    score: float
    frame_id: str
    first_seen_s: float
    last_seen_s: float
    confirmed: bool
    missed: int


@dataclass(frozen=True)
class HearingObservation:
    timestamp_s: float
    directions: tuple[Direction, ...]
    tracks: tuple[SourceTrack, ...]
    frame_id: str
    backend: str
    status: str
