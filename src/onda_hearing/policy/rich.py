"""Extensible structured hearing representation (additive to v0.14 API)."""
from __future__ import annotations
from dataclasses import dataclass, field

from onda_hearing.contracts import SpatialResult, CaptureInterval
from onda_hearing.contracts.tracking import SourceTrack
from .semantic import SemanticResult, SemanticStatus


@dataclass(frozen=True, slots=True)
class AuditoryObservation:
    """Typed perception results; does not include source scene coordinates.

    Fields may be unsupported or empty. This representation is variable-length;
    use the policy encoders to construct fixed-shape ML inputs.
    """
    capture: CaptureInterval
    spatial: SpatialResult
    semantic: SemanticResult = field(default_factory=lambda: SemanticResult(SemanticStatus.UNSUPPORTED))
    tracks: tuple[SourceTrack, ...] = ()

    def __post_init__(self):
        if self.spatial.capture != self.capture:
            raise ValueError('capture mismatch')
        if self.semantic.clock_id != self.capture.clock_id:
            raise ValueError('semantic and capture clock mismatch')
        tracks = tuple(self.tracks)
        if any(not isinstance(t, SourceTrack) for t in tracks):
            raise TypeError('invalid source track')
        if len(set(t.track_id for t in tracks)) != len(tracks):
            raise ValueError('duplicate source track IDs')
        object.__setattr__(self, 'tracks', tracks)


@dataclass(frozen=True, slots=True)
class SourcePolicyConfig:
    """Fixed-width top-K tracking representation, not source ground truth."""
    max_sources: int = 4
    max_age_s: float = 1.0
    uncertainty_scale_rad: float = 1.0

    def __post_init__(self):
        import math
        if type(self.max_sources) is not int or not 1 <= self.max_sources <= 128:
            raise ValueError('max_sources must be 1..128')
        if not math.isfinite(self.max_age_s) or self.max_age_s <= 0:
            raise ValueError('max_age_s must be positive')
        if not math.isfinite(self.uncertainty_scale_rad) or self.uncertainty_scale_rad <= 0:
            raise ValueError('uncertainty_scale_rad must be positive')


class SourcePolicyEncoder:
    """Padded (K, 7) tensor: sin, cos, scaled angular std, std_known, age, confirmed, valid.

    Tracks retain their identifier in structured AuditoryObservation; this
    numerical tensor does not equate a row index to a persistent source ID.
    Unknown angular uncertainty is explicitly masked with std_known=0.
    """
    def __init__(self, config: SourcePolicyConfig | None = None):
        self.config = config or SourcePolicyConfig()

    def encode(self, tracks: tuple[SourceTrack, ...], *, now_ns: int, frame_id: str) -> 'np.ndarray':
        import math
        import numpy as np
        from onda_hearing.contracts.tracking import TrackState
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError('invalid now_ns')
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError('invalid frame_id')
        output = np.zeros((self.config.max_sources, 7), dtype=np.float32)
        candidates = []
        for track in tracks:
            if not isinstance(track, SourceTrack) or track.frame_id != frame_id:
                raise ValueError('source frame/type mismatch')
            age_ns = now_ns - track.last_observed_ns
            if age_ns < 0:
                raise ValueError('source track in the future')
            if age_ns / 1e9 <= self.config.max_age_s:
                candidates.append(track)
        # Rank confirmed, then freshness, then ID to make ties deterministic.
        candidates.sort(key=lambda t: (t.state is not TrackState.CONFIRMED,
                                       -t.last_observed_ns, t.track_id))
        for i, track in enumerate(candidates[:self.config.max_sources]):
            output[i] = (
                math.sin(track.azimuth_rad), math.cos(track.azimuth_rad),
                0. if track.angular_std_rad is None else min(track.angular_std_rad / self.config.uncertainty_scale_rad, 1.),
                float(track.angular_std_rad is not None),
                (now_ns - track.last_observed_ns) / 1e9,
                float(track.state is TrackState.CONFIRMED), 1.
            )
        return output
