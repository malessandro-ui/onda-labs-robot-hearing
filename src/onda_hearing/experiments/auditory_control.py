"""Acoustic-only decision boundary for closed-loop demonstrations.

The policy receives audio and array pose, never scene source positions. When
multiple hypotheses are close in strength, the controller holds instead of
pretending the candidate count is known.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from onda_hearing.control import AcousticBearing, wrap_degrees
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer
from onda_hearing.runtime.streaming import PerceptionWindow
from onda_hearing.contracts import MicrophoneArray


@dataclass(frozen=True, slots=True)
class AcousticSelectionConfig:
    min_score: float = 0.08
    ambiguous_ratio: float = 0.88
    stability_tolerance_deg: float = 18.0
    min_consistent_windows: int = 2

    def __post_init__(self):
        values = (self.min_score, self.ambiguous_ratio, self.stability_tolerance_deg)
        if not all(math.isfinite(v) for v in values) or self.min_score < 0 or not 0 < self.ambiguous_ratio <= 1 or not 0 < self.stability_tolerance_deg <= 180:
            raise ValueError('invalid acoustic selection settings')
        if type(self.min_consistent_windows) is not int or not 1 <= self.min_consistent_windows <= 16:
            raise ValueError('invalid stability window count')


@dataclass(frozen=True, slots=True)
class AuditoryDecision:
    bearing: AcousticBearing | None
    raw_bearings_deg: tuple[float, ...]
    raw_scores: tuple[float, ...]
    reason: str


class AcousticOnlySelector:
    """Conservative single-target choice from motion-aware acoustic evidence.

    Not a multisource association algorithm. This policy intentionally refuses
    ambiguous choices; it does not use simulation truth or source IDs.
    """
    def __init__(self, localizer: MotionAwareSRPLocalizer,
                 config: AcousticSelectionConfig = AcousticSelectionConfig()):
        self.localizer = localizer
        self.config = config
        self.reset()

    def reset(self):
        self._previous: float | None = None
        self._consistent = 0
        if hasattr(self.localizer, 'reset'):
            self.localizer.reset()

    def process(self, window: PerceptionWindow, array: MicrophoneArray) -> AuditoryDecision:
        frame = self.localizer.estimate(window.audio, array, window.poses,
                                        pose_clock_id=window.pose_clock_id,
                                        world_frame_id='world')
        ds = sorted(frame.result.directions, key=lambda d: d.score, reverse=True)
        angles = tuple(math.degrees(d.azimuth_rad) for d in ds)
        scores = tuple(float(d.score) for d in ds)
        if not ds or ds[0].score < self.config.min_score:
            self._previous = None
            self._consistent = 0
            return AuditoryDecision(None, angles, scores, 'no_supported_bearing')
        if len(ds) > 1 and ds[1].score >= ds[0].score * self.config.ambiguous_ratio:
            self._previous = None
            self._consistent = 0
            return AuditoryDecision(None, angles, scores, 'ambiguous_spatial_candidates')
        a = angles[0]
        if self._previous is not None and abs(wrap_degrees(a - self._previous)) <= self.config.stability_tolerance_deg:
            self._consistent += 1
        else:
            self._consistent = 1
        self._previous = a
        if self._consistent < self.config.min_consistent_windows:
            return AuditoryDecision(None, angles, scores, 'awaiting_consistent_audio_evidence')
        bearing = AcousticBearing(window.audio.capture.end_ns, a, scores[0])
        return AuditoryDecision(bearing, angles, scores, 'acoustic_bearing_accepted')
