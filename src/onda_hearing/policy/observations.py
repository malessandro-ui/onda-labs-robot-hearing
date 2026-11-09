"""Simulator-independent, policy-facing acoustic observations.

The five floats are, in order: sin(azimuth), cos(azimuth), scaled evidence,
validity (0/1), and age in seconds.  An invalid observation is all zeros
except for its age; the value does not encode a fictitious forward bearing.
Evidence is NOT a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

POLICY_FEATURES = (
    "bearing_sin", "bearing_cos", "evidence_scaled", "valid", "age_s"
)


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class PolicyHearingConfig:
    frame_id: str = "world"
    min_evidence: float = 0.08
    evidence_scale: float = 1.0
    max_age_s: float = 0.3
    ambiguous_ratio: float = 0.88

    def __post_init__(self) -> None:
        if not self.frame_id or not isinstance(self.frame_id, str):
            raise ValueError("frame_id must be a nonempty string")
        if _finite(self.min_evidence, "min_evidence") < 0:
            raise ValueError("min_evidence must be nonnegative")
        if _finite(self.evidence_scale, "evidence_scale") <= 0:
            raise ValueError("evidence_scale must be positive")
        if _finite(self.max_age_s, "max_age_s") <= 0:
            raise ValueError("max_age_s must be positive")
        if not 0 < _finite(self.ambiguous_ratio, "ambiguous_ratio") <= 1:
            raise ValueError("ambiguous_ratio must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class AcousticEvidence:
    """Single audio-derived candidate; no source position or ground truth."""
    time_ns: int
    azimuth_rad: float | None
    score: float = 0.0
    frame_id: str = "world"
    accepted: bool = False

    def __post_init__(self) -> None:
        if type(self.time_ns) is not int or self.time_ns < 0:
            raise ValueError("time_ns must be a nonnegative integer")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id required")
        if _finite(self.score, "score") < 0:
            raise ValueError("score must be nonnegative")
        if self.accepted and self.azimuth_rad is None:
            raise ValueError("accepted evidence requires a bearing")
        if self.azimuth_rad is not None:
            _finite(self.azimuth_rad, "azimuth_rad")


class PolicyHearingEncoder:
    """Owns one environment's acoustic state, timestamp discipline and reset.

    `ingest` updates state only with a new acoustic decision. A rejected decision
    explicitly clears prior evidence; a gap without an event does not. `read`
    returns a copy of fixed-shape float32 data and never mutates the timestamp.
    """

    def __init__(self, config: PolicyHearingConfig | None = None):
        self.config = config or PolicyHearingConfig()
        self.reset()

    def reset(self) -> None:
        self._last_event_ns: int | None = None
        self._bearing_rad: float | None = None
        self._score = 0.0
        self._accepted = False

    def ingest(self, event: AcousticEvidence) -> None:
        if not isinstance(event, AcousticEvidence):
            raise TypeError("event must be AcousticEvidence")
        if event.frame_id != self.config.frame_id:
            raise ValueError("acoustic frame mismatch")
        if self._last_event_ns is not None and event.time_ns <= self._last_event_ns:
            raise ValueError("out-of-order or duplicate acoustic event")
        self._last_event_ns = event.time_ns
        accepted = event.accepted and event.score >= self.config.min_evidence
        self._accepted = bool(accepted)
        self._bearing_rad = event.azimuth_rad if accepted else None
        self._score = event.score if accepted else 0.0

    def ingest_hearing(self, observation: Any) -> None:
        """Consume existing v0.14 HearingObservation, without simulator truth.

        Its `spatial.directions` contains hypotheses, not calibrated probabilities.
        If top two hypotheses are ambiguous, reject the entire update.
        """
        spatial = observation.spatial
        if observation.capture != spatial.capture:
            raise ValueError("capture mismatch")
        directions = sorted(spatial.directions, key=lambda d: d.score, reverse=True)
        ambiguous = (
            len(directions) > 1
            and directions[1].score >= directions[0].score * self.config.ambiguous_ratio
        )
        accepted = bool(directions and not ambiguous)
        direction = directions[0] if accepted else None
        if direction is not None and direction.frame_id != spatial.frame_id:
            raise ValueError("direction frame mismatch")
        self.ingest(AcousticEvidence(
            time_ns=int(observation.capture.end_ns),
            azimuth_rad=float(direction.azimuth_rad) if direction is not None else None,
            score=float(direction.score) if direction is not None else 0.0,
            frame_id=spatial.frame_id,
            accepted=accepted,
        ))

    def ingest_loop(self, observation: Any) -> None:
        """Consume a new accepted/rejected v0.14 LoopObservation event.

        Retained telemetry rows are intentionally not treated as fresh audio.
        """
        if not observation.new_acoustic_decision:
            return
        accepted = (
            observation.acoustic_reason == "acoustic_bearing_accepted"
            and observation.estimated_bearing_deg is not None
            and observation.evidence_score is not None
        )
        self.ingest(AcousticEvidence(
            time_ns=int(observation.time_ns),
            azimuth_rad=math.radians(float(observation.estimated_bearing_deg)) if accepted else None,
            score=float(observation.evidence_score) if accepted else 0.0,
            frame_id=self.config.frame_id,
            accepted=accepted,
        ))

    def read(self, now_ns: int) -> np.ndarray:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        if self._last_event_ns is None:
            return np.zeros(5, dtype=np.float32)
        if now_ns < self._last_event_ns:
            raise ValueError("policy clock precedes latest acoustic event")
        age_s = (now_ns - self._last_event_ns) / 1e9
        clipped_age = min(age_s, self.config.max_age_s)
        result = np.zeros(5, dtype=np.float32)
        result[4] = clipped_age
        if self._accepted and age_s <= self.config.max_age_s:
            assert self._bearing_rad is not None
            result[0] = math.sin(self._bearing_rad)
            result[1] = math.cos(self._bearing_rad)
            result[2] = min(self._score / self.config.evidence_scale, 1.0)
            result[3] = 1.0
        return result
