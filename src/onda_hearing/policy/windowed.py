"""Simulator-neutral microphone PCM -> windowed spatial perception -> policy features.

The renderer alone may read SceneSnapshot source positions. The perception
selector sees only audio samples, array geometry, and the microphone trajectory.
Robot action, reward, and physics stepping belong to callers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from onda_hearing.contracts import DirectionEstimate, SpatialResult
from onda_hearing.contracts.observations import HearingObservation
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AuditoryDecision
from onda_hearing.policy.observations import AcousticEvidence, PolicyHearingConfig, PolicyHearingEncoder
from onda_hearing.runtime import HearingRuntime, HearingWindowBuffer, SceneSnapshot


@dataclass(frozen=True, slots=True)
class WindowedPipelineConfig:
    perception_period_ns: int = 100_000_000
    window_samples: int = 4096
    capacity_samples: int = 8192
    frame_id: str = 'world'

    def __post_init__(self):
        if type(self.perception_period_ns) is not int or self.perception_period_ns <= 0:
            raise ValueError('perception_period_ns must be a positive integer')
        if type(self.window_samples) is not int or type(self.capacity_samples) is not int:
            raise ValueError('window/capacity must be integers')
        if not 320 <= self.window_samples <= self.capacity_samples:
            raise ValueError('invalid window/capacity sizes')
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError('frame_id required')


@dataclass(frozen=True, slots=True)
class PolicyFrame:
    """Returned at every sensor step; decision is None between updates.

    features is copied, fixed shape (5,), float32. The public observation is a
    genuine audio-inferred measurement on new decision steps, not scene truth.
    """
    time_ns: int
    features: np.ndarray
    new_decision: bool
    decision_reason: str | None
    observation: HearingObservation | None
    audio_samples: int


class WindowedHearingPipeline:
    """Owns one array/environment; caller owns simulation stepping.

    The first advance initializes the clock. On each later call, the renderer
    advances exactly once. A bounded window of generated PCM plus pose history
    is passed to the existing AcousticOnlySelector at a deterministic cadence.
    A rejection invalidates the previous direction; missing updates simply age.
    """

    def __init__(self, runtime: HearingRuntime, selector: AcousticOnlySelector, *,
                 config: WindowedPipelineConfig | None = None,
                 policy_config: PolicyHearingConfig | None = None):
        if not isinstance(runtime, HearingRuntime):
            raise TypeError('runtime must be HearingRuntime')
        if not isinstance(selector, AcousticOnlySelector):
            raise TypeError('selector must be AcousticOnlySelector')
        if runtime.perception is not None:
            raise ValueError('windowed pipeline owns perception; runtime.perception must be None')
        self.runtime = runtime
        self.selector = selector
        self.config = config or WindowedPipelineConfig()
        pc = policy_config or PolicyHearingConfig(frame_id=self.config.frame_id)
        if pc.frame_id != self.config.frame_id:
            raise ValueError('policy and pipeline coordinate frames differ')
        self.encoder = PolicyHearingEncoder(pc)
        self.buffer = HearingWindowBuffer(runtime.array,
            sample_rate_hz=runtime.config.sample_rate_hz, clock_id=runtime.clock_id,
            window_samples=self.config.window_samples,
            capacity_samples=self.config.capacity_samples)
        self.reset()

    def reset(self) -> None:
        self.runtime.reset()
        self.selector.reset()
        self.buffer.reset()
        self.encoder.reset()
        self._last_time_ns: int | None = None
        self._next_perception_ns: int | None = None
        self._decisions = 0
        self._accepted = 0

    @property
    def decisions(self) -> int:
        return self._decisions

    @property
    def accepted(self) -> int:
        return self._accepted

    def advance(self, snapshot: SceneSnapshot) -> PolicyFrame:
        if not isinstance(snapshot, SceneSnapshot):
            raise TypeError('snapshot must be SceneSnapshot')
        now = snapshot.time_ns
        if self._last_time_ns is not None and now <= self._last_time_ns:
            raise ValueError('simulation time must increase; call reset after rewind')
        result = self.runtime.advance(snapshot)
        self.buffer.push_pose(snapshot)
        if self._last_time_ns is None:
            self._next_perception_ns = now + self.config.perception_period_ns
        self._last_time_ns = now
        if result is not None:
            self.buffer.push_audio(result.audio)
        observation = None
        reason = None
        changed = False
        if self._next_perception_ns is not None and now >= self._next_perception_ns:
            # Advance deadline even if a sensor step was late; never invent audio.
            n = (now - self._next_perception_ns) // self.config.perception_period_ns + 1
            self._next_perception_ns += n * self.config.perception_period_ns
            if self.buffer.ready:
                window = self.buffer.window()
                decision: AuditoryDecision = self.selector.process(window, self.runtime.array)
                changed = True
                self._decisions += 1
                reason = decision.reason
                good = decision.bearing is not None and reason == 'acoustic_bearing_accepted'
                if good:
                    self._accepted += 1
                event = AcousticEvidence(
                    time_ns=window.audio.capture.end_ns,
                    azimuth_rad=math.radians(decision.bearing.world_azimuth_deg) if good else None,
                    score=float(decision.bearing.evidence_score) if good else 0.,
                    frame_id=self.config.frame_id,
                    accepted=good,
                )
                self.encoder.ingest(event)
                directions = (DirectionEstimate(event.azimuth_rad,
                    self.config.frame_id, event.score),) if good else ()
                spatial = SpatialResult(window.audio.capture, self.config.frame_id,
                    directions, 'motion_srp_windowed_selector')
                observation = HearingObservation(window.audio.capture, spatial)
        return PolicyFrame(now, self.encoder.read(now), changed, reason, observation,
                           0 if result is None else int(len(result.audio.pcm)))
