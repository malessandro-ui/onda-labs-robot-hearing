"""Isaac-independent closed-loop scheduler for a single physical yaw actuator.

The scene provider supplies world poses to the renderer. Ground-truth source
positions are NEVER forwarded to the localizer, evidence selector, or controller.
Caller owns Isaac's physics simulation and step ordering. No simulator packages
are imported here. For a physical robot, this bridge is only an example: it is
not a safety-rated controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.control import HeadYawConfig, HeadYawController, AcousticBearing
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig, AuditoryDecision
from onda_hearing.runtime import HearingRuntime, HearingWindowBuffer, ScheduledSource, SceneSnapshot
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig


class YawActuator(Protocol):
    def get_joint_yaw_rad(self) -> float: ...
    def get_base_yaw_rad(self) -> float: ...
    def apply_target_rad(self, target_rad: float) -> None: ...


class PoseProvider(Protocol):
    def snapshot(self, time_ns: int) -> SceneSnapshot: ...


@dataclass(frozen=True)
class HearingLoopConfig:
    physics_hz: int = 500
    sensor_interval_steps: int = 10
    perception_interval_steps: int = 50
    audio_rate_hz: int = 16000
    window_samples: int = 4096
    capacity_samples: int = 8192

    def __post_init__(self):
        for field in ('physics_hz', 'sensor_interval_steps', 'perception_interval_steps',
                      'audio_rate_hz', 'window_samples', 'capacity_samples'):
            v = getattr(self, field)
            if type(v) is not int or v <= 0:
                raise ValueError(f'{field} must be a positive integer')
        if self.physics_hz > 20000 or self.audio_rate_hz < 8000 or self.audio_rate_hz > 192000:
            raise ValueError('unsupported rates')
        if self.perception_interval_steps % self.sensor_interval_steps:
            raise ValueError('perception interval must be a multiple of sensor interval')
        if self.window_samples < 320 or self.window_samples > self.capacity_samples:
            raise ValueError('invalid audio window/capacity')
        if self.capacity_samples > self.audio_rate_hz * 10:
            raise ValueError('capacity must be bounded to ten seconds')


@dataclass(frozen=True)
class LoopObservation:
    time_ns: int
    joint_yaw_rad: float
    commanded_joint_rad: float
    estimated_bearing_deg: float | None
    evidence_score: float | None
    acoustic_reason: str
    control_reason: str
    new_acoustic_decision: bool


class IsaacHearingLoop:
    """One-step-before / one-step-after interface for Isaac Sim physics loops.

    Call `before_step(k)` immediately before physics step k, and `after_step(k)`
    immediately after physics step k, with k starting at 1. In particular,
    do not put world.step(render=True) between two calls to before_step.
    """

    def __init__(self, *, array: MicrophoneArray, sources: tuple[ScheduledSource, ...],
                 poses: PoseProvider, actuator: YawActuator,
                 config: HearingLoopConfig = HearingLoopConfig(),
                 localizer=None, selector_config: AcousticSelectionConfig | None = None,
                 controller_config: HeadYawConfig | None = None):
        self.array, self.poses, self.actuator, self.config = array, poses, actuator, config
        self.runtime = HearingRuntime(array, sources, sample_rate_hz=config.audio_rate_hz)
        self.buffer = HearingWindowBuffer(array, sample_rate_hz=config.audio_rate_hz,
                                          window_samples=config.window_samples,
                                          capacity_samples=config.capacity_samples)
        if localizer is None:
            localizer = MotionAwareSRPLocalizer(MotionSRPConfig(
                block_samples=320, hop_samples=160, angular_resolution_deg=2,
                max_candidates=3, min_relative_score=.65, min_prominence=.012))
        self.selector = AcousticOnlySelector(localizer, selector_config or AcousticSelectionConfig(
            min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
        self.controller = HeadYawController(controller_config or HeadYawConfig(
            max_setpoint_speed_deg_s=125., min_evidence_score=.08,
            max_observation_age_s=.3))
        self._last_step = 0
        self._awaiting_after = False
        self._last_command = None
        self._last_bearing: AcousticBearing | None = None
        self._last_reason = 'not_yet_observed'
        self._pcm: list[np.ndarray] = []
        self._observations: list[LoopObservation] = []
        self._new_decisions = 0
        self._accepted_decisions = 0
        self._initialized = False

    def _time_ns(self, step: int) -> int:
        # Integer scheduler, not repeated float addition.
        return round(step * 1_000_000_000 / self.config.physics_hz)

    def reset(self) -> None:
        self.runtime.reset()
        self.buffer.reset()
        self.selector.reset()
        self.controller.reset(initial_joint_rad=self.actuator.get_joint_yaw_rad())
        self._last_step = 0
        self._awaiting_after = False
        self._last_command = None
        self._last_bearing = None
        self._last_reason = 'not_yet_observed'
        self._pcm.clear()
        self._observations.clear()
        self._new_decisions = 0
        self._accepted_decisions = 0
        initial = self.poses.snapshot(0)
        if initial.time_ns != 0:
            raise ValueError('initial scene snapshot must start at time_ns=0')
        self.runtime.advance(initial)
        self.buffer.push_pose(initial)
        self._initialized = True

    def before_step(self, step: int) -> None:
        if not self._initialized:
            raise RuntimeError('call reset() first')
        if type(step) is not int or step != self._last_step + 1 or self._awaiting_after:
            raise ValueError('physics steps must be strictly sequential, alternating before/after')
        self._last_command = self.controller.command(
            now_ns=self._time_ns(step), base_world_yaw_rad=self.actuator.get_base_yaw_rad(),
            dt_s=1.0 / self.config.physics_hz)
        self.actuator.apply_target_rad(self._last_command.target_joint_rad)
        self._awaiting_after = True

    def after_step(self, step: int) -> LoopObservation | None:
        if not self._awaiting_after or step != self._last_step + 1:
            raise ValueError('after_step must follow the same before_step')
        self._awaiting_after = False
        self._last_step = step
        if step % self.config.sensor_interval_steps:
            return None
        now_ns = self._time_ns(step)
        scene = self.poses.snapshot(now_ns)
        if scene.time_ns != now_ns:
            raise ValueError('pose provider returned a mismatched timestamp')
        packet = self.runtime.advance(scene)
        self.buffer.push_pose(scene)
        if packet is not None:
            self.buffer.push_audio(packet.audio)
            self._pcm.append(np.array(packet.audio.pcm, copy=True))
        new_decision = False
        if step % self.config.perception_interval_steps == 0 and self.buffer.ready:
            decision: AuditoryDecision = self.selector.process(self.buffer.window(), self.array)
            new_decision = True
            self._new_decisions += 1
            self._last_reason = decision.reason
            self._last_bearing = decision.bearing
            self.controller.set_bearing(decision.bearing)
            if decision.bearing is not None:
                self._accepted_decisions += 1
        assert self._last_command is not None
        observation = LoopObservation(
            time_ns=now_ns,
            joint_yaw_rad=self.actuator.get_joint_yaw_rad(),
            commanded_joint_rad=self._last_command.target_joint_rad,
            estimated_bearing_deg=None if self._last_bearing is None else self._last_bearing.world_azimuth_deg,
            evidence_score=None if self._last_bearing is None else self._last_bearing.evidence_score,
            acoustic_reason=self._last_reason,
            control_reason=self._last_command.reason,
            new_acoustic_decision=new_decision)
        self._observations.append(observation)
        return observation

    @property
    def observations(self) -> tuple[LoopObservation, ...]:
        return tuple(self._observations)

    @property
    def new_decisions(self) -> int:
        return self._new_decisions

    @property
    def accepted_decisions(self) -> int:
        return self._accepted_decisions

    def microphone_pcm(self) -> np.ndarray:
        if not self._pcm:
            return np.zeros((0, self.array.channels), dtype=np.float32)
        return np.concatenate(self._pcm, axis=0)
