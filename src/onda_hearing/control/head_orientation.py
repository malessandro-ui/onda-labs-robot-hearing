"""A small bounded head-yaw setpoint controller; not a physical safety system.

This code never accepts simulator source coordinates. It only accepts a
source bearing estimated from audio, plus the robot's own measured pose.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def wrap_degrees(deg: float) -> float:
    if not math.isfinite(deg):
        raise ValueError('angle must be finite')
    return (deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class AcousticBearing:
    time_ns: int
    world_azimuth_deg: float
    evidence_score: float
    world_frame: str = 'world'

    def __post_init__(self):
        if type(self.time_ns) is not int or self.time_ns < 0:
            raise ValueError('time_ns must be nonnegative int')
        if not isinstance(self.world_frame, str) or not self.world_frame:
            raise ValueError('world frame required')
        if not all(map(math.isfinite, (self.world_azimuth_deg, self.evidence_score))) or self.evidence_score < 0:
            raise ValueError('invalid bearing/score')


@dataclass(frozen=True, slots=True)
class HeadYawConfig:
    min_yaw_deg: float = -90.0
    max_yaw_deg: float = 90.0
    max_setpoint_speed_deg_s: float = 150.0
    min_evidence_score: float = 0.0
    max_observation_age_s: float = 0.5

    def __post_init__(self):
        values = (self.min_yaw_deg, self.max_yaw_deg, self.max_setpoint_speed_deg_s,
                  self.min_evidence_score, self.max_observation_age_s)
        if not all(map(math.isfinite, values)) or not -180 <= self.min_yaw_deg < self.max_yaw_deg <= 180:
            raise ValueError('invalid joint range')
        if self.max_setpoint_speed_deg_s <= 0 or self.min_evidence_score < 0 or self.max_observation_age_s <= 0:
            raise ValueError('invalid control limits')


@dataclass(frozen=True, slots=True)
class HeadYawCommand:
    target_joint_rad: float
    accepted: bool
    reason: str


class HeadYawController:
    """Produces a rate-limited, joint-limited position-actuator target.

    A caller must apply independent simulator/robot torque and speed limits.
    A missing or stale detection holds the last setpoint, rather than moving
    toward a privileged target. Reset is mandatory between episodes.
    """
    def __init__(self, config: HeadYawConfig = HeadYawConfig()):
        self.config = config
        self.reset()

    def reset(self, *, initial_joint_rad: float = 0.0):
        if not math.isfinite(initial_joint_rad):
            raise ValueError('initial joint must be finite')
        self._setpoint_deg = min(max(math.degrees(initial_joint_rad), self.config.min_yaw_deg), self.config.max_yaw_deg)
        self._last_ns: int | None = None
        self._bearing: AcousticBearing | None = None

    def set_bearing(self, value: AcousticBearing | None):
        if value is not None and not isinstance(value, AcousticBearing):
            raise TypeError('only AcousticBearing is accepted')
        if value is not None and self._bearing is not None and value.time_ns < self._bearing.time_ns:
            raise ValueError('out-of-order acoustic observation')
        self._bearing = value

    def command(self, *, now_ns: int, base_world_yaw_rad: float, dt_s: float) -> HeadYawCommand:
        if type(now_ns) is not int or now_ns < 0 or not math.isfinite(base_world_yaw_rad) or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('invalid time/base pose/control dt')
        if self._last_ns is not None and now_ns <= self._last_ns:
            raise ValueError('control timestamp must increase')
        self._last_ns = now_ns
        observation = self._bearing
        if observation is None:
            return HeadYawCommand(math.radians(self._setpoint_deg), False, 'no_acoustic_observation')
        age = (now_ns - observation.time_ns) / 1e9
        if age < 0 or age > self.config.max_observation_age_s:
            return HeadYawCommand(math.radians(self._setpoint_deg), False, 'stale_acoustic_observation')
        if observation.evidence_score < self.config.min_evidence_score:
            return HeadYawCommand(math.radians(self._setpoint_deg), False, 'insufficient_evidence')
        requested = wrap_degrees(observation.world_azimuth_deg - math.degrees(base_world_yaw_rad))
        desired = min(max(requested, self.config.min_yaw_deg), self.config.max_yaw_deg)
        delta = min(max(desired - self._setpoint_deg,
                        -self.config.max_setpoint_speed_deg_s * dt_s),
                    self.config.max_setpoint_speed_deg_s * dt_s)
        self._setpoint_deg += delta
        return HeadYawCommand(math.radians(self._setpoint_deg), True, 'audio_derived_target')
