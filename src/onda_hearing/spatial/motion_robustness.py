"""Explicit pose-clock alignment and bounded pose perturbations for evaluation.

No clock-offset or pose-noise estimation is performed: callers must provide a
calibrated offset. Random perturbations represent stress tests, not an IMU model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from onda_hearing.physics.propagation import PoseTrajectory, _axis_angle

@dataclass(frozen=True)
class ClockAlignment:
    """audio_time_s = pose_time_s + offset_s (same time scale, no drift)."""
    pose_to_audio_offset_s: float
    audio_clock_id: str
    pose_clock_id: str
    def __post_init__(self):
        if not math.isfinite(self.pose_to_audio_offset_s):
            raise ValueError('clock offset must be finite')
        if not self.audio_clock_id or not self.pose_clock_id:
            raise ValueError('nonempty clock domain names required')

    def align(self, pose: PoseTrajectory, *, audio_clock_id: str, pose_clock_id: str) -> PoseTrajectory:
        if audio_clock_id != self.audio_clock_id or pose_clock_id != self.pose_clock_id:
            raise ValueError('clock identifier mismatch: explicit alignment refused')
        return PoseTrajectory(pose.times_s + self.pose_to_audio_offset_s,
                              pose.positions_m, pose.rotations)


def perturb_poses(poses: PoseTrajectory, *, seed: int,
                  yaw_std_deg: float = 0., position_std_m: float = 0.) -> PoseTrajectory:
    """Independent per-knot perturbations; seeded and intentionally synthetic.

    This is *not* a gyro-bias process, covariance propagation, or calibrated IMU.
    """
    if not (math.isfinite(yaw_std_deg) and 0 <= yaw_std_deg <= 90 and
            math.isfinite(position_std_m) and 0 <= position_std_m <= 5):
        raise ValueError('invalid pose perturbation standard deviation')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    errors = rng.normal(0., math.radians(yaw_std_deg), len(poses.times_s))
    positions = poses.positions_m + rng.normal(0., position_std_m, poses.positions_m.shape)
    rotations = np.stack([r @ _axis_angle(np.array([0.,0.,1.]), float(err))
                          for r,err in zip(poses.rotations,errors)])
    return PoseTrajectory(poses.times_s, positions, rotations)
