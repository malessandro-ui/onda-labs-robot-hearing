"""Timestamped local/world bearings. Explicitly separate observation and pose clocks.

A static world source changes *array-frame* bearing during array rotation.
Transforming a sensor-local direction with the array pose at its actual
capture time returns a stable world bearing (assuming calibrated extrinsics).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from onda_hearing.physics.propagation import PoseTrajectory


def wrap_rad(a):
    if not math.isfinite(float(a)):raise ValueError('nonfinite bearing')
    return (float(a)+math.pi)%(2*math.pi)-math.pi

@dataclass(frozen=True)
class TimedBearing:
    azimuth_rad:float
    timestamp_s:float
    frame_id:str
    clock_id:str
    def __post_init__(self):
        if not math.isfinite(self.azimuth_rad) or not math.isfinite(self.timestamp_s):
            raise ValueError('bearing/time must be finite')
        if not self.frame_id or not self.clock_id:raise ValueError('frame_id and clock_id required')
        object.__setattr__(self,'azimuth_rad',wrap_rad(self.azimuth_rad))

@dataclass(frozen=True)
class WorldBearing:
    azimuth_rad:float
    timestamp_s:float
    frame_id:str='world'
    def __post_init__(self):
        object.__setattr__(self,'azimuth_rad',wrap_rad(self.azimuth_rad))
        if not math.isfinite(self.timestamp_s):raise ValueError('nonfinite timestamp')

def to_world(bearing:TimedBearing,poses:PoseTrajectory,*,pose_clock_id:str,array_frame_id:str,
             world_frame_id:str='world',max_pose_gap_s:float=.05):
    """Local planar DOA -> world azimuth at timestamp, without inventing range.

    Reject unmatched clocks/frame and stale pose intervals. Static single-pose
    trajectories are accepted as fixed extrinsics; moving trajectories require
    bracketing pose samples within the configured gap.
    """
    if bearing.clock_id!=pose_clock_id:raise ValueError('clock mismatch: synchronize before transforming')
    if bearing.frame_id!=array_frame_id:raise ValueError('bearing frame differs from array pose frame')
    if not world_frame_id or not math.isfinite(max_pose_gap_s) or max_pose_gap_s<=0:
        raise ValueError('invalid world frame or pose gap')
    t=bearing.timestamp_s;times=poses.times_s
    if len(times)>1:
        if t<times[0] or t>times[-1]:raise ValueError('pose extrapolation refused')
        ix=np.searchsorted(times,t,side='left')
        if ix==0:gap=t-times[0]
        elif ix==len(times):gap=times[-1]-t
        else:gap=min(t-times[ix-1],times[ix]-t)
        if gap>max_pose_gap_s+1e-12:raise ValueError('nearest pose sample too far from observation')
    _,rot=poses.at(t)
    v=rot@np.array([math.cos(bearing.azimuth_rad),math.sin(bearing.azimuth_rad),0.])
    if np.hypot(v[0],v[1])<1e-7:raise ValueError('world ray nearly vertical: azimuth undefined')
    return WorldBearing(math.atan2(float(v[1]),float(v[0])),t,world_frame_id)

def angular_error_deg(actual_deg,estimate_deg):
    if not np.isfinite([actual_deg,estimate_deg]).all():raise ValueError('finite angles required')
    return abs((float(estimate_deg)-float(actual_deg)+180)%360-180)
