"""Simulator-neutral, explicit acoustic scene contracts. All distances in meters.

World convention: right-handed XYZ; quaternions XYZW. Times are *simulation*
seconds, and the PCM sample index is the authoritative clock. No ground-truth
emitter position is included in robot-facing observations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import math
import numpy as np


def vector3(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('expected a finite 3-vector')
    return tuple(float(x) for x in a)


def quat4(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all() or not np.isclose(np.linalg.norm(a), 1., atol=1e-5):
        raise ValueError('expected a unit XYZW quaternion')
    return tuple(float(x) for x in a/np.linalg.norm(a))


@dataclass(frozen=True)
class Pose3:
    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0., 0., 0., 1.)

    def __post_init__(self):
        object.__setattr__(self, 'xyz', vector3(self.xyz))
        object.__setattr__(self, 'quat_xyzw', quat4(self.quat_xyzw))


@dataclass(frozen=True)
class Binding:
    """Exactly one of `world` (fixed) or `object_path` (tracked)."""
    world: Pose3 | None = None
    object_path: str | None = None
    local_offset_m: tuple[float, float, float] = (0., 0., 0.)
    local_rotation_xyzw: tuple[float, float, float, float] = (0., 0., 0., 1.)

    def __post_init__(self):
        if (self.world is None) == (self.object_path is None):
            raise ValueError('binding requires exactly one of world or object_path')
        if self.object_path is not None and (not isinstance(self.object_path, str) or not self.object_path.strip()):
            raise ValueError('object_path must be nonempty')
        object.__setattr__(self, 'local_offset_m', vector3(self.local_offset_m))
        object.__setattr__(self, 'local_rotation_xyzw', quat4(self.local_rotation_xyzw))


@dataclass(frozen=True)
class Schedule:
    onset_s: float | None = 0.
    duration_s: float | None = None
    loop: bool = False

    def __post_init__(self):
        if self.onset_s is not None and (not math.isfinite(self.onset_s) or self.onset_s < 0):
            raise ValueError('onset must be finite and >=0')
        if self.duration_s is not None and (not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError('duration must be finite and >0')


@dataclass(frozen=True)
class Emitter:
    emitter_id: str
    asset_id: str
    binding: Binding
    schedule: Schedule = field(default_factory=Schedule)
    gain_db: float = 0.
    directivity: Literal['omni', 'cardioid'] = 'omni'
    enabled: bool = True

    def __post_init__(self):
        if not self.emitter_id or not self.asset_id:
            raise ValueError('emitter_id and asset_id required')
        if not math.isfinite(self.gain_db) or not -120 <= self.gain_db <= 40:
            raise ValueError('gain_db out of supported bounds')
        if self.directivity not in ('omni', 'cardioid'):
            raise ValueError('unsupported directivity')


@dataclass(frozen=True)
class MicArray:
    binding: Binding
    local_positions_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if not 1 <= len(self.local_positions_m) <= 64:
            raise ValueError('array must contain 1..64 microphones')
        object.__setattr__(self, 'local_positions_m', tuple(vector3(x) for x in self.local_positions_m))


def multiply_xyzw(a,b):
    x,y,z,w = a
    X,Y,Z,W = b
    q = (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
         w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)
    return quat4(q)


def rotate(q, vec):
    qv = np.asarray(q[:3], dtype=np.float64)
    x = np.asarray(vec, dtype=np.float64)
    t = 2*np.cross(qv, x)
    return x + float(q[3])*t + np.cross(qv,t)


def compose_pose(parent: Pose3, offset_xyz, offset_quat) -> Pose3:
    return Pose3(tuple(np.asarray(parent.xyz) + rotate(parent.quat_xyzw,offset_xyz)),
                 multiply_xyzw(parent.quat_xyzw,offset_quat))
