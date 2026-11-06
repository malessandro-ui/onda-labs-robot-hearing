"""Strict SI geometry and coordinate-frame conventions: right-handed XYZ."""
from __future__ import annotations
from dataclasses import dataclass
import math
import re
import numpy as np

_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_./-]*$")

def frame_id(value: str) -> str:
    if not isinstance(value,str) or not _FRAME.fullmatch(value) or "//" in value or ".." in value:
        raise ValueError("frame_id must be a nonempty, well-formed frame identifier")
    return value

def vector3(value, *, name="vector") -> tuple[float,float,float]:
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all(): raise ValueError(f"{name} must have three finite numbers")
    return tuple(float(v) for v in a)

@dataclass(frozen=True,slots=True)
class Pose:
    parent_frame: str
    child_frame: str
    translation_m: tuple[float,float,float]
    quaternion_xyzw: tuple[float,float,float,float]
    def __post_init__(self):
        frame_id(self.parent_frame);frame_id(self.child_frame)
        if self.parent_frame==self.child_frame: raise ValueError("pose parent and child must differ")
        object.__setattr__(self,'translation_m',vector3(self.translation_m,name='translation_m'))
        q=np.asarray(self.quaternion_xyzw,dtype=float)
        if q.shape!=(4,) or not np.isfinite(q).all() or not 0.999<=np.linalg.norm(q)<=1.001:
            raise ValueError('quaternion_xyzw must be a finite unit quaternion; normalize explicitly before constructing')
        object.__setattr__(self,'quaternion_xyzw',tuple(float(x) for x in q))

def wrap_azimuth(angle:float)->float:
    if not math.isfinite(angle):raise ValueError('angle must be finite')
    return (angle+math.pi)%(2*math.pi)-math.pi
