"""Simulator-independent scene snapshots and scheduled source waveforms.

Coordinates: world XYZ meters, orientation quaternion XYZW (array -> world).
Times: nonnegative integer nanoseconds in a single named simulation clock.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import math
import numpy as np
from onda_hearing.contracts.geometry import vector3


def _q(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-6:
        raise ValueError('orientation_xyzw must be a finite unit quaternion')
    return tuple(float(x) for x in q)


def rotation_from_xyzw(value) -> np.ndarray:
    """Proper local-to-world 3x3 active rotation (Hamilton XYZW)."""
    x,y,z,w = _q(value)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)


def xyzw_from_rotation(value) -> tuple[float,float,float,float]:
    """Convert a proper active rotation matrix into normalized XYZW."""
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (3,3) or not np.isfinite(m).all() or not np.allclose(m.T@m,np.eye(3),atol=1e-6) or abs(np.linalg.det(m)-1)>1e-6:
        raise ValueError('rotation must be orthonormal with determinant +1')
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace+1)*2
        w=(.25*s);x=(m[2,1]-m[1,2])/s;y=(m[0,2]-m[2,0])/s;z=(m[1,0]-m[0,1])/s
    else:
        i=int(np.argmax(np.diag(m)))
        if i==0:
            s=2*math.sqrt(max(0,1+m[0,0]-m[1,1]-m[2,2]));x=.25*s;y=(m[0,1]+m[1,0])/s;z=(m[0,2]+m[2,0])/s;w=(m[2,1]-m[1,2])/s
        elif i==1:
            s=2*math.sqrt(max(0,1+m[1,1]-m[0,0]-m[2,2]));y=.25*s;x=(m[0,1]+m[1,0])/s;z=(m[1,2]+m[2,1])/s;w=(m[0,2]-m[2,0])/s
        else:
            s=2*math.sqrt(max(0,1+m[2,2]-m[0,0]-m[1,1]));z=.25*s;x=(m[0,2]+m[2,0])/s;y=(m[1,2]+m[2,1])/s;w=(m[1,0]-m[0,1])/s
    q=np.array([x,y,z,w],dtype=np.float64);q/=np.linalg.norm(q)
    return tuple(float(v) for v in q)


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    time_ns: int
    array_position_m: tuple[float,float,float]
    array_orientation_xyzw: tuple[float,float,float,float]
    source_positions_m: Mapping[str,tuple[float,float,float]]

    def __post_init__(self):
        if type(self.time_ns) is not int or self.time_ns < 0:
            raise ValueError('time_ns must be a nonnegative integer')
        object.__setattr__(self,'array_position_m',vector3(self.array_position_m,name='array_position_m'))
        object.__setattr__(self,'array_orientation_xyzw',_q(self.array_orientation_xyzw))
        if not isinstance(self.source_positions_m,Mapping):
            raise TypeError('source_positions_m must be a mapping')
        sources={}
        for name,position in self.source_positions_m.items():
            if not isinstance(name,str) or not name:
                raise ValueError('source IDs must be nonempty strings')
            sources[name]=vector3(position,name='source position')
        object.__setattr__(self,'source_positions_m',MappingProxyType(sources))


@dataclass(frozen=True, slots=True)
class ScheduledSource:
    source_id: str
    waveform: np.ndarray
    start_ns: int
    amplitude: float = 1.

    def __post_init__(self):
        if not isinstance(self.source_id,str) or not self.source_id:
            raise ValueError('source_id must be nonempty')
        if type(self.start_ns) is not int or self.start_ns < 0:
            raise ValueError('start_ns must be nonnegative int')
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError('amplitude must be finite and nonnegative')
        signal=np.array(self.waveform,dtype=np.float64,copy=True)
        if signal.ndim!=1 or signal.size==0 or not np.isfinite(signal).all():
            raise ValueError('waveform must be finite, nonempty mono samples')
        signal.setflags(write=False)
        object.__setattr__(self,'waveform',signal)
