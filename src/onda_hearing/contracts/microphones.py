"""Arbitrary microphone geometry; channel ordering is explicit."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .geometry import frame_id,vector3

@dataclass(frozen=True,slots=True)
class Microphone:
    name: str
    channel: int
    position_m: tuple[float,float,float]
    def __post_init__(self):
        frame_id(self.name)
        if type(self.channel) is not int or self.channel<0: raise ValueError('channel must be a nonnegative integer')
        object.__setattr__(self,'position_m',vector3(self.position_m,name='microphone position'))

@dataclass(frozen=True,slots=True)
class MicrophoneArray:
    array_id: str
    frame_id: str
    microphones: tuple[Microphone,...]
    geometry_id: str
    def __post_init__(self):
        frame_id(self.array_id);frame_id(self.frame_id);frame_id(self.geometry_id)
        m=tuple(self.microphones)
        if not m:raise ValueError('at least one microphone is required')
        if len(m)>256:raise ValueError('over 256 microphones requires explicit SDK extension')
        if any(not isinstance(x,Microphone) for x in m):raise TypeError('microphones must contain Microphone objects')
        if len({x.name for x in m})!=len(m):raise ValueError('microphone names must be unique')
        if sorted(x.channel for x in m)!=list(range(len(m))):
            raise ValueError('channel indices must be contiguous 0..N-1 without duplicates')
        m=tuple(sorted(m,key=lambda x:x.channel))
        p=np.asarray([x.position_m for x in m])
        if len(m)>1 and np.unique(p,axis=0).shape[0]!=len(m):raise ValueError('coincident microphone positions not permitted')
        object.__setattr__(self,'microphones',m)
    @property
    def channels(self)->int:return len(self.microphones)
    @property
    def positions_m(self)->np.ndarray:
        arr=np.asarray([x.position_m for x in self.microphones],dtype=np.float64)
        arr.setflags(write=False);return arr
    @property
    def geometry_rank(self)->int:
        p=self.positions_m
        return int(np.linalg.matrix_rank(p-p.mean(axis=0),tol=1e-8))
    @property
    def xy_rank(self)->int:
        p=self.positions_m[:,:2]
        return int(np.linalg.matrix_rank(p-p.mean(axis=0),tol=1e-8))
    @classmethod
    def from_positions(cls,positions_m,*,array_id='array',frame_id='microphone_array',geometry_id='custom'):
        p=np.asarray(positions_m,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3:raise ValueError('positions_m must be [N,3]')
        return cls(array_id,frame_id,tuple(Microphone(f'mic_{i}',i,tuple(row)) for i,row in enumerate(p)),geometry_id)
