"""Persistent track records, distinct from spatial candidate observations."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from .geometry import frame_id,wrap_azimuth

class TrackState(str,Enum):
    TENTATIVE='tentative'
    CONFIRMED='confirmed'
    COASTING='coasting'

@dataclass(frozen=True,slots=True)
class SourceTrack:
    track_id:str
    frame_id:str
    azimuth_rad:float
    state:TrackState
    last_observed_ns:int
    angular_std_rad:float|None=None
    def __post_init__(self):
        frame_id(self.track_id);frame_id(self.frame_id)
        if not isinstance(self.state,TrackState):raise ValueError('invalid state')
        if not math.isfinite(self.azimuth_rad):raise ValueError('nonfinite bearing')
        object.__setattr__(self,'azimuth_rad',wrap_azimuth(self.azimuth_rad))
        if type(self.last_observed_ns) is not int or self.last_observed_ns<0:raise ValueError('invalid last observed timestamp')
        if self.angular_std_rad is not None and (not math.isfinite(self.angular_std_rad) or self.angular_std_rad<=0):raise ValueError('invalid angular std')
