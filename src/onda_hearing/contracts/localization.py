"""Direction measurements are hypotheses, not guaranteed independent sources."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .geometry import frame_id,wrap_azimuth
from .clocks import CaptureInterval

@dataclass(frozen=True,slots=True)
class DirectionEstimate:
    azimuth_rad: float
    frame_id: str
    score: float
    elevation_rad: float|None=None
    angular_std_rad: float|None=None
    def __post_init__(self):
        frame_id(self.frame_id)
        if not math.isfinite(self.azimuth_rad):raise ValueError('nonfinite azimuth')
        object.__setattr__(self,'azimuth_rad',wrap_azimuth(self.azimuth_rad))
        if not math.isfinite(self.score) or self.score<0:raise ValueError('score must be finite and nonnegative (not necessarily probability)')
        if self.elevation_rad is not None and (not math.isfinite(self.elevation_rad) or abs(self.elevation_rad)>math.pi/2):
            raise ValueError('elevation must be within [-pi/2,pi/2]')
        if self.angular_std_rad is not None and (not math.isfinite(self.angular_std_rad) or self.angular_std_rad<=0 or self.angular_std_rad>math.pi):
            raise ValueError('invalid angular uncertainty')

@dataclass(frozen=True,slots=True)
class SpatialResult:
    capture:CaptureInterval
    frame_id:str
    directions:tuple[DirectionEstimate,...]
    backend_id:str
    def __post_init__(self):
        frame_id(self.frame_id);frame_id(self.backend_id)
        if not isinstance(self.capture,CaptureInterval):raise TypeError('capture interval required')
        ds=tuple(self.directions)
        if any(not isinstance(d,DirectionEstimate) or d.frame_id!=self.frame_id for d in ds):
            raise ValueError('all direction estimates must use result frame')
        object.__setattr__(self,'directions',ds)
