"""Conservative bearing-only candidate tracker in a *fixed* coordinate frame.

Identity is a track hypothesis, not verified physical source identity. Moving
arrays require external timestamped pose compensation before passing bearings.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .interfaces import LocalizationFrame

def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi

@dataclass(frozen=True,slots=True)
class BearingTrack:
    track_id:int
    bearing_rad:float
    frame_id:str
    last_observed_s:float
    hits:int
    misses:int
    confirmed:bool
    coasting:bool

@dataclass
class _MutableTrack:
    id:int; bearing:float; frame:str; last:float; hits:int=1; consecutive:int=1; misses:int=0

class CandidateTracker:
    def __init__(self,*,gate_deg=18.,confirmation_hits=3,max_misses=3,max_idle_s=1.,smoothing=.3):
        if not(0<gate_deg<=90 and type(confirmation_hits)==int and confirmation_hits>=1 and type(max_misses)==int and max_misses>=0 and 0<max_idle_s and math.isfinite(max_idle_s) and 0<smoothing<=1):raise ValueError('invalid tracker parameters')
        self.gate=math.radians(gate_deg);self.confirmation_hits=confirmation_hits;self.max_misses=max_misses;self.max_idle_s=max_idle_s;self.smoothing=smoothing;self.reset()
    def reset(self):self.tracks=[];self._frame=None;self._time=None;self._next=1
    def update(self,observations:LocalizationFrame,timestamp_s:float)->tuple[BearingTrack,...]:
        if not math.isfinite(timestamp_s) or (self._time is not None and timestamp_s<=self._time):raise ValueError('timestamps must increase strictly')
        frame=observations.result.frame_id
        if self._frame is not None and frame!=self._frame:raise ValueError('frame changed: transform observations or reset')
        self._time=timestamp_s;self._frame=frame
        self.tracks=[s for s in self.tracks if timestamp_s-s.last<=self.max_idle_s]
        obs=observations.result.directions
        pairs=sorted((abs(wrap(s.bearing-o.azimuth_rad)),s.id,j) for s in self.tracks for j,o in enumerate(obs))
        assigned_tracks=set(); assigned_obs=set()
        for distance,tid,j in pairs:
            if distance>=self.gate or tid in assigned_tracks or j in assigned_obs:continue
            tr=next(s for s in self.tracks if s.id==tid);o=obs[j]
            tr.bearing=wrap(tr.bearing+self.smoothing*wrap(o.azimuth_rad-tr.bearing))
            tr.hits+=1;tr.consecutive=tr.consecutive+1 if tr.misses==0 else 1;tr.misses=0;tr.last=timestamp_s
            assigned_tracks.add(tid);assigned_obs.add(j)
        for tr in self.tracks:
            if tr.id not in assigned_tracks:tr.misses+=1;tr.consecutive=0
        self.tracks=[t for t in self.tracks if t.misses<=self.max_misses]
        for j,o in enumerate(obs):
            if j not in assigned_obs:
                self.tracks.append(_MutableTrack(self._next,o.azimuth_rad,frame,timestamp_s));self._next+=1
        return tuple(BearingTrack(t.id,t.bearing,frame,t.last,t.hits,t.misses,t.hits>=self.confirmation_hits and t.misses==0,t.misses>0) for t in sorted(self.tracks,key=lambda x:x.id))
