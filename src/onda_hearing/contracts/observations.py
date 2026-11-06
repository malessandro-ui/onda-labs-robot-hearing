"""Policy-facing observations never contain privileged simulator truth."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .clocks import CaptureInterval
from .localization import SpatialResult
from .tracking import SourceTrack
from .geometry import frame_id

@dataclass(frozen=True,slots=True)
class HearingObservation:
    capture:CaptureInterval
    spatial:SpatialResult
    tracks:tuple[SourceTrack,...]=()
    semantic_events:tuple=()
    schema_version:str='1.0'
    def __post_init__(self):
        if self.spatial.capture!=self.capture:raise ValueError('spatial/capture time mismatch')
        if self.schema_version!='1.0':raise ValueError('unsupported observation schema version')
        ts=tuple(self.tracks)
        if any(not isinstance(t,SourceTrack) for t in ts):raise TypeError('tracks must be SourceTrack values')
        if len({t.track_id for t in ts})!=len(ts):raise ValueError('duplicate track ID')
        object.__setattr__(self,'tracks',ts)
        object.__setattr__(self,'semantic_events',tuple(self.semantic_events))
