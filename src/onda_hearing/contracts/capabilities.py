"""Explicit capability negotiation; multi-peak output is not multisource support."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol,runtime_checkable
from .audio import AudioPacket
from .microphones import MicrophoneArray
from .localization import SpatialResult

class UnsupportedConfigurationError(ValueError):pass

@dataclass(frozen=True,slots=True)
class SpatialCapabilities:
    backend_id:str
    min_channels:int
    max_channels:int|None
    sample_rates_hz:frozenset[int]
    min_geometry_rank:int=0
    min_xy_rank:int=0
    multi_source:bool=False
    waveform_separation:bool=False
    stateful:bool=False
    max_sources:int=1
    def __post_init__(self):
        if not self.backend_id or type(self.min_channels) is not int or self.min_channels<1:
            raise ValueError('invalid backend/minimum channels')
        if self.max_channels is not None and (type(self.max_channels) is not int or self.max_channels<self.min_channels):raise ValueError('invalid maximum channels')
        if not self.sample_rates_hz or any(type(x) is not int or x<=0 for x in self.sample_rates_hz):raise ValueError('sample rate set required')
        if not 0<=self.min_geometry_rank<=3 or not 0<=self.min_xy_rank<=2:raise ValueError('invalid rank')
        if type(self.max_sources) is not int or self.max_sources<1:raise ValueError('invalid max_sources')
        if not self.multi_source and self.max_sources!=1:raise ValueError('single-source backend must expose max_sources=1')
        if self.waveform_separation and not self.multi_source:raise ValueError('waveform separation implies multisource capability')
    def validate(self,audio:AudioPacket,array:MicrophoneArray,*,requested_sources:int=1,require_separation:bool=False)->None:
        audio.validate_array(array)
        if not self.min_channels<=array.channels or (self.max_channels is not None and array.channels>self.max_channels):
            raise UnsupportedConfigurationError('microphone count unsupported by backend')
        if audio.sample_rate_hz not in self.sample_rates_hz:raise UnsupportedConfigurationError('sample rate unsupported by backend')
        if array.geometry_rank<self.min_geometry_rank or array.xy_rank<self.min_xy_rank:
            raise UnsupportedConfigurationError('microphone geometry lacks required independent dimensions')
        if requested_sources<1 or requested_sources>self.max_sources or (requested_sources>1 and not self.multi_source):
            raise UnsupportedConfigurationError('requested simultaneous-source capability unsupported')
        if require_separation and not self.waveform_separation:raise UnsupportedConfigurationError('waveform separation not supported')

@runtime_checkable
class SpatialBackend(Protocol):
    @property
    def capabilities(self)->SpatialCapabilities: ...
    def process(self,audio:AudioPacket,geometry:MicrophoneArray)->SpatialResult: ...
    def reset(self)->None: ...

def run_checked(backend:SpatialBackend,audio:AudioPacket,geometry:MicrophoneArray,*,requested_sources:int=1,require_separation:bool=False)->SpatialResult:
    backend.capabilities.validate(audio,geometry,requested_sources=requested_sources,require_separation=require_separation)
    result=backend.process(audio,geometry)
    if not isinstance(result,SpatialResult) or result.capture!=audio.capture or result.frame_id!=audio.frame_id:
        raise RuntimeError('backend violated public SpatialResult contract')
    if len(result.directions)>backend.capabilities.max_sources:
        raise RuntimeError('backend returned more directions than its declared capability')
    return result
