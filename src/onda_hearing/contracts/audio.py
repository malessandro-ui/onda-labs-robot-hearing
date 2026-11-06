"""Immutable capture contracts: frame-major normalized float32 PCM."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .clocks import CaptureInterval
from .geometry import frame_id
from .microphones import MicrophoneArray

@dataclass(frozen=True,slots=True)
class AudioPacket:
    pcm: np.ndarray
    sample_rate_hz: int
    capture: CaptureInterval
    array_id: str
    frame_id: str
    geometry_id: str
    sequence: int = 0
    discontinuity: bool = False
    def __post_init__(self):
        frame_id(self.array_id);frame_id(self.frame_id);frame_id(self.geometry_id)
        if not isinstance(self.capture,CaptureInterval):raise TypeError('capture must be CaptureInterval')
        if type(self.sample_rate_hz) is not int or not 8000<=self.sample_rate_hz<=384000:
            raise ValueError('sample_rate_hz must be integer in 8000..384000')
        if type(self.sequence) is not int or self.sequence<0 or type(self.discontinuity) is not bool:
            raise ValueError('invalid sequence/discontinuity')
        a=np.array(self.pcm,dtype=np.float32,copy=True)
        if a.ndim!=2 or min(a.shape)<1 or a.shape[1]>256 or not np.isfinite(a).all():
            raise ValueError('pcm must be finite, nonempty [frames,channels]')
        if np.max(np.abs(a))>1+1e-6:raise ValueError('PCM must be normalized to [-1,1]; explicitly scale raw integer PCM')
        # Allow <= half a sample discrepancy due to integer-ns conversion.
        expected_ns=len(a)*1e9/self.sample_rate_hz
        if abs((self.capture.end_ns-self.capture.start_ns)-expected_ns)>max(1,0.51e9/self.sample_rate_hz):
            raise ValueError('capture interval disagrees with sample count/sample rate')
        a.setflags(write=False)
        object.__setattr__(self,'pcm',a)
    @property
    def channel_count(self):return self.pcm.shape[1]
    def validate_array(self,array:MicrophoneArray)->None:
        if (self.channel_count!=array.channels or self.array_id!=array.array_id or
            self.frame_id!=array.frame_id or self.geometry_id!=array.geometry_id):
            raise ValueError('audio/array channel or geometry identity mismatch')
    @classmethod
    def from_pcm(cls,pcm,sample_rate_hz,*,start_ns,clock_id,array:MicrophoneArray,sequence=0,discontinuity=False):
        n=np.asarray(pcm).shape[0]
        duration_ns=round(n*1e9/sample_rate_hz)
        return cls(pcm,sample_rate_hz,CaptureInterval(clock_id,start_ns,start_ns+duration_ns),array.array_id,array.frame_id,array.geometry_id,sequence,discontinuity)
