"""Policy projection: independent spatial / semantic streams, NO fake association."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class HearingFrame:
    capture_time_ns:int
    bearing_local_deg:float|None
    spatial_confidence:float
    events:tuple[tuple[str,float],...]
    spatial_age_s:float
    semantic_age_s:float
    spatial_valid:bool
    semantic_valid:bool
    association_valid:bool=False

class PolicyEncoder:
    """Stable vector [sin bearing,cos bearing,spatial score,spatial valid,
    spatial age, semantic valid, semantic age, per-vocabulary label scores].

    0 is an absent score, with separate validity bit to disambiguate.
    A label score describes the WHOLE audio window, never a tracked source.
    """
    def __init__(self,vocabulary:tuple[str,...]):
        if len(set(vocabulary))!=len(vocabulary) or not vocabulary:
            raise ValueError('vocabulary must be unique nonempty labels')
        self.vocabulary=tuple(vocabulary);self.index={v:i for i,v in enumerate(vocabulary)}
    @property
    def size(self):return 7+len(self.vocabulary)
    def encode(self,spatial,semantic,now_capture_ns:int):
        x=np.zeros(self.size,dtype=np.float32)
        if spatial.valid:
            a=math.radians(spatial.bearing_local_deg)
            x[0:5]=[math.sin(a),math.cos(a),spatial.confidence,1.,max(0,(now_capture_ns-spatial.capture_time_ns)/1e9)]
        if semantic.valid:
            x[5:7]=[1.,max(0,(now_capture_ns-semantic.capture_mid_ns)/1e9)]
            for e in semantic.events:
                if e.label in self.index:
                    i=7+self.index[e.label];x[i]=max(x[i],e.score)
        return x

class UnifiedHearing:
    """Thin glue around v0.16: separate bounded PCM rings, shared sample clock."""
    def __init__(self,stream,semantic,encoder:PolicyEncoder):
        if semantic.buffer is stream.buffer:
            raise ValueError('semantic PCM ring must be independent of spatial ring')
        if (semantic.buffer.capture_start_ns!=stream.buffer.capture_start_ns or
            semantic.buffer.config.sample_rate!=stream.buffer.config.sample_rate or
            semantic.buffer.config.channels!=stream.buffer.config.channels):
            raise ValueError('semantic/spatial rings must share time origin, rate and channels')
        self.stream=stream;self.semantic=semantic;self.encoder=encoder
    def push_pose(self,pose):return self.stream.push_pose(pose)
    def push_audio(self,pcm,start_sample):
        self.stream.push_audio(pcm,start_sample)
        self.semantic.buffer.append(pcm,start_sample)
        self.semantic.notify_audio()
    def poll(self,now_capture_ns):
        spatial,a=self.stream.poll(now_capture_ns)
        semantic,b=self.semantic.poll(now_capture_ns)
        return self.encoder.encode(spatial,semantic,now_capture_ns),spatial,semantic,{**a,**b}
    def close(self):
        semantic_ok=self.semantic.close()
        spatial_ok=self.stream.close()
        return semantic_ok and spatial_ok

def attach_sound_brain(stream, brain=None, *, vocabulary=None,
                       semantic_window_s=2.0, semantic_hop_s=0.5,
                       semantic_buffer_s=4.0, reference_channel=0,
                       start_worker=True, **onda_brain_kwargs):
    """High-level one-call semantic attachment to an existing v0.16 stream.

    Default: Onda Sound Brain (BEATs), loaded from configured trusted local
    checkpoint. Alternative: pass any `SoundBrain` implementation and a stable
    policy vocabulary. No silent or fabricated fallback if weights are missing.
    """
    from .beats import OndaSoundBrain
    from .core import AsyncSemanticHearing
    from onda_hearing.streaming.core import PCMBuffer,StreamConfig
    if brain is None:
        brain=OndaSoundBrain(**onda_brain_kwargs)
    elif onda_brain_kwargs:
        raise ValueError('BEATs-specific options cannot be used with custom brain')
    if vocabulary is None:
        if not hasattr(brain,'labels'):
            raise ValueError('custom SoundBrain requires explicit vocabulary')
        vocabulary=tuple(brain.labels)
    config=stream.config
    semantic_config=StreamConfig(sample_rate=config.sample_rate,
                                 channels=config.channels,
                                 window_samples=int(round(semantic_window_s*config.sample_rate)),
                                 hop_samples=int(round(semantic_hop_s*config.sample_rate)),
                                 buffer_seconds=semantic_buffer_s)
    buffer=PCMBuffer(semantic_config,stream.buffer.capture_start_ns)
    worker=AsyncSemanticHearing(buffer,brain,
                                window_samples=semantic_config.window_samples,
                                hop_samples=semantic_config.hop_samples,
                                reference_channel=reference_channel,
                                start_worker=start_worker)
    return UnifiedHearing(stream,worker,PolicyEncoder(tuple(vocabulary)))
