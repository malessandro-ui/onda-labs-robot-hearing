"""Shared offline/chunked hearing runtime; optional audio-derived perception.

No simulation truth is provided to the perception backend. Scene source positions
are used only to produce physical microphone PCM.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, SpatialResult, CaptureInterval
from onda_hearing.contracts.observations import HearingObservation
from onda_hearing.contracts.geometry import frame_id
from onda_hearing.physics.propagation import AcousticConfig, DirectPathRenderer, PhysicalSource, PoseTrajectory
from .scene import SceneSnapshot,ScheduledSource,rotation_from_xyzw
from .scheduler import SampleScheduler

@dataclass(frozen=True, slots=True)
class RuntimeOutput:
    audio: AudioPacket
    observation: HearingObservation
    sample_start: int
    sample_stop: int

class HearingRuntime:
    """One microphone array, finite scheduled sources, one simulation clock.

    Call advance(initial_snapshot) to establish an epoch, then call with strictly
    increasing snapshots. Source samples are referenced to simulation absolute time.
    Long advances beyond AcousticConfig.max_samples fail closed. No clipping.
    """
    def __init__(self,array:MicrophoneArray, sources:tuple[ScheduledSource,...],*,
                 sample_rate_hz:int=16000,clock_id:str='simulation',
                 perception=None, config:AcousticConfig|None=None):
        if not isinstance(array,MicrophoneArray):raise TypeError('array must be MicrophoneArray')
        frame_id(clock_id)
        src=tuple(sources)
        if any(not isinstance(s,ScheduledSource) for s in src) or len(set(s.source_id for s in src))!=len(src):
            raise ValueError('sources must be distinct ScheduledSource instances')
        self.array=array;self.sources=src;self.clock_id=clock_id;self.perception=perception
        self.config=config or AcousticConfig(sample_rate_hz=sample_rate_hz)
        if self.config.sample_rate_hz!=sample_rate_hz:raise ValueError('renderer and runtime sample rates differ')
        self.renderer=DirectPathRenderer(self.config)
        self.reset()

    def reset(self)->None:
        self._history:list[SceneSnapshot]=[]
        self._scheduler:SampleScheduler|None=None
        self._sequence=0
        if self.perception is not None and hasattr(self.perception,'reset'):
            self.perception.reset()

    def _trajectory(self,snapshots,positions):
        times=np.array([s.time_ns/1e9 for s in snapshots],dtype=np.float64)
        return PoseTrajectory(times,np.array(positions,dtype=np.float64))

    def advance(self,snapshot:SceneSnapshot)->RuntimeOutput|None:
        if not isinstance(snapshot,SceneSnapshot):raise TypeError('snapshot must be SceneSnapshot')
        required={s.source_id for s in self.sources}
        if set(snapshot.source_positions_m)!=required:
            raise ValueError('snapshot source IDs must exactly match scheduled source IDs')
        if self._scheduler is None:
            self._history=[snapshot]
            self._scheduler=SampleScheduler(self.config.sample_rate_hz,snapshot.time_ns)
            return None
        scheduler=self._scheduler
        rng=scheduler.preview(snapshot.time_ns)
        history=[*self._history,snapshot]
        if rng.count>self.config.max_samples:
            raise ValueError('advance exceeds maximum audio samples; use smaller simulation steps')
        if rng.count==0:
            self._history=history;scheduler.commit(snapshot.time_ns,rng)
            return None
        times=np.array([s.time_ns/1e9 for s in history],dtype=np.float64)
        positions=np.array([s.array_position_m for s in history],dtype=np.float64)
        rotations=np.array([rotation_from_xyzw(s.array_orientation_xyzw) for s in history])
        array_trajectory=PoseTrajectory(times,positions,rotations)
        physical=[]
        for source in self.sources:
            src_positions=np.array([s.source_positions_m[source.source_id] for s in history],dtype=np.float64)
            physical.append(PhysicalSource(source.source_id,source.waveform,
                    PoseTrajectory(times,src_positions),start_s=source.start_ns/1e9,
                    amplitude=source.amplitude))
        pcm=self.renderer.render(self.array.positions_m,array_trajectory,physical,
                  rng.count/self.config.sample_rate_hz,
                  start_s=(scheduler.epoch_ns/1e9+rng.first/self.config.sample_rate_hz))
        if not np.isfinite(pcm).all() or np.max(np.abs(pcm))>1.0:
            raise ValueError('rendered PCM outside [-1,1]; reduce source levels, no silent clipping')
        packet=AudioPacket(pcm,self.config.sample_rate_hz,
                  CaptureInterval(self.clock_id,rng.start_ns,rng.end_ns),
                  self.array.array_id,self.array.frame_id,self.array.geometry_id,
                  sequence=self._sequence)
        if self.perception is None:
            spatial=SpatialResult(packet.capture,packet.frame_id,(),'no_perception')
        else:
            # Crucial security boundary: only PCM and fixed array calibration.
            if hasattr(self.perception,'estimate'):
                estimate=self.perception.estimate(packet,self.array)
                spatial=estimate.result if hasattr(estimate,'result') else estimate
            else:
                spatial=self.perception.process(packet,self.array)
            if not isinstance(spatial,SpatialResult) or spatial.capture!=packet.capture or spatial.frame_id!=packet.frame_id:
                raise ValueError('perception output has invalid capture or frame')
        result=RuntimeOutput(packet,HearingObservation(packet.capture,spatial),rng.first,rng.stop)
        self._history=history;scheduler.commit(snapshot.time_ns,rng);self._sequence+=1
        return result
