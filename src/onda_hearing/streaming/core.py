"""Onda continuous-PCM streaming reference backend (Python/NumPy).

Time domain: all capture and pose timestamps MUST share a monotonic nanosecond clock.
The audio producer never invokes acoustic inference; only the worker does.
This module intentionally does not import Isaac Lab or CUDA.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from threading import Event, Lock, Thread
from time import monotonic_ns, perf_counter_ns
from typing import Protocol
import math
import numpy as np


class StreamError(RuntimeError):
    """Invalid stream sequencing or unrecoverable worker failure."""


@dataclass(frozen=True)
class StreamConfig:
    sample_rate: int = 16000
    channels: int = 4
    window_samples: int = 2048
    hop_samples: int = 512
    buffer_seconds: float = 4.0
    max_pose_gap_ms: float = 150.0
    max_observation_age_ms: float = 300.0
    max_publish_age_ms: float = 750.0
    min_rms: float = 0.00002

    def __post_init__(self):
        if not 1000 <= self.sample_rate <= 192000 or not 2 <= self.channels <= 64:
            raise ValueError('invalid sample rate or channel count')
        if not 64 <= self.hop_samples <= self.window_samples:
            raise ValueError('require 64 <= hop_samples <= window_samples')
        if not (self.window_samples <= self.sample_rate * self.buffer_seconds):
            raise ValueError('audio buffer smaller than one window')
        if self.sample_rate*self.buffer_seconds*self.channels*4 > 256*1024*1024:
            raise ValueError('audio buffer exceeds 256 MiB memory safety cap')
        for field in ('max_pose_gap_ms','max_observation_age_ms','max_publish_age_ms'):
            if not math.isfinite(getattr(self,field)) or getattr(self,field) <= 0:
                raise ValueError(f'{field} must be positive and finite')
        if not math.isfinite(self.min_rms) or self.min_rms < 0:
            raise ValueError('min_rms must be finite and nonnegative')


@dataclass(frozen=True)
class Pose:
    time_ns: int
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        if not isinstance(self.time_ns, int) or self.time_ns < 0:
            raise ValueError('pose time must be nonnegative integer nanoseconds')
        p = np.asarray(self.position_xyz,dtype=float)
        q = np.asarray(self.quaternion_xyzw,dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('invalid pose')
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
            raise ValueError('quaternion must be unit XYZW')


def _slerp_xyzw(a, b, fraction):
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    dot = float(a @ b)
    if dot < 0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1., 1.))
    if dot > .9995:
        output = a + fraction * (b-a)
    else:
        theta = math.acos(dot)
        output = (math.sin((1-fraction)*theta)*a + math.sin(fraction*theta)*b)/math.sin(theta)
    return tuple((output/np.linalg.norm(output)).tolist())


def rotate_xyzw(q, v):
    """Rotate a 3D local vector into world space using XYZW quaternion."""
    x,y,z,w = map(float,q)
    v = np.asarray(v, dtype=float)
    xyz = np.array([x,y,z])
    return v + 2*w*np.cross(xyz,v) + 2*np.cross(xyz,np.cross(xyz,v))


class PoseHistory:
    def __init__(self, max_entries: int = 4096):
        if max_entries < 2:
            raise ValueError('max_entries must be >=2')
        self._poses = deque(maxlen=max_entries)
        self._lock = Lock()

    def add(self, pose: Pose):
        with self._lock:
            if self._poses and pose.time_ns <= self._poses[-1].time_ns:
                raise StreamError('pose timestamps must increase strictly')
            self._poses.append(pose)

    def at(self, time_ns: int, max_gap_ns: int) -> Pose | None:
        with self._lock:
            poses = tuple(self._poses)
        if not poses or time_ns < poses[0].time_ns or time_ns > poses[-1].time_ns:
            return None  # Never extrapolate or use future orientation.
        lo,hi=0,len(poses)-1
        while lo < hi:
            mid=(lo+hi)//2
            if poses[mid].time_ns < time_ns: lo=mid+1
            else: hi=mid
        b=poses[lo]
        if b.time_ns == time_ns: return b
        a=poses[lo-1]
        if b.time_ns-a.time_ns > max_gap_ns: return None
        alpha=(time_ns-a.time_ns)/(b.time_ns-a.time_ns)
        position=tuple((np.asarray(a.position_xyz)*(1-alpha)+np.asarray(b.position_xyz)*alpha).tolist())
        return Pose(time_ns, position, _slerp_xyzw(a.quaternion_xyzw,b.quaternion_xyzw,alpha))


@dataclass(frozen=True)
class PCMWindow:
    samples: np.ndarray  # [samples, channels], independent read-only copy
    start_sample: int
    end_sample: int
    capture_mid_ns: int
    epoch: int


class PCMBuffer:
    """Preallocated circular sample store: O(chunk) writes, O(window) reads.

    Old samples are overwritten ONLY when bounded capacity is exhausted. Every
    overwrite is counted and unread inference windows are explicitly skipped.
    Source sample indices remain absolute across buffer wraparounds.
    """
    def __init__(self, config: StreamConfig, capture_start_ns: int):
        if capture_start_ns < 0:
            raise ValueError('capture_start_ns must be nonnegative')
        self.config=config
        self.capture_start_ns=int(capture_start_ns)
        self.capacity=int(config.sample_rate*config.buffer_seconds)
        self._ring=np.empty((self.capacity,config.channels),dtype=np.float32)
        self._start=0
        self._end=0
        self.epoch=0
        self.overwritten_samples=0
        self.discontinuities=0
        self.ingested_samples=0
        self._lock=Lock()

    def time_ns(self, sample_index: int):
        return self.capture_start_ns + (int(sample_index)*1_000_000_000 + self.config.sample_rate//2)//self.config.sample_rate

    def append(self, pcm, start_sample: int):
        data=np.asarray(pcm)
        if data.ndim != 2 or data.shape[1] != self.config.channels or data.shape[0] == 0:
            raise ValueError('PCM must be nonempty [samples, channels]')
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError('PCM must be finite numeric samples')
        data=np.array(data,dtype=np.float32,order='C',copy=True)
        if data.shape[0] > self.capacity:
            raise ValueError('chunk exceeds ring capacity; split capture chunks')
        if not isinstance(start_sample,int) or start_sample < 0:
            raise ValueError('start_sample must be nonnegative integer')
        with self._lock:
            if start_sample < self._end:
                raise StreamError('duplicate/overlapping/backward audio samples')
            if start_sample != self._end:
                self.discontinuities += 1
                self.epoch += 1
                self._start=start_sample
                self._end=start_sample
            old_start=self._start
            slot=start_sample%self.capacity
            first=min(len(data),self.capacity-slot)
            self._ring[slot:slot+first]=data[:first]
            if first<len(data):
                self._ring[:len(data)-first]=data[first:]
            self._end=start_sample+len(data)
            self._start=max(self._start,self._end-self.capacity)
            self.overwritten_samples+=self._start-old_start
            self.ingested_samples+=len(data)

    def bounds(self):
        with self._lock:
            return self._start,self._end,self.epoch

    def window(self, end_sample: int) -> PCMWindow | None:
        with self._lock:
            start=end_sample-self.config.window_samples
            if start < self._start or end_sample > self._end:
                return None
            slot=start%self.capacity
            n=self.config.window_samples
            first=min(n,self.capacity-slot)
            if first==n:
                x=self._ring[slot:slot+n].copy()
            else:
                x=np.concatenate((self._ring[slot:slot+first],self._ring[:n-first]),axis=0)
            epoch=self.epoch
        x.setflags(write=False)
        return PCMWindow(x,start,end_sample,self.time_ns((start+end_sample)//2),epoch)


@dataclass(frozen=True)
class BearingEstimate:
    local_bearing_deg: float
    confidence: float
    valid: bool
    evidence: float = 0.0


class Estimator(Protocol):
    def estimate(self, pcm: np.ndarray) -> BearingEstimate: ...


class GCCPHATSRP:
    """Reference 2D far-field GCC-PHAT/SRP estimator for known planar mic geometry.

    Does not classify sound events, separate multiple concurrent sources, or model
    near-field/reverberation. For integration validation, not calibrated hardware.
    """
    def __init__(self, microphone_positions_m, sample_rate=16000, step_deg=2, sound_speed=343.0,
                 min_contrast=0.06, min_normalized_peak=0.08):
        xyz=np.asarray(microphone_positions_m,dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz)<2 or not np.isfinite(xyz).all():
            raise ValueError('microphone geometry must be finite [channels,3]')
        if not 1<=step_deg<=30 or sample_rate<1000 or sound_speed<=0:
            raise ValueError('invalid estimator configuration')
        self.positions=xyz.copy()
        self.sample_rate=sample_rate
        self.angles=np.deg2rad(np.arange(-180,180,step_deg,dtype=float))
        d=np.stack((np.cos(self.angles),np.sin(self.angles),np.zeros_like(self.angles)),axis=1)
        self.pairs=[(i,j) for i in range(len(xyz)) for j in range(i+1,len(xyz))]
        self.delays=np.stack([-(d @ (xyz[i]-xyz[j]))*sample_rate/sound_speed for i,j in self.pairs])
        if not (0 <= min_normalized_peak <= 1):
            raise ValueError("min_normalized_peak must be between 0 and 1")
        self.min_contrast=min_contrast
        self.min_normalized_peak=min_normalized_peak

    def estimate(self, pcm):
        data=np.asarray(pcm,dtype=np.float64)
        if data.ndim!=2 or data.shape[1]!=len(self.positions) or not np.isfinite(data).all():
            raise ValueError('invalid microphone PCM')
        data=data-np.mean(data,axis=0,keepdims=True)
        if np.max(np.sqrt(np.mean(data**2,axis=0))) < 1e-6:
            return BearingEstimate(0.,0.,False)
        taper=np.hanning(len(data))[:,None]
        nfft=1 << (2*len(data)-1).bit_length()  # unaliased linear GCC lags
        spectrum=np.fft.rfft(data*taper,n=nfft,axis=0)
        scores=np.zeros(len(self.angles))
        for pair,(i,j) in enumerate(self.pairs):
            cross=spectrum[:,i]*np.conj(spectrum[:,j])
            cross/=np.maximum(np.abs(cross),1e-12)
            gcc=np.fft.irfft(cross,n=nfft)
            tau=self.delays[pair]
            lower=np.floor(tau).astype(int)
            fraction=tau-lower
            # Modulo wrap maps negative lags onto circular FFT output.
            values=(1-fraction)*gcc[lower%nfft]+fraction*gcc[(lower+1)%nfft]
            scores+=values
        idx=int(np.argmax(scores))
        contrast=float((scores[idx]-np.median(scores))/(abs(scores[idx])+1e-9))
        confidence=float(np.clip(contrast,0.,1.))
        coherent_peak=float(scores[idx]/len(self.pairs))
        return BearingEstimate(float(np.rad2deg(self.angles[idx])),confidence,
                               bool(contrast>=self.min_contrast and coherent_peak>=self.min_normalized_peak),
                               float(scores[idx]))


@dataclass(frozen=True)
class HearingObservation:
    valid: bool = False
    bearing_local_deg: float = 0.
    bearing_world_deg: float = 0.
    confidence: float = 0.
    evidence: float = 0.
    capture_time_ns: int = 0
    publish_mono_ns: int = 0
    source_sample_start: int = -1
    source_sample_end: int = -1
    epoch: int = -1
    reason: str = 'not_ready'
    processing_ms: float = 0.


class StreamingHearing:
    """Single-producer PCM ingestion, independently scheduled estimator, nonblocking policy read.

    Under overload the worker skips redundant inference windows, NEVER splices
    discontinuous PCM. Retained PCM is lost only if bounded ring is overrun,
    which is surfaced as overwritten_samples. No hard real-time guarantee.
    """
    def __init__(self, config: StreamConfig, capture_start_ns: int, estimator: Estimator,
                 pose_history: PoseHistory | None = None, *, start_worker=True):
        self.config=config
        self.buffer=PCMBuffer(config,capture_start_ns)
        self.poses=pose_history if pose_history is not None else PoseHistory()
        self.estimator=estimator
        self._lock=Lock()
        self._wake=Event()
        self._stop=Event()
        self._worker: Thread | None=None
        self._next_end=config.window_samples
        self._cursor_epoch=0
        self._last=HearingObservation()
        self._failure: str | None = None
        self._windows_processed=0
        self._windows_skipped=0
        self._pose_misses=0
        self._overrun_recoveries=0
        if start_worker: self.start()

    def start(self):
        with self._lock:
            if self._worker is not None:
                raise StreamError('worker already started')
            if self._stop.is_set():
                raise StreamError('worker cannot restart after stop')
            self._worker=Thread(target=self._run,daemon=True,name='onda-streaming-pcm')
            self._worker.start()

    def push_audio(self, pcm, start_sample: int):
        if self._stop.is_set(): raise StreamError('stream closed')
        self.buffer.append(pcm,start_sample)
        self._wake.set()

    def push_pose(self, pose: Pose):
        if self._stop.is_set(): raise StreamError('stream closed')
        self.poses.add(pose)
        self._wake.set()

    def _one(self):
        first,last,epoch=self.buffer.bounds()
        if last-first < self.config.window_samples: return False
        if epoch != self._cursor_epoch:
            self._cursor_epoch=epoch
            self._next_end=first+self.config.window_samples
            with self._lock:
                self._last=HearingObservation(reason='audio_discontinuity',epoch=epoch)
        if self._next_end-self.config.window_samples < first:
            # Bounded ring has overwritten historical PCM; skip unavailable windows.
            new_end=first+self.config.window_samples
            skipped=max(1,(new_end-self._next_end+self.config.hop_samples-1)//self.config.hop_samples)
            self._next_end += skipped*self.config.hop_samples
            with self._lock:
                self._windows_skipped+=skipped
                self._overrun_recoveries+=1
        if self._next_end>last: return False
        # If behind, process the most recent complete *contiguous* window only.
        latest_end=self._next_end+((last-self._next_end)//self.config.hop_samples)*self.config.hop_samples
        if latest_end>self._next_end:
            with self._lock: self._windows_skipped+=(latest_end-self._next_end)//self.config.hop_samples
            self._next_end=latest_end
        window=self.buffer.window(self._next_end)
        if window is None: return False
        self._next_end+=self.config.hop_samples
        pose=self.poses.at(window.capture_mid_ns,int(self.config.max_pose_gap_ms*1e6))
        if pose is None:
            with self._lock: self._pose_misses+=1
            return True  # No output without pose alignment.
        start=perf_counter_ns()
        rms=float(np.sqrt(np.mean(np.square(window.samples,dtype=np.float64))))
        if rms<self.config.min_rms:
            estimate=BearingEstimate(0.,0.,False)
            reason='silence'
        else:
            estimate=self.estimator.estimate(window.samples)
            reason='ok' if estimate.valid else 'no_detection'
        local=float(estimate.local_bearing_deg)
        if not math.isfinite(local) or not math.isfinite(estimate.confidence):
            raise StreamError('estimator generated nonfinite result')
        direction=rotate_xyzw(pose.quaternion_xyzw, (math.cos(math.radians(local)),math.sin(math.radians(local)),0.))
        world=(math.degrees(math.atan2(direction[1],direction[0]))+180)%360-180
        result=HearingObservation(bool(estimate.valid),local,float(world),
                                  float(np.clip(estimate.confidence,0.,1.)),float(estimate.evidence),
                                  window.capture_mid_ns,monotonic_ns(),window.start_sample,
                                  window.end_sample,window.epoch,reason,(perf_counter_ns()-start)/1e6)
        with self._lock:
            if window.epoch == self.buffer.bounds()[2] and not self._stop.is_set():
                self._last=result
                self._windows_processed+=1
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.wait(.05)
                self._wake.clear()
                # Work at most once per wake; newest-window coalescing prevents catch-up storms.
                self._one()
        except BaseException as exc:
            with self._lock:
                self._failure=f'{type(exc).__name__}: {exc}'
                self._last=HearingObservation(reason='worker_error')
            self._stop.set()

    def poll(self, now_capture_ns: int, *, now_mono_ns: int | None = None):
        """Never calls inference; returns a fresh validity decision and telemetry.

        now_capture_ns is the latest source-clock time, NOT wall time. The caller
        must supply a matching monotonically increasing capture timebase.
        """
        if now_mono_ns is None: now_mono_ns=monotonic_ns()
        with self._lock:
            result=self._last
            metrics={'windows_processed':self._windows_processed,'inference_windows_skipped':self._windows_skipped,
                     'pose_misses':self._pose_misses,'ring_overrun_recoveries':self._overrun_recoveries,
                     'worker_error':self._failure,'worker_alive':bool(self._worker and self._worker.is_alive())}
        begin,end,epoch=self.buffer.bounds()
        metrics.update(buffer_oldest_sample=begin,buffer_next_sample=end,audio_epoch=epoch,
                       overwritten_audio_samples=self.buffer.overwritten_samples,
                       audio_discontinuities=self.buffer.discontinuities,
                       ingested_audio_samples=self.buffer.ingested_samples)
        if self._failure: return replace(result,valid=False,reason='worker_error'),metrics
        if not result.valid: return result,metrics
        if result.epoch!=epoch:
            return replace(result,valid=False,reason='audio_epoch_changed'),metrics
        age_ms=(now_capture_ns-result.capture_time_ns)/1e6
        publication_age_ms=(now_mono_ns-result.publish_mono_ns)/1e6
        metrics.update(capture_age_ms=age_ms,publish_age_ms=publication_age_ms)
        if age_ms<0 or age_ms>self.config.max_observation_age_ms or publication_age_ms<0 or publication_age_ms>self.config.max_publish_age_ms:
            return replace(result,valid=False,reason='stale'),metrics
        return result,metrics

    def policy_features(self, now_capture_ns: int):
        """[sin(local), cos(local), confidence, valid, age_s] float32 + metadata."""
        obs,metrics=self.poll(now_capture_ns)
        features=np.zeros(5,dtype=np.float32)
        if obs.valid:
            angle=math.radians(obs.bearing_local_deg)
            features[:]=[math.sin(angle),math.cos(angle),obs.confidence,1.,
                         max(0,(now_capture_ns-obs.capture_time_ns)/1e9)]
        return features,obs,metrics

    def close(self, timeout_s=5.):
        if timeout_s<0: raise ValueError('timeout_s must be nonnegative')
        self._stop.set();self._wake.set()
        with self._lock:
            self._last=HearingObservation(reason='closed')
        if self._worker: self._worker.join(timeout_s)
        return not (self._worker and self._worker.is_alive())
