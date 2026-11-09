"""Pluggable, independently scheduled sound recognition for Onda PCM streaming.

Never equates a semantic label with a localized source without association.
A sound label applies to the WHOLE analysis window, not a specific bearing.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from threading import Thread, Event, Lock
from time import monotonic_ns
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class EventScore:
    label: str
    score: float
    def __post_init__(self):
        if not isinstance(self.label,str) or not self.label.strip() or not np.isfinite(self.score) or not 0<=self.score<=1:
            raise ValueError('invalid semantic event')

class SoundBrain(Protocol):
    name: str
    def classify(self, mono:np.ndarray, sample_rate:int)->tuple[EventScore,...]: ...

@dataclass(frozen=True)
class SemanticObservation:
    valid:bool=False
    events:tuple[EventScore,...]=()
    brain_name:str=''
    capture_mid_ns:int=0
    publish_mono_ns:int=0
    start_sample:int=-1
    end_sample:int=-1
    epoch:int=-1
    reason:str='not_ready'

class AsyncSemanticHearing:
    """Own bounded v0.16 PCMBuffer configured for semantic-length windows.

    It MUST NOT reuse the spatial PCMBuffer: v0.16 PCMBuffer.window() always
    returns `buffer.config.window_samples`, so sharing it would silently feed
    2048-sample chunks instead of the required 1-2 second semantic frames.
    Producer duplicates PCM to both bounded rings. All model inference runs on
    this worker; stale windows are skipped. No inference on policy thread.
    """
    def __init__(self,pcm_buffer,brain:SoundBrain,*,window_samples=16000,
                 hop_samples=8000,min_rms=1e-5,max_age_ms=2000.,
                 reference_channel:int=0,start_worker=True):
        if not 1<=hop_samples<=window_samples<=pcm_buffer.capacity:
            raise ValueError('semantic window/hop must fit PCM ring')
        if window_samples != pcm_buffer.config.window_samples:
            raise ValueError('semantic PCMBuffer must be configured with matching window_samples')
        if not 0<=reference_channel<pcm_buffer.config.channels:
            raise ValueError('invalid semantic reference channel')
        self.reference_channel=int(reference_channel)
        self.buffer=pcm_buffer;self.brain=brain
        self.window_samples=int(window_samples);self.hop_samples=int(hop_samples)
        self.min_rms=float(min_rms);self.max_age_ms=float(max_age_ms)
        if not 0<=min_rms<1 or not 0<max_age_ms<1e7:
            raise ValueError('invalid semantic thresholds')
        self._next_end=self.window_samples
        self._epoch=0;self._latest=SemanticObservation()
        self._lock=Lock();self._wake=Event();self._stop=Event();self._thread=None
        self._error=None;self.windows_processed=0;self.windows_skipped=0
        if start_worker:self.start()

    def start(self):
        if self._thread is not None or self._stop.is_set(): raise RuntimeError('semantic worker cannot restart')
        self._thread=Thread(target=self._run,daemon=True,name='onda-semantic-brain')
        self._thread.start()

    def notify_audio(self): self._wake.set()

    def _step(self):
        oldest,newest,epoch=self.buffer.bounds()
        if epoch!=self._epoch:
            self._epoch=epoch;self._next_end=oldest+self.window_samples
            with self._lock:self._latest=SemanticObservation(reason='audio_epoch_changed',epoch=epoch)
        if newest-oldest<self.window_samples:return False
        if self._next_end-self.window_samples<oldest:
            self._next_end=oldest+self.window_samples
        if newest<self._next_end:return False
        end=self._next_end+((newest-self._next_end)//self.hop_samples)*self.hop_samples
        with self._lock:
            self.windows_skipped+=max(0,(end-self._next_end)//self.hop_samples)
        window=self.buffer.window(end)
        self._next_end=end+self.hop_samples
        if window is None:return False
        # Do not average spatially separated mics: their phase offsets can cancel sound.
        # Mono BEATs takes one *actual* microphone channel by default.
        pcm=np.ascontiguousarray(window.samples[:,self.reference_channel],dtype=np.float32)
        if float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))>=self.min_rms:
            scores=self.brain.classify(pcm,self.buffer.config.sample_rate)
            if not isinstance(scores,(tuple,list)) or any(not isinstance(x,EventScore) for x in scores):
                raise TypeError('brain must return sequence[EventScore]')
            scores=tuple(sorted(scores,key=lambda x:-x.score))
            valid=bool(scores)
            reason='ok' if valid else 'no_event'
        else:
            scores=();valid=False;reason='silence'
        result=SemanticObservation(valid,scores,self.brain.name,window.capture_mid_ns,
                                   monotonic_ns(),window.start_sample,window.end_sample,window.epoch,reason)
        with self._lock:
            if window.epoch==self.buffer.bounds()[2] and not self._stop.is_set():
                self._latest=result;self.windows_processed+=1
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.wait(.05);self._wake.clear()
                self._step()
        except BaseException as err:
            with self._lock:
                self._error=f'{type(err).__name__}: {err}'
                self._latest=SemanticObservation(reason='brain_error')
            self._stop.set()

    def poll(self,now_capture_ns:int):
        with self._lock:
            obs=self._latest;error=self._error
            processed=self.windows_processed;skipped=self.windows_skipped
        _,_,epoch=self.buffer.bounds()
        metrics={'semantic_windows_processed':processed,
                 'semantic_windows_skipped':skipped,
                 'semantic_worker_error':error,
                 'semantic_worker_alive':bool(self._thread and self._thread.is_alive())}
        if error:return replace(obs,valid=False,reason='brain_error'),metrics
        if obs.epoch!=epoch:return replace(obs,valid=False,reason='epoch_changed'),metrics
        if not obs.valid:return obs,metrics
        age_ms=(now_capture_ns-obs.capture_mid_ns)/1e6
        pub_age_ms=(monotonic_ns()-obs.publish_mono_ns)/1e6
        if age_ms<0 or age_ms>self.max_age_ms or pub_age_ms<0 or pub_age_ms>max(self.max_age_ms,2000.):
            return replace(obs,valid=False,reason='stale'),metrics
        return obs,metrics

    def close(self,timeout_s=5.):
        self._stop.set();self._wake.set()
        if self._thread:self._thread.join(timeout_s)
        return not(self._thread and self._thread.is_alive())


class RuleBasedTestBrain:
    """TEST ONLY: simple energy-based adapter, never advertised as a classifier."""
    name='test-energy-brain'
    def classify(self,mono,sample_rate):
        energy=float(np.sqrt(np.mean(np.asarray(mono,dtype=np.float64)**2)))
        return (EventScore('test_non_silent',min(1.,energy*10)),) if energy>1e-5 else ()
