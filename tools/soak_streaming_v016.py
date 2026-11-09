#!/usr/bin/env python3
"""Onda v0.16 sustained wall-clock-paced audio/pose streaming acceptance.

Default 1800 s means **1800 actual wall-clock seconds**, NOT accelerated sim.
Native Isaac Lab validation is a separate runner. Physics and audio propagation
are not simulated here; PCM uses an explicit analytic source model.

This is a bounded-memory test of the current prototype, not real hardware or
hard-real-time certification. Writes checkpoint JSONL and atomic final JSON.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

from onda_hearing.streaming import StreamConfig, Pose, GCCPHATSRP, StreamingHearing
from onda_hearing.streaming.simulated_capture import DEFAULT_ARRAY_METERS
from onda_soak_common import (Counter, CyclicSource, LatencyHistogram, Checkpoints,
                              angle_error, atomic_json, rss_bytes, sample_timestamp_ns,
                              yaw_at_seconds, yaw_xyzw, NS)


class DelayedEstimator:
    """Explicit, observable inference overload; never sleeps in audio ingestion."""
    def __init__(self, wrapped, *, begin_mono_ns: int, duration_s: float,
                 extra_ms: float, enabled: bool):
        self.wrapped = wrapped
        self.begin = begin_mono_ns
        self.duration = duration_s
        self.delay_ms = extra_ms
        self.enabled = enabled
        self.injected = 0
        self._lock = threading.Lock()

    def estimate(self, pcm):
        elapsed = (time.monotonic_ns() - self.begin)/NS
        # Two short, clearly identified overload intervals across the test.
        span = min(8.0, max(0.75, self.duration * 0.025))
        overloaded = self.enabled and any(
            a*self.duration <= elapsed < a*self.duration+span for a in (0.25, 0.65)
        )
        if overloaded:
            time.sleep(self.delay_ms/1000)
            with self._lock:
                self.injected += 1
        return self.wrapped.estimate(pcm)


def run(args):
    if args.duration_s <= 0 or args.chunk_ms <= 0 or args.overload_ms < 0:
        raise ValueError('duration, chunk, and overload must be positive/nonnegative')
    cfg = StreamConfig(sample_rate=16000, channels=4, window_samples=2048,
                       hop_samples=512, buffer_seconds=4., max_pose_gap_ms=120.,
                       max_observation_age_ms=300., max_publish_age_ms=750.)
    frames_per_chunk = int(cfg.sample_rate*args.chunk_ms/1000)
    if frames_per_chunk < 64 or frames_per_chunk * 1000 != int(cfg.sample_rate*args.chunk_ms):
        raise ValueError('chunk duration must produce a whole number >=64 of PCM frames')
    total_chunks = math.ceil(args.duration_s * cfg.sample_rate / frames_per_chunk)
    origin = time.monotonic_ns() + 100_000_000
    fixed_source = np.array([1.5*math.cos(math.radians(38)),
                             1.5*math.sin(math.radians(38)),0.],dtype=np.float64)
    source = CyclicSource()
    estimator = DelayedEstimator(GCCPHATSRP(DEFAULT_ARRAY_METERS),
                                 begin_mono_ns=origin,duration_s=args.duration_s,
                                 extra_ms=args.overload_ms, enabled=not args.no_overload)
    stream = StreamingHearing(cfg, capture_start_ns=origin,estimator=estimator)
    stream.push_pose(Pose(origin,(0.,0.,0.),yaw_xyzw(0.)))
    stats = Counter()
    capture_publish = LatencyHistogram()
    publish_read = LatencyHistogram()
    read_latency = LatencyHistogram()
    ingest_latency = LatencyHistogram()
    render_latency = LatencyHistogram()
    reader_deadline_misses = 0
    producer_deadline_misses = 0
    reader_exception = []
    finish = threading.Event()
    progress = Checkpoints(args.output.with_suffix('.checkpoints.jsonl'))
    checkpoint_count = 0
    max_read_lag_ms = 0.0
    rss_start = rss_bytes()
    rss_warmup = None
    rss_peak_after_warmup = rss_start
    accepted_capture_latency = LatencyHistogram()

    def reader():
        nonlocal reader_deadline_misses,max_read_lag_ms
        period_ns = round(NS / 60)
        tick = 0
        try:
            while not finish.is_set():
                goal = origin + tick*period_ns
                now = time.monotonic_ns()
                if now < goal:
                    finish.wait((goal-now)/NS)
                if finish.is_set():
                    break
                actual = time.monotonic_ns()
                if actual-goal > 2*period_ns:
                    reader_deadline_misses += 1
                    # Rejoin wall clock; never silently fabricate intermediate reads.
                    tick = max(tick, (actual-origin)//period_ns)
                tick += 1
                started = time.perf_counter_ns()
                features,obs,telem = stream.policy_features(actual)
                read_latency.add((time.perf_counter_ns()-started)/1e6)
                if not np.isfinite(features).all() or features.shape!=(5,):
                    raise AssertionError('policy features not finite [5]')
                if not obs.valid and np.any(features):
                    raise AssertionError('invalid/stale hearing leaked to policy')
                if obs.valid and (features[3]<.5 or obs.capture_time_ns>actual):
                    raise AssertionError('invalid feature validity or future capture timestamp')
                fresh = obs.valid and obs.source_sample_end != stats.last_seen_sample_end
                stats.track(obs,38.,args.max_error_deg)
                if fresh:
                    published_latency = (obs.publish_mono_ns-obs.capture_time_ns)/1e6
                    read_delay = (actual-obs.publish_mono_ns)/1e6
                    capture_publish.add(published_latency)
                    publish_read.add(max(0.,read_delay))
                    accepted_capture_latency.add((actual-obs.capture_time_ns)/1e6)
                if 'capture_age_ms' in telem:
                    max_read_lag_ms = max(max_read_lag_ms,telem['capture_age_ms'])
        except BaseException as exc:
            reader_exception.append(f'{type(exc).__name__}: {exc}')
            finish.set()

    thread=threading.Thread(target=reader,name='onda-soak-policy-reader',daemon=True)
    thread.start()
    exit_reason='completed'
    last_progress_sec=-1
    try:
        for k in range(total_chunks):
            if finish.is_set():
                exit_reason='reader_failed'
                break
            first=k*frames_per_chunk
            end=min((k+1)*frames_per_chunk,round(args.duration_s*cfg.sample_rate))
            frames=end-first
            target=sample_timestamp_ns(origin,end,cfg.sample_rate)
            now=time.monotonic_ns()
            if now < target:
                time.sleep((target-now)/NS)
            else:
                if now-target > 2*frames_per_chunk*NS//cfg.sample_rate:
                    producer_deadline_misses+=1
            ts = sample_timestamp_ns(origin,end,cfg.sample_rate)
            seconds=(first+end)/(2*cfg.sample_rate)
            yaw=yaw_at_seconds(seconds)
            stream.push_pose(Pose(ts,(0.,0.,0.),yaw_xyzw(yaw_at_seconds(end/cfg.sample_rate))))
            render_started=time.perf_counter_ns()
            pcm=source.render(first,frames,cfg.sample_rate,(0.,0.,0.),
                              yaw_xyzw(yaw), fixed_source, DEFAULT_ARRAY_METERS)
            render_latency.add((time.perf_counter_ns()-render_started)/1e6)
            ingest_started=time.perf_counter_ns()
            stream.push_audio(pcm,first)
            ingest_latency.add((time.perf_counter_ns()-ingest_started)/1e6)
            elapsed=end/cfg.sample_rate
            checkpoint_index=int(elapsed//args.checkpoint_s)
            if checkpoint_index>checkpoint_count or k==total_chunks-1:
                checkpoint_count=checkpoint_index
                obs,met=stream.poll(time.monotonic_ns())
                rss=rss_bytes()
                if rss_warmup is None and elapsed>=min(5,args.duration_s/4):rss_warmup=rss
                if rss is not None and rss_warmup is not None:
                    rss_peak_after_warmup=max(rss_peak_after_warmup or 0,rss)
                item={'elapsed_s':round(elapsed,3), 'elapsed_wall_s':round((time.monotonic_ns()-origin)/NS,3),
                      'rss_bytes':rss, 'reader_reads':stats.total,'reader_valid':stats.valid,
                      'unique_valid':stats.unique_valid,'capture_publish_p99_ms':capture_publish.percentile(99),
                      'read_p99_ms':read_latency.percentile(99),
                      'producer_deadline_misses':producer_deadline_misses,
                      'reader_deadline_misses':reader_deadline_misses,
                      'injected_slow_inferences':estimator.injected,
                      'buffer':{key:met.get(key) for key in ('ingested_audio_samples','audio_discontinuities',
                                 'overwritten_audio_samples','windows_processed','inference_windows_skipped',
                                 'pose_misses','worker_error','ring_overrun_recoveries')}}
                progress.append(item)
                print('ONDA SOAK: checkpoint',item,flush=True)
        # Let all producer-side chunks finish; no artificial catch-up delay in latency metrics.
        finish.set()
        thread.join(timeout=3.)
        obs,telemetry=stream.poll(time.monotonic_ns())
        result={'suite':'Onda v0.16 continuous PCM wall-clock soak','result':'FAIL',
                'host_clock':'CLOCK_MONOTONIC via time.monotonic_ns',
                'duration_requested_s':args.duration_s,
                'actual_wall_duration_s':round((time.monotonic_ns()-origin)/NS,3),
                'audio_duration_s':round(telemetry['ingested_audio_samples']/cfg.sample_rate,3),
                'stream_config':{'sample_rate':cfg.sample_rate,'channels':cfg.channels,
                                 'window_samples':cfg.window_samples,'hop_samples':cfg.hop_samples,
                                 'buffer_seconds':cfg.buffer_seconds},
                'input':{'source':'bounded periodic synthetic broadband waveform',
                         'direction_world_deg':38.,'yaw_amplitude_deg':50.,'yaw_period_s':24.,
                         'paced_to_actual_wall_clock':True,'duration_is_not_accelerated':True},
                'overload':{'enabled':not args.no_overload,'injected_delay_ms':args.overload_ms,
                            'injected_inference_calls':estimator.injected,
                            'inference_windows_skipped_expected_during_overload':True},
                'reads':vars(stats),
                'timing':{'capture_to_publication':capture_publish.summary(),
                          'capture_to_first_valid_read':accepted_capture_latency.summary(),
                          'publication_to_read':publish_read.summary(),
                          'nonblocking_policy_read':read_latency.summary(),
                          'pcm_ingestion':ingest_latency.summary(),
                          'analytic_pcm_render':render_latency.summary()},
                'deadline_misses':{'audio_producer':producer_deadline_misses,'policy_reader':reader_deadline_misses},
                'memory':{'rss_start_bytes':rss_start,'rss_warmup_bytes':rss_warmup,
                          'rss_peak_after_warmup_bytes':rss_peak_after_warmup,
                          'rss_growth_after_warmup_bytes':(rss_peak_after_warmup-rss_warmup)
                          if rss_warmup is not None and rss_peak_after_warmup is not None else None},
                'telemetry':telemetry,'reader_error':reader_exception,
                'exit_reason':exit_reason,'limitations':[
                    'Synthetic coherent audio, not hardware microphone capture.',
                    'Reference single-source planar localizer, not full semantic/multi-source stack.',
                    'Python threads are not hard real time; capture callback jitter is visible.',
                    'Latency histogram uses 0.1ms bins; values above 5s counted in overflow.']}
        failures=[]
        if exit_reason!='completed' or reader_exception:failures.append('reader_failure')
        if result['audio_duration_s'] < args.duration_s-0.001:failures.append('incomplete_audio_duration')
        if telemetry['audio_discontinuities']!=0:failures.append('audio_discontinuity')
        # Old PCM eviction after the configured retention interval is normal.
        # Data loss is indicated by unread-window overrun recovery, not eviction alone.
        if telemetry['ring_overrun_recoveries']!=0:failures.append('unread_audio_overrun')
        if telemetry['worker_error'] is not None:failures.append('inference_worker_failure')
        if telemetry['pose_misses']!=0:failures.append('pose_miss')
        if stats.unique_valid<max(5,int(args.duration_s*0.4)):failures.append('insufficient_unique_valid_bearings')
        if stats.bearing_errors>max(1,int(stats.unique_valid*.05)):
            failures.append('too_many_angular_errors')
        if not args.no_overload and estimator.injected==0:failures.append('overload_not_exercised')
        if capture_publish.count and capture_publish.percentile(99)>args.max_capture_publish_p99_ms:
            failures.append('capture_to_publish_p99_exceeded')
        if read_latency.count and read_latency.percentile(99)>args.max_policy_read_p99_ms:
            failures.append('policy_read_p99_exceeded')
        growth=result['memory']['rss_growth_after_warmup_bytes']
        if growth is not None and growth>args.max_rss_growth_mb*1024*1024:
            failures.append('rss_growth_exceeded')
        if reader_deadline_misses>max(3,stats.total*args.max_deadline_miss_rate):failures.append('policy_deadline_misses')
        if producer_deadline_misses>max(3,total_chunks*args.max_deadline_miss_rate):failures.append('capture_deadline_misses')
        result['failures']=failures
        result['result']='PASS' if not failures else 'FAIL'
        return result
    finally:
        finish.set()
        thread.join(timeout=3.)
        progress.close()
        clean=stream.close(timeout_s=5.)
        if 'result' in locals():
            result['worker_clean_shutdown']=clean
            if not clean:
                result['result']='FAIL'
                result.setdefault('failures',[]).append('unclean_worker_shutdown')
        if not clean:print('ONDA SOAK: WARNING worker did not stop',file=sys.stderr,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration-s',type=float,default=1800.)
    p.add_argument('--chunk-ms',type=float,default=20.)
    p.add_argument('--overload-ms',type=float,default=95.)
    p.add_argument('--no-overload',action='store_true')
    p.add_argument('--checkpoint-s',type=float,default=60.)
    p.add_argument('--max-error-deg',type=float,default=10.)
    p.add_argument('--max-capture-publish-p99-ms',type=float,default=300.)
    p.add_argument('--max-policy-read-p99-ms',type=float,default=10.)
    p.add_argument('--max-rss-growth-mb',type=float,default=64.)
    p.add_argument('--max-deadline-miss-rate',type=float,default=.05)
    p.add_argument('--output',type=Path,default=Path('docs/validation/streaming_v016_soak/wall_clock_soak.json'))
    args=p.parse_args()
    if not 0 < args.duration_s <= 86400 or args.checkpoint_s <= 0:
        p.error('duration must be in (0,86400], checkpoint must be positive')
    try:
        data=run(args)
    except BaseException as e:
        data={'suite':'Onda v0.16 continuous PCM wall-clock soak','result':'FAIL',
              'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}
    atomic_json(args.output,data)
    print('ONDA SOAK: SAVED',args.output,flush=True)
    print('ONDA SOAK: FINAL VERDICT',data['result'],data.get('failures',''),flush=True)
    return 0 if data['result']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
