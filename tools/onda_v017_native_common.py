"""Shared native integration verification. Never leaks simulator truth to policies."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
import numpy as np

MIC_LOCAL = ((.05,.03,0.),(-.045,.035,0.),(-.05,-.038,0.),(.042,-.047,0.))

def angle_error(a:float,b:float)->float:
    return abs((float(a)-float(b)+180.)%360.-180.)

def truth_bearing(array_pose,source_pose)->float:
    d=np.asarray(source_pose.xyz)-np.asarray(array_pose.xyz)
    return math.degrees(math.atan2(float(d[1]),float(d[0])))

def finite_policy(x,channels=4,semantics=527):
    a=np.asarray(x)
    if a.shape!=(7+semantics,) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise AssertionError(f'invalid policy output shape/dtype/values {a.shape} {a.dtype}')

def atomic_json(path:Path,data:dict):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf8') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)

def quantiles(values):
    return {key:float(np.percentile(values,p)) if values else None for key,p in [('p50',50),('p75',75),('p95',95),('p99',99)]}

class Accumulator:
    def __init__(self):
        self.spatial=0;self.semantic=0;self.errors=[];self.labels={}
        self.spatial_unique=set();self.semantic_unique=set();self.policy_read_ms=[]
        self.max_semantic_score=0.;self.association_claimed=False;self.last_policy_vector=None
    def observe(self,hearing,clock_ns,ground_truth,expected_dim):
        t=time.perf_counter_ns()
        features,spatial,semantic,metrics=hearing.poll(clock_ns)
        self.policy_read_ms.append((time.perf_counter_ns()-t)/1e6)
        a=np.asarray(features)
        if a.shape!=(expected_dim,) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise AssertionError(f'bad policy vector {a.shape} {a.dtype}')
        self.last_policy_vector=a.copy()
        if spatial.valid:
            self.spatial+=1
            if spatial.source_sample_end not in self.spatial_unique:
                self.spatial_unique.add(spatial.source_sample_end)
                truth=ground_truth(spatial.capture_time_ns)
                if truth is not None:self.errors.append(angle_error(spatial.bearing_world_deg,truth))
        if semantic.valid:
            self.semantic+=1
            if semantic.end_sample not in self.semantic_unique:
                self.semantic_unique.add(semantic.end_sample)
                for ev in semantic.events:
                    self.labels[ev.label]=max(self.labels.get(ev.label,0.),float(ev.score))
                    self.max_semantic_score=max(self.max_semantic_score,float(ev.score))
        return spatial,semantic,metrics
    def report(self):
        return {'valid_spatial_reads':self.spatial,'unique_spatial_windows':len(self.spatial_unique),
                'valid_semantic_reads':self.semantic,'unique_semantic_windows':len(self.semantic_unique),
                'angular_error_deg':quantiles(self.errors),
                'top_labels':sorted(self.labels.items(),key=lambda x:-x[1])[:8],
                'policy_read_ms':quantiles(self.policy_read_ms)}

def make_hearing(brain,array_mics,sample_rate=16000,origin_ns=0):
    from onda_hearing.streaming.core import StreamConfig, StreamingHearing,GCCPHATSRP
    from onda_hearing.semantics import attach_sound_brain
    cfg=StreamConfig(sample_rate=sample_rate,channels=len(array_mics),window_samples=2048,
        hop_samples=512,buffer_seconds=4.,max_pose_gap_ms=130.,max_observation_age_ms=500.,
        max_publish_age_ms=1500.)
    spatial=StreamingHearing(cfg,origin_ns,GCCPHATSRP(np.array(array_mics)))
    try:
        unified=attach_sound_brain(spatial,brain=brain,vocabulary=tuple(brain.labels),
            semantic_window_s=2.,semantic_hop_s=.5,semantic_buffer_s=4.)
    except BaseException:
        spatial.close();raise
    return unified

def make_brain(args):
    from onda_hearing.semantics import OndaSoundBrain
    return OndaSoundBrain(checkpoint_path=args.checkpoint,labels_path=args.labels,
        beats_module_dir=args.beats_dir,device=args.device,allow_unsafe_checkpoint=True,
        min_score=0.,top_k=5)

def validate_metrics(metrics,acc,min_spatial=10,require_semantic=True,max_p75=15.,max_policy_p99_ms=10.):
    reasons=[]
    if len(acc.spatial_unique)<min_spatial:reasons.append('insufficient_unique_spatial_windows')
    err=quantiles(acc.errors)['p75']
    if err is None or err>max_p75:reasons.append(f'bearing_p75_error>{max_p75}: {err}')
    if require_semantic and len(acc.semantic_unique)<1:reasons.append('no_valid_semantic_window')
    if metrics.get('worker_error') or metrics.get('semantic_worker_error'):reasons.append('worker_error')
    for key in ('audio_discontinuities','ring_overrun_recoveries','pose_misses'):
        if metrics.get(key,0):reasons.append(f'{key}={metrics[key]}')
    p99=quantiles(acc.policy_read_ms)['p99']
    if p99 is None or p99>max_policy_p99_ms:reasons.append('policy_read_p99_exceeded')
    return reasons
