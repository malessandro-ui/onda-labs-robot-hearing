"""Run CPU reference profiles without Kit; no benchmark score is fabricated.

From Onda repo: python tools/native_isaaclab_realtime/benchmark_acoustic.py
Results are comparable only when executed on the same host/settings.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.runtime import HearingRuntime, ScheduledSource
from onda_hearing.policy import WindowedHearingPipeline, WindowedPipelineConfig
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig
from onda_hearing.simulators.isaac_lab_runtime import IsaacLabAcousticConfig, IsaacLabHearingManager

POSITIONS = [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]]
STEP_NS = 16_666_667
PROFILES = {
    'baseline': dict(period_ms=100, window_samples=4096, capacity_samples=8192,
                     angular_resolution_deg=2, block_samples=320, hop_samples=160),
    # Faster candidate, not automatically a recommended production default.
    'balanced_candidate': dict(period_ms=200, window_samples=2048, capacity_samples=4096,
                              angular_resolution_deg=4, block_samples=320, hop_samples=160),
    'lower_rate_candidate': dict(period_ms=250, window_samples=2048, capacity_samples=4096,
                                angular_resolution_deg=4, block_samples=320, hop_samples=160),
}


def summarize(values):
    if not values:
        return {'count': 0, 'p50_ms': None, 'p95_ms': None}
    return {'count': len(values), 'p50_ms': float(np.percentile(values,50)),
            'p95_ms': float(np.percentile(values,95))}


def evaluate(profile, audio, *, steps, seed):
    array = MicrophoneArray.from_positions(POSITIONS, array_id='head_array',
              frame_id='robot/head', geometry_id='asym90mm_v015')
    def factory(i):
        runtime = HearingRuntime(array, (ScheduledSource('sound', audio, 0),), sample_rate_hz=16000)
        selector = AcousticOnlySelector(MotionAwareSRPLocalizer(MotionSRPConfig(
            block_samples=profile['block_samples'], hop_samples=profile['hop_samples'],
            angular_resolution_deg=profile['angular_resolution_deg'], max_candidates=3,
            min_relative_score=.65, min_prominence=.012)),
            AcousticSelectionConfig(min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
        cfg = WindowedPipelineConfig(
            perception_period_ns=profile['period_ms']*1_000_000,
            window_samples=profile['window_samples'],
            capacity_samples=profile['capacity_samples'])
        return WindowedHearingPipeline(runtime, selector, config=cfg)
    manager = IsaacLabHearingManager(IsaacLabAcousticConfig(2, STEP_NS), factory)
    p = np.array([[0.,0.,0.],[4.,0.,0.]], dtype=np.float64)
    q = np.array([[1.,0.,0.,0.],[1.,0.,0.,0.]])
    bearings = np.radians([38.,-38.])
    source = p.copy()
    source[:,0] += 1.5*np.cos(bearings)
    source[:,1] += 1.5*np.sin(bearings)
    poses = (p,q,{'sound':source})
    manager.seed(*poses)
    times, errors = [], [[],[]]
    valid = [0,0]
    decision_counts = [0,0]
    for j in range(steps):
        start = time.perf_counter_ns()
        f = manager.advance(*poses)
        times.append((time.perf_counter_ns() - start)/1e6)
        for i in range(2):
            if manager.frames()[i] is not None and manager.frames()[i].new_decision:
                decision_counts[i] += 1
            if j > steps//2 and f[i,3] > 0.5:
                valid[i] += 1
                estimate = math.degrees(math.atan2(float(f[i,0]),float(f[i,1])))
                error = abs((estimate - math.degrees(bearings[i]) + 180.)%360-180.)
                errors[i].append(error)
    return {
        'profile':profile, 'timing':summarize(times),
        'valid_second_half': valid,
        'new_decisions':[int(x) for x in decision_counts],
        'p75_angular_error_deg':[float(np.percentile(e,75)) if e else None for e in errors],
        'accuracy_pass':all(len(e)>=5 and np.percentile(e,75)<=10 for e in errors)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--output',default='docs/validation/native_isaaclab_realtime_v015/profiles.json')
    args=parser.parse_args()
    if args.steps < 60:
        parser.error('--steps must be >=60 for sufficient acoustic decisions')
    rng=np.random.default_rng(42)
    audio=(.2*rng.standard_normal(16000*6)).astype(np.float32)
    output={'suite':'Onda v0.15 acoustic CPU profile comparison',
            'note':'CPU/pose staging + renderer + SRP + selector; no Kit. Candidate settings are not production defaults.',
            'profiles':{}}
    for name, cfg in PROFILES.items():
        print('ONDA PROFILE:',name,flush=True)
        result=evaluate(cfg,audio,steps=args.steps,seed=42)
        output['profiles'][name]=result
        print(json.dumps({'name':name,**result}),flush=True)
    base=output['profiles']['baseline']
    output['candidate_assessment']={}
    for name in PROFILES:
        if name=='baseline':continue
        item=output['profiles'][name]
        base_ms=base['timing']['p50_ms']
        cand_ms=item['timing']['p50_ms']
        output['candidate_assessment'][name]={
            'relative_p50_speedup':base_ms/cand_ms if cand_ms>0 else None,
            'eligible_for_further_validation': bool(item['accuracy_pass'] and cand_ms < base_ms),
            'not_yet_validated_in_native_motion':True,
        }
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2)+'\n')
    print('ONDA PROFILE: SAVED',target,flush=True)
    if not base['accuracy_pass']:
        print('ONDA PROFILE: BASELINE ACCURACY FAIL',flush=True)
        return 1
    print('ONDA PROFILE: BASELINE ACCURACY PASS (candidates reported independently)',flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
