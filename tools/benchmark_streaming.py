#!/usr/bin/env python3
"""CPU GCC-PHAT single-source throughput benchmark, separate from Onda SRP baseline."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from onda_hearing.streaming import GCCPHATSRP
from onda_hearing.streaming.simulated_capture import DEFAULT_ARRAY_METERS,render_point_source


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--windows',type=int,default=250)
    p.add_argument('--out',type=Path,default=Path('docs/validation/streaming_v016/cpu_benchmark.json'))
    a=p.parse_args()
    if a.windows<50:p.error('at least 50 windows for meaningful percentiles')
    rng=np.random.default_rng(7)
    mono=rng.standard_normal(100000).astype(np.float32)
    src=(1.5*math.cos(math.radians(38)),1.5*math.sin(math.radians(38)),0.)
    pcm=render_point_source(mono,10000,2048,16000,(0,0,0),(0,0,0,1),src)
    est=GCCPHATSRP(DEFAULT_ARRAY_METERS)
    times=[];angles=[]
    for i in range(a.windows+10):
        start=perf_counter_ns()
        d=est.estimate(pcm)
        ms=(perf_counter_ns()-start)/1e6
        if i>=10:times.append(ms);angles.append(d.local_bearing_deg)
    score=float(np.percentile(np.abs((np.array(angles)-38+180)%360-180),75))
    output={'algorithm':'single-source GCC-PHAT SRP reference (not legacy MotionAwareSRPLocalizer)',
            'tested_windows':a.windows,'sample_rate_hz':16000,'window_samples':2048,
            'hop_samples':512,'p50_estimator_ms':float(np.percentile(times,50)),
            'p95_estimator_ms':float(np.percentile(times,95)),
            'estimated_hop_real_time_factor_p50':float(np.percentile(times,50))/32.,
            'angular_p75_error_deg':score,'accuracy_pass':bool(score<=10),
            'warning':'Estimator-only microbenchmark, not robot 60Hz certification or acoustic model equivalence.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    return 0 if output['accuracy_pass'] else 1

if __name__=='__main__':raise SystemExit(main())
