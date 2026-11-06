"""Reproducible diagnostic matrix, not a passed multisource acceptance benchmark.

Each independent source uses a seeded broadband waveform. Source-count metrics
are evaluated against *known simulated truth*; truth is never sent to inference.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.physics.propagation import AcousticConfig,DirectPathRenderer,PhysicalSource,PoseTrajectory
from onda_hearing.spatial import SRPCandidateLocalizer,SRPConfig

ANGLES=[-60.,-30.,0.,30.,60.]
ARRAY=MicrophoneArray.from_positions([[-.08,-.07,0],[.08,-.07,0],[.08,.07,0],[-.08,.07,0]],array_id='benchmark',frame_id='array',geometry_id='quad_16x14cm')

def angular_error(a,b):return abs((a-b+180)%360-180)

def run(output:Path,*,fs:int=16000,duration:float=.32):
    output.mkdir(parents=True,exist_ok=True)
    renderer=DirectPathRenderer(AcousticConfig(sample_rate_hz=fs))
    detector=SRPCandidateLocalizer(SRPConfig(max_candidates=3,angular_resolution_deg=3,min_relative_score=.5,min_prominence=.006))
    rows=[]
    cases=[(1,[-45.],[0.]),(2,[-45.,45.],[0.,0.]),(2,[-15.,15.],[0.,-10.]),(2,[-40.,50.],[0.,-20.]),(3,[-65.,0.,65.],[0.,-3.,-6.])]
    for case_index,(count,angles,levels) in enumerate(cases):
        rng=np.random.default_rng(1901+case_index)
        source=[]
        for k,(angle,level) in enumerate(zip(angles,levels)):
            theta=np.deg2rad(angle)
            waveform=rng.standard_normal(round(duration*fs))
            waveform=np.convolve(waveform,np.ones(3)/3,mode='same')
            waveform=(.08*10**(level/20))*waveform/max(1.,np.max(np.abs(waveform)))
            position=(2.2*np.cos(theta),2.2*np.sin(theta),0.)
            source.append(PhysicalSource(f'source_{k}',waveform,PoseTrajectory.stationary(position)))
        pcm=renderer.render(ARRAY.positions_m,PoseTrajectory.stationary([0,0,0]),source,duration)
        packet=AudioPacket.from_pcm(pcm,fs,start_ns=0,clock_id='sim',array=ARRAY)
        candidates=detector.estimate(packet,ARRAY)
        estimates=[float(np.rad2deg(d.azimuth_rad)) for d in candidates.result.directions]
        matched=sum(any(angular_error(estimate,t)<15 for estimate in estimates) for t in angles)
        false=max(0,len(estimates)-matched)
        rows.append({'case':case_index,'sources':count,'angles_true_deg':json.dumps(angles),'levels_db':json.dumps(levels),'estimated_deg':json.dumps([round(x,1) for x in estimates]),'matched_within_15deg':matched,'unmatched_candidates_proxy':false,'exact_count':len(estimates)==count})
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps({'cases':len(rows),'exact_source_count_cases':sum(r['exact_count'] for r in rows),'note':'Diagnostic only. Source-count estimates are spatial peak hypotheses, not validated independent sources. Non-greedy matching needed for formal metrics.'},indent=2))
    return rows

if __name__=='__main__':
    for row in run(Path('benchmark_results/v05_multisource')):print(row)
