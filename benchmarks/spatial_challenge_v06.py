"""Paired, frozen-mixture benchmark for two experimental source-bearing estimators.

The same saved PCM is passed to each algorithm. Ground truth is used ONLY by
scoring. Reverberation uses controlled delayed copies, NOT calibrated room
acoustics. Rotating arrays use a snapshot orientation within each window;
continuous within-window rotation and world-track consistency remain untested.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.physics.propagation import AcousticConfig, DirectPathRenderer, PhysicalSource, PoseTrajectory
from onda_hearing.spatial import SRPConfig, SRPCandidateLocalizer, SubbandConfig, SubbandSRPLocalizer

FS=16000
DURATION=.20
BASE=np.array([[-.09,-.065,0],[.09,-.065,0],[.09,.065,0],[-.09,.065,0]],dtype=float)

def wrap(deg): return (float(deg)+180)%360-180

def distance(a,b):return abs(wrap(a-b))

def cases():
    out=[]
    out += [dict(name='silence',angles=[],levels=[],snr=None,echo=0,rotation=0)]
    for angle in [-150,-90,-45,0,45,90,150]:
        out.append(dict(name=f'single_{angle}',angles=[float(angle)],levels=[0.],snr=None,echo=0,rotation=0))
    for sep in [10,15,20,30,45,60,90,120,150,180]:
        for attenuation in [0,-10,-20]:
            # Offset avoids relying solely on symmetric cardinal bearings.
            out.append(dict(name=f'two_sep{sep}_db{attenuation}', angles=[wrap(-25-sep/2),wrap(-25+sep/2)],levels=[0.,float(attenuation)],snr=None,echo=0,rotation=0))
    out += [dict(name=f'three_{i}',angles=a,levels=l,snr=None,echo=0,rotation=0) for i,(a,l) in enumerate([
        ([-70,0,70],[0,-3,-6]),([-30,0,30],[0,0,0]),([-90,-10,65],[0,-10,-20]),([-150,-30,90],[0,-3,-10])])]
    for snr in [20,10,0]:
        out.append(dict(name=f'snr_{snr}',angles=[-50,45],levels=[0,-6],snr=snr,echo=0,rotation=0))
    for echo in [.2,.5,.8]:
        out.append(dict(name=f'echo_{echo}',angles=[-50,45],levels=[0,-6],snr=None,echo=echo,rotation=0))
    for rotation in [0,45,90,135,180]:
        out.append(dict(name=f'array_rotation_{rotation}',angles=[-40,55],levels=[0,-6],snr=None,echo=0,rotation=rotation))
    return out

def generate(case,seed):
    theta=np.deg2rad(case['rotation']); c=np.cos(theta);s=np.sin(theta)
    rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    positions=BASE @ rot.T
    array=MicrophoneArray.from_positions(positions,array_id='bench',frame_id='world',geometry_id='bench_geometry')
    rng=np.random.default_rng(seed)
    sources=[]
    for k,(angle,db) in enumerate(zip(case['angles'],case['levels'])):
        n=int(DURATION*FS)
        # Independent, spectrally varied broadband noise (no oracle mixture labels).
        y=rng.standard_normal(n)
        # Shape one source differently to exercise frequency diversity.
        if k%2:y=np.convolve(y,[.25,.5,.25],mode='same')
        y=y/(np.max(np.abs(y))+1e-12)*(.25*10**(db/20))
        a=np.deg2rad(angle)
        sources.append(PhysicalSource(str(k),y,PoseTrajectory.stationary((2.5*np.cos(a),2.5*np.sin(a),0))))
    renderer=DirectPathRenderer(AcousticConfig(sample_rate_hz=FS))
    signal=renderer.render(positions,PoseTrajectory.stationary((0,0,0)),sources,DURATION)
    if case['echo']:
        # Controlled delayed reflection per channel. A transparent stress test,
        # not a physically calibrated 3D RIR; RIR fidelity is a separate gate.
        delay=int(.012*FS)
        reflected=np.zeros_like(signal)
        reflected[delay:]=signal[:-delay]
        signal=signal+case['echo']*reflected
    if case['snr'] is not None and np.any(signal):
        rms=np.sqrt(np.mean(signal.astype(float)**2))
        noise_rms=rms*10**(-case['snr']/20)
        signal=signal+rng.normal(0,noise_rms,signal.shape)
    signal=np.asarray(signal,dtype=np.float32)
    peak=np.max(np.abs(signal))
    if peak>1:signal=signal/peak
    return signal,array

def match_one_to_one(truth,pred,tolerance=15):
    """Maximum bipartite matching, avoiding duplicate credit for a single prediction."""
    adjacency=[[j for j,p in enumerate(pred) if distance(t,p)<=tolerance] for t in truth]
    owner={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):
                owner[j]=i;return True
        return False
    for i in range(len(truth)):augment(i,set())
    return len(owner)

def evaluate(save_to:Path,*,limit=None):
    save_to.mkdir(parents=True,exist_ok=True)
    pcm_dir=save_to/'mixtures';pcm_dir.mkdir(exist_ok=True)
    localizers={
        'srp_phat_v05':SRPCandidateLocalizer(SRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=12,min_relative_score=.5,min_prominence=.006)),
        'subband_srp_v06':SubbandSRPLocalizer(SubbandConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=12,min_relative_score=.64,min_prominence=.10)),
    }
    rows=[]
    selected=cases()[:limit]
    for index,case in enumerate(selected):
        pcm,array=generate(case,seed=21610+index)
        path=pcm_dir/f'{index:03d}.npz'
        # Record exact inference input and world geometry, rather than rely on regenerating.
        np.savez_compressed(path,pcm=pcm,positions_m=array.positions_m,sample_rate_hz=FS)
        with np.load(path,allow_pickle=False) as frozen:
            frozen_pcm=frozen['pcm'];frozen_positions=frozen['positions_m']
        observed_array=MicrophoneArray.from_positions(frozen_positions,array_id='bench',frame_id='world',geometry_id='bench_geometry')
        audio=AudioPacket.from_pcm(frozen_pcm,FS,start_ns=0,clock_id='benchmark',array=observed_array)
        for algorithm,localizer in localizers.items():
            result=localizer.estimate(audio,observed_array)
            predicted=[wrap(np.rad2deg(d.azimuth_rad)) for d in result.result.directions]
            matched=match_one_to_one(case['angles'],predicted)
            rows.append(dict(case_id=index,scenario=case['name'],algorithm=algorithm,
                             truth=json.dumps(case['angles']),estimated=json.dumps([round(x,1) for x in predicted]),
                             sources=len(case['angles']),detections=len(predicted),true_positives=matched,
                             false_positives=len(predicted)-matched,false_negatives=len(case['angles'])-matched,
                             exact_source_count=len(predicted)==len(case['angles']),snr_db=case['snr'],
                             echo_gain=case['echo'],array_rotation_deg=case['rotation'],mixture=str(path.relative_to(save_to))))
    with (save_to/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={}
    for name in localizers:
        subset=[r for r in rows if r['algorithm']==name]
        tp=sum(r['true_positives'] for r in subset);fp=sum(r['false_positives'] for r in subset);fn=sum(r['false_negatives'] for r in subset)
        summaries[name]=dict(cases=len(subset),tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                             recall=tp/(tp+fn) if tp+fn else None,
                             source_count_accuracy=sum(r['exact_source_count'] for r in subset)/len(subset))
    report=dict(seed=21610,fs=FS,duration_s=DURATION,angular_match_tolerance_deg=15,
                pairing='identical saved PCM, matching array positions, max one-to-one bipartite',
                stress_model='static snapshot rotation; synthetic echo, not calibrated RIR',
                metrics=summaries)
    (save_to/'summary.json').write_text(json.dumps(report,indent=2))
    (save_to/'cases.json').write_text(json.dumps(selected,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmark_results/v06_spatial_challenge');p.add_argument('--limit',type=int,default=None)
    args=p.parse_args()
    print(json.dumps(evaluate(Path(args.output),limit=args.limit),indent=2))
