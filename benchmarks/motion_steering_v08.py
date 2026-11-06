"""Paired comparison on saved v0.7 physical direct-path recordings.

Two stationary world sources, rotating rigid body microphone array.
No source truth enters an estimator. Evaluate exact source counts and matched
angular bearings; NO tracking or separation claims.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from onda_hearing.contracts import AudioPacket,MicrophoneArray
from onda_hearing.spatial import MotionSRPConfig,MotionAwareSRPLocalizer,SRPConfig,SRPCandidateLocalizer
from onda_hearing.physics.propagation import AcousticConfig,DirectPathRenderer,PhysicalSource,PoseTrajectory
from room_motion_v07 import FS,DURATION,CENTER,OFFSETS,rz,match

def run(output:Path,*,limit:int|None=None):
    output.mkdir(parents=True,exist_ok=True)
    recordings=output/'mixtures';recordings.mkdir(exist_ok=True)
    cases=[{'rotation_deg':r,'separation_deg':s,'level_db':db} for r in (0,45,90,180) for s,db in ((45,0),(90,-6),(120,-12))]
    if limit is not None:cases=cases[:limit]
    rows=[]
    for index,case in enumerate(cases):
        rng=np.random.default_rng(808000+index)
        true=[-case['separation_deg']/2,case['separation_deg']/2]
        sources=[];n=int(FS*DURATION)
        for idx,angle in enumerate(true):
            signal=rng.normal(size=n+1024)
            signal=np.convolve(signal,([.7,.3] if idx else [.25,.5,.25]),mode='same')
            signal=.16*10**((case['level_db'] if idx else 0)/20)*signal/max(1e-9,np.max(np.abs(signal)))
            location=CENTER+2*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.])
            sources.append(PhysicalSource(f's{idx}',signal,PoseTrajectory.stationary(location)))
        times=np.linspace(0,DURATION,41)
        trajectory=PoseTrajectory(times,np.repeat(CENTER[None,:],len(times),axis=0),np.stack([rz(t) for t in np.linspace(0,np.deg2rad(case['rotation_deg']),len(times))]))
        pcm=DirectPathRenderer(AcousticConfig(sample_rate_hz=FS)).render(OFFSETS,trajectory,sources,DURATION)
        # Pose-enabled algorithm sees original rigid microphone BODY offsets.
        body_array=MicrophoneArray.from_positions(OFFSETS,array_id='robot',frame_id='array_body',geometry_id=f'v08_{index}')
        packet=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=body_array)
        # Static baseline gets same PCM but world geometry fixed at window midpoint.
        center_offsets=OFFSETS@rz(np.deg2rad(case['rotation_deg'])/2).T
        midpoint_array=MicrophoneArray.from_positions(center_offsets,array_id='robot',frame_id='world_static',geometry_id=f'static_{index}')
        static_packet=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=midpoint_array)
        np.savez_compressed(recordings/f'{index:03d}.npz',pcm=pcm,body_offsets_m=OFFSETS,pose_times_s=times,pose_rotations=trajectory.rotations,truth_deg=np.array(true))
        models={
            'static_midpoint_srp_v05':SRPCandidateLocalizer(SRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=18,min_relative_score=.5,min_prominence=.006)),
            'motion_world_srp_v08':MotionAwareSRPLocalizer(MotionSRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=18,min_relative_score=.55,min_prominence=.01)),
        }
        for method,model in models.items():
            if method.startswith('static_'):frame=model.estimate(static_packet,midpoint_array)
            else:frame=model.estimate(packet,body_array,trajectory,pose_clock_id='sim')
            estimates=[round(float(np.rad2deg(d.azimuth_rad)),3) for d in frame.result.directions]
            tp=match(true,estimates)
            rows.append(dict(case=index,method=method,rotation_deg=case['rotation_deg'],separation_deg=case['separation_deg'],level_db=case['level_db'],truth_deg=json.dumps(true),estimates_deg=json.dumps(estimates),tp=tp,fp=len(estimates)-tp,fn=len(true)-tp,exact_count=len(estimates)==len(true)))
    if rows:
        with (output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    def summary(group):
        tp=sum(r['tp'] for r in group);fp=sum(r['fp'] for r in group);fn=sum(r['fn'] for r in group)
        return dict(cases=len(group),tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,exact_count_rate=sum(r['exact_count'] for r in group)/len(group) if group else None)
    output_data={'seed_base':808000,'notes':['direct-path motion only: moving-room reflections not simulated','no track identity or waveform separation evaluated','source count unverified; candidates can be false','poses in simulation seconds in same sim clock as audio'], 'overall':{},'by_rotation':{}}
    for method in ('static_midpoint_srp_v05','motion_world_srp_v08'):
        subset=[r for r in rows if r['method']==method]
        output_data['overall'][method]=summary(subset)
        output_data['by_rotation'][method]={str(deg):summary([r for r in subset if r['rotation_deg']==deg]) for deg in (0,45,90,180)}
    (output/'summary.json').write_text(json.dumps(output_data,indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps(output_data,indent=2))
    return output_data

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('benchmark_results/v08_motion_steering'));parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.output,limit=args.limit)
