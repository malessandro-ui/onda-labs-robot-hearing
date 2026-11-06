"""Controlled rotation/clock/pose/room stress matrix and paired estimator ablations.

Run from repository root: python benchmarks/motion_stress_v09.py
Measurements are *synthetic*. Reporting includes misses and excess candidates.
Both estimator parameter sets see exactly the same saved audio and pose series.
No tracking, real-room RIR, real IMU, or source separation is evaluated.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.physics.propagation import AcousticConfig, DirectPathRenderer, PhysicalSource, PoseTrajectory
from onda_hearing.spatial.motion_srp import MotionSRPConfig, MotionAwareSRPLocalizer
from onda_hearing.spatial.motion_robustness import ClockAlignment, perturb_poses
from onda_hearing.rooms.rir import RoomMaterial, ShoeboxRoom, image_source_rir
from onda_hearing.physics.propagation import convolve_rir

FS = 16000
OFFSETS = np.array([[-.045,-.035,0.],[.045,-.035,0.],[.045,.035,0.],[-.045,.035,0.]])
CENTER = np.array([3.,3.,1.5])

def rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def cases():
    # Mixes are generated once per row; all parameter variants run on that same PCM.
    return [
        dict(name='static_control',duration=.35,rotation=0,sep=90,level=0,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='fast_90',duration=.35,rotation=90,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='fast_180',duration=.35,rotation=180,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='fast_270',duration=.35,rotation=270,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='long_90',duration=.8,rotation=90,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='long_180',duration=.8,rotation=180,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='close_30',duration=.35,rotation=90,sep=30,level=-6,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='quiet_minus20',duration=.35,rotation=90,sep=90,level=-20,noise=0,room=False,pose_noise=0,clock_offset=0),
        dict(name='noise_10db',duration=.35,rotation=90,sep=90,level=-6,noise=10,room=False,pose_noise=0,clock_offset=0),
        dict(name='pose_jitter_2deg',duration=.35,rotation=90,sep=90,level=-6,noise=0,room=False,pose_noise=2,clock_offset=0),
        dict(name='clock_5ms_corrected',duration=.35,rotation=90,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=.005),
        dict(name='clock_5ms_uncorrected',duration=.35,rotation=90,sep=90,level=-6,noise=0,room=False,pose_noise=0,clock_offset=.005),
        dict(name='static_room_rir',duration=.35,rotation=0,sep=90,level=-6,noise=0,room=True,pose_noise=0,clock_offset=0),
    ]

def match(truth,pred,tol=15):
    edges=sorted((abs((a-b+180)%360-180),i,j) for i,a in enumerate(truth) for j,b in enumerate(pred))
    taken_a=set();taken_b=set();matched=[]
    for err,i,j in edges:
        if err > tol:break
        if i not in taken_a and j not in taken_b:
            taken_a.add(i);taken_b.add(j);matched.append(float(err))
    return matched

def _pcm(case,seed):
    rng=np.random.default_rng(seed)
    duration=case['duration']; n=int(FS*duration)
    truth=[-case['sep']/2,case['sep']/2]
    sources=[]
    for idx,angle in enumerate(truth):
        noise=rng.standard_normal(n+1024)
        signal=np.convolve(noise,([.6,.3,.1] if idx else [.2,.6,.2]),mode='same')
        signal=.2*10**((case['level'] if idx else 0)/20)*signal/max(1e-9,np.max(abs(signal)))
        pos=CENTER+2*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.])
        sources.append(PhysicalSource(f's{idx}',signal,PoseTrajectory.stationary(pos)))
    t=np.linspace(0,duration,max(3,int(np.ceil(duration/.01))+1))
    trajectory=PoseTrajectory(t,np.repeat(CENTER[None],len(t),axis=0),
                              np.stack([rz(a) for a in np.linspace(0,np.deg2rad(case['rotation']),len(t))]))
    if case['room']:
        if case['rotation']!=0:raise ValueError('static RIR applied to moving array: invalid physical model')
        mats=tuple(RoomMaterial((.28,)*7) for _ in range(6))
        room=ShoeboxRoom((7.,7.,3.),mats)
        src_pos=np.stack([s.trajectory.positions_m[0] for s in sources])
        mic_pos=CENTER+OFFSETS
        h=image_source_rir(room,src_pos,mic_pos,sample_rate_hz=FS,max_duration_s=.07)
        source_pcm=np.stack([s.signal[:n] for s in sources],axis=1)
        pcm=convolve_rir(source_pcm,h,output_mode='same')
    else:
        pcm=DirectPathRenderer(AcousticConfig(sample_rate_hz=FS)).render(OFFSETS,trajectory,sources,duration)
    if case['noise']:
        rms=float(np.sqrt(np.mean(pcm**2)))
        pcm=pcm+rng.standard_normal(pcm.shape)*rms*10**(-case['noise']/20)
    return pcm,trajectory,truth

def run(output:Path, *, limit: int | None = None):
    output.mkdir(parents=True,exist_ok=True)
    recordings=output/'mixtures';recordings.mkdir(exist_ok=True)
    configs=[
        ('default_v08', MotionSRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=18,min_relative_score=.55,min_prominence=.01)),
        ('finer_scan', MotionSRPConfig(angular_resolution_deg=1.5,max_candidates=3,min_separation_deg=18,min_relative_score=.55,min_prominence=.01)),
        ('short_blocks', MotionSRPConfig(block_samples=160,hop_samples=80,angular_resolution_deg=3,max_candidates=3,min_separation_deg=18,min_relative_score=.55,min_prominence=.01)),
        ('relaxed_selection', MotionSRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=18,min_relative_score=.40,min_prominence=.006)),
    ]
    rows=[];selected=cases() if limit is None else cases()[:limit]
    for i,case in enumerate(selected):
        pcm,ground_poses,truth=_pcm(case,909000+i)
        raw_pose=PoseTrajectory(ground_poses.times_s-case['clock_offset'],ground_poses.positions_m,ground_poses.rotations)
        if case['name'].endswith('uncorrected'):
            aligned=raw_pose # deliberately wrong; evaluation only, no hidden correction
        else:
            aligned=ClockAlignment(case['clock_offset'],'sim','pose').align(raw_pose,audio_clock_id='sim',pose_clock_id='pose')
        if case['pose_noise']:
            aligned=perturb_poses(aligned,seed=909010+i,yaw_std_deg=case['pose_noise'])
        array=MicrophoneArray.from_positions(OFFSETS,array_id='benchmark_array',frame_id='body',geometry_id='v09')
        packet=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=array)
        np.savez_compressed(recordings/f'{i:03d}.npz',pcm=pcm,truth=np.asarray(truth),
                            pose_times=aligned.times_s,pose_positions=aligned.positions_m,
                            pose_rotations=aligned.rotations,original_pose_times=raw_pose.times_s)
        for label,conf in configs:
            status='ok';pred=[];errors=[]
            try:
                result=MotionAwareSRPLocalizer(conf).estimate(packet,array,aligned,pose_clock_id='sim')
                pred=[float(np.rad2deg(d.azimuth_rad)) for d in result.result.directions]
                errors=match(truth,pred)
            except ValueError as exc:
                status=f'rejected: {exc}'
            rows.append(dict(case=case['name'],method=label,status=status,rotation_deg=case['rotation'],
                             duration_s=case['duration'],separation_deg=case['sep'],level_db=case['level'],
                             noise_snr_db=case['noise'],room=case['room'],pose_noise_deg=case['pose_noise'],
                             clock_offset_s=case['clock_offset'],truth_deg=json.dumps(truth),estimates_deg=json.dumps(pred),
                             tp=len(errors),fp=len(pred)-len(errors),fn=len(truth)-len(errors),
                             mean_error_deg=float(np.mean(errors)) if errors else None,
                             exact_count=(len(pred)==len(truth) and len(errors)==len(truth))))
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case','method']);writer.writeheader();writer.writerows(rows)
    def aggregate(group):
        tp=sum(r['tp'] for r in group);fp=sum(r['fp'] for r in group);fn=sum(r['fn'] for r in group)
        return dict(cases=len(group),tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,exact_count_rate=sum(r['exact_count'] for r in group)/len(group) if group else None,
                    rejected=sum(r['status']!='ok' for r in group))
    report=dict(seed_base=909000,notes=['synthetic reference, not physical validation','first-order static RIR tested only on nonmoving array',
      'pose clock offset is known and explicitly corrected except designated negative control',
      'pose perturbation is independent synthetic jitter, not a calibrated IMU model',
      'no persistent source identity or waveform separation evaluated'],
      overall={label:aggregate([r for r in rows if r['method']==label]) for label,_ in configs},
      by_case={c['name']:{label:aggregate([r for r in rows if r['method']==label and r['case']==c['name']]) for label,_ in configs} for c in selected})
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(report['overall'],indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('benchmark_results/v09_motion_stress'));p.add_argument('--limit',type=int)
    args=p.parse_args();run(args.output,limit=args.limit)
