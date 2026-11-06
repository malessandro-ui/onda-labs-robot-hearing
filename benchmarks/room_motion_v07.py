"""Paired v0.7 acoustic stress test. Truth used for scoring, never inference.

Static-room trials: first-order image paths (not calibrated physical RIRs).
Motion trials: exact time-varying array poses, direct path only. Moving-room
scattering or moving RIR rendering is NOT modeled. Both baselines see the
same saved PCM and geometry for each window. Relative/world metrics are
computed only when timestamps/poses are available; no source-ID claims.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from onda_hearing.contracts import AudioPacket,MicrophoneArray
from onda_hearing.physics.propagation import AcousticConfig,DirectPathRenderer,PhysicalSource,PoseTrajectory
from onda_hearing.rooms import RoomMaterial,ShoeboxRoom,image_source_rir,MeasuredRIR
from onda_hearing.motion import TimedBearing,to_world,angular_error_deg
from onda_hearing.spatial import SRPConfig,SRPCandidateLocalizer,SubbandConfig,SubbandSRPLocalizer

FS=16000;DURATION=.2;C=343.
OFFSETS=np.array([[-.07,-.05,0],[.07,-.05,0],[.07,.05,0],[-.07,.05,0]])
CENTER=np.array([3.,2.5,1.5])

def rz(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def wrap(a):return (a+180)%360-180

def match(truth,pred,tol=15):
    # One-to-one matching, with no reuse of a predicted peak.
    owner={}
    adj=[[j for j,p in enumerate(pred) if angular_error_deg(t,p)<=tol] for t in truth]
    def augment(i,seen):
        for j in adj[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(truth)):augment(i,set())
    return len(owner)

def scenarios():
    result=[]
    for sep in (15,30,60,90):
        for level in (0,-10,-20):
            result.append(dict(group='angular_level',sep=sep,db=level,snr=None,absorption=None,rotation=0,n_sources=2))
    for absorption in (.15,.5,.85):
        for n in (1,2,3):
            result.append(dict(group='room_absorption',sep=70,db=-6,snr=None,absorption=absorption,rotation=0,n_sources=n))
    for snr in (0,10,20):
        result.append(dict(group='noise',sep=70,db=-6,snr=snr,absorption=None,rotation=0,n_sources=2))
    for rotate in (45,90,180):
        result.append(dict(group='continuous_rotation',sep=70,db=-6,snr=None,absorption=None,rotation=rotate,n_sources=1))
    return result

def generate(case,index):
    rng=np.random.default_rng(700_000+index);n=int(FS*DURATION)
    bearings=([-45.] if case['n_sources']==1 else
              [-case['sep']/2,case['sep']/2] if case['n_sources']==2 else [-65.,0.,65.])
    dbs=[0]+[case['db']]*(len(bearings)-1)
    positions=np.stack([CENTER+2*np.array([np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a)),0]) for a in bearings])
    input_pcm=np.zeros((n,len(bearings)))
    for i,db in enumerate(dbs):
        y=rng.normal(size=n+2048)
        if i%2:y=np.convolve(y,[.2,.6,.2],mode='same')
        y=y/(max(np.max(abs(y)),1e-12))*.2*10**(db/20)
        input_pcm[:,i]=y[:n]
    origins=np.array([CENTER]*21)
    theta=np.linspace(0,np.deg2rad(case['rotation']),21)
    rot=np.stack([rz(a) for a in theta])
    pose=PoseTrajectory(np.linspace(0,DURATION,21),origins,rot)
    # Full-window continuous rotation; detector receives window-center geometry
    # and is expected to be imperfect because most estimators assume static arrays.
    mics=CENTER+OFFSETS@rz(float(theta[len(theta)//2])).T
    array=MicrophoneArray.from_positions(mics,array_id='bench',frame_id='world',geometry_id=f'v07_{index}')
    if case['absorption'] is not None:
        material=RoomMaterial((case['absorption'],)*7)
        room=ShoeboxRoom((7,6,3.5),(material,)*6)
        h=image_source_rir(room,positions,mics,sample_rate_hz=FS,max_duration_s=.1)
        rir=MeasuredRIR(h,FS,tuple(f'mic{i}' for i in range(4)),tuple(f's{i}' for i in range(len(bearings))),f'v07 image-source order1 index={index}')
        signal=rir.render(input_pcm,sample_rate_hz=FS,source_ids=rir.source_ids,microphone_names=rir.microphone_names,output_mode='same')
    else:
        sources=[]
        for i,pos in enumerate(positions):
            sources.append(PhysicalSource(f's{i}',input_pcm[:,i],PoseTrajectory.stationary(pos)))
        signal=DirectPathRenderer(AcousticConfig(sample_rate_hz=FS)).render(OFFSETS,pose,sources,DURATION)
    if case['snr'] is not None:
        rms=float(np.sqrt(np.mean(signal.astype(float)**2)))
        signal=signal+rng.normal(0,rms*10**(-case['snr']/20),signal.shape)
    peak=float(np.max(np.abs(signal)))
    if peak>1:signal=signal/peak
    return signal.astype(np.float32),array,bearings,pose

def run(output:Path,limit=None):
    output.mkdir(parents=True,exist_ok=True);saved=output/'mixtures';saved.mkdir(exist_ok=True)
    models={
        'srp_phat_v05':SRPCandidateLocalizer(SRPConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=12,min_relative_score=.5,min_prominence=.006)),
        'subband_srp_v06':SubbandSRPLocalizer(SubbandConfig(angular_resolution_deg=3,max_candidates=3,min_separation_deg=12,min_relative_score=.64,min_prominence=.10)),
    }
    rows=[]
    for idx,case in enumerate(scenarios()[:limit]):
        pcm,array,truth,pose=generate(case,idx)
        np.savez_compressed(saved/f'{idx:03d}.npz',pcm=pcm,positions_m=array.positions_m,
                            truth_deg=np.array(truth),pose_times_s=pose.times_s,pose_rotations=pose.rotations)
        audio=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=array)
        for method,model in models.items():
            obs=model.estimate(audio,array)
            pred=[float(np.rad2deg(d.azimuth_rad)) for d in obs.result.directions]
            # Algorithm was given window-center WORLD-space geometry; outputs
            # already represent world bearings (not local bearings). Do not
            # apply the pose transform a second time.
            tp=match(truth,pred)
            rows.append(dict(case=idx,method=method,group=case['group'],separation_deg=case['sep'],
                             level_db=case['db'],snr_db=case['snr'],absorption=case['absorption'],
                             rotation_deg=case['rotation'],n_sources=len(truth),tp=tp,
                             fp=len(pred)-tp,fn=len(truth)-tp,exact_count=len(pred)==len(truth),
                             true_deg=json.dumps(truth),predicted_deg=json.dumps(pred)))
    keys=list(rows[0]) if rows else []
    if rows:
        with (output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    def summarize(group):
        tp=sum(int(x['tp']) for x in group);fp=sum(int(x['fp']) for x in group);fn=sum(int(x['fn']) for x in group)
        return dict(cases=len(group),tp=tp,fp=fp,fn=fn,
                    precision=(tp/(tp+fp) if tp+fp else None),recall=(tp/(tp+fn) if tp+fn else None),
                    exact_count_rate=sum(bool(x['exact_count']) for x in group)/len(group) if group else None)
    summary={'seed_base':700000,'fs':FS,'duration_s':DURATION,'matching_tolerance_deg':15,
             'warnings':['first-order image-source is NOT a measured RIR','moving-room reflections NOT modeled',
                         'no persistent ID tracking or waveform separation validation',
                         'held window-center geometry makes continuous-motion estimators approximate'],
             'overall':{},'by_condition':{}}
    for method in models:
        subset=[r for r in rows if r['method']==method]
        summary['overall'][method]=summarize(subset)
        summary['by_condition'][method]={}
        for cond in sorted({r['group'] for r in subset}):
            summary['by_condition'][method][cond]=summarize([r for r in subset if r['group']==cond])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps(scenarios()[:limit],indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('benchmark_results/v07_room_motion'))
    ap.add_argument('--limit',type=int);args=ap.parse_args()
    run(args.output,args.limit)
