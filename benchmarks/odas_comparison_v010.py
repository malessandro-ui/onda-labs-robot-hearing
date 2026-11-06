"""Paired benchmarks: v0.9 motion-SRP, covariance MUSIC, optional real ODAS.

Usage: python benchmarks/odas_comparison_v010.py --limit 3
       python benchmarks/odas_comparison_v010.py --odas-binding /absolute/path/binding.json

No ODAS scores are synthesized. Moving-array cases are explicitly unsupported
for ODAS and static MUSIC to avoid comparing unlike frame semantics. Existing
v0.9 scenario generator creates identical PCM once, saved before inference.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'benchmarks'))
from motion_stress_v09 import FS, OFFSETS, cases, _pcm, match
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig
from onda_hearing.spatial.music import CovarianceMUSICLocalizer
from onda_hearing.spatial.odas_adapter import ODASBinding, ODASOfflineBackend, ODASUnavailableError, ODASProtocolError

METHODS=('motion_srp_v09','covariance_music_v010','odas_offline_ssl')

def summarize(rows,method):
    applicable=[r for r in rows if r['method']==method and r['status']=='ok']
    skipped=[r for r in rows if r['method']==method and r['status']!='ok']
    tp=sum(r['tp'] for r in applicable);fp=sum(r['fp'] for r in applicable);fn=sum(r['fn'] for r in applicable)
    return {'evaluated':len(applicable),'unavailable_or_unsupported':len(skipped),
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'exact_count_rate':sum(r['exact_count'] for r in applicable)/len(applicable) if applicable else None}

def run(output:Path,limit: int|None=None,odas_binding:Path|None=None):
    output.mkdir(parents=True,exist_ok=True)
    mixtures=output/'mixtures';mixtures.mkdir(exist_ok=True)
    binding=None
    if odas_binding is not None:
        d=json.loads(odas_binding.read_text())
        binding=ODASBinding(**d)
    rows=[]
    chosen=cases() if limit is None else cases()[:limit]
    for i,case in enumerate(chosen):
        pcm,poses,truth=_pcm(case,909000+i)
        array=MicrophoneArray.from_positions(OFFSETS,array_id='benchmark_array',frame_id='body',geometry_id='v09')
        packet=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=array)
        np.savez_compressed(mixtures/f'{i:03d}.npz',pcm=pcm,truth_deg=np.asarray(truth),
                            poses_times_s=poses.times_s,poses_positions_m=poses.positions_m,
                            poses_rotations=poses.rotations)
        for method in METHODS:
            status='ok';est=[]
            try:
                if method=='motion_srp_v09':
                    frame=MotionAwareSRPLocalizer(MotionSRPConfig()).estimate(packet,array,poses,pose_clock_id='sim')
                    est=[float(np.rad2deg(d.azimuth_rad)) for d in frame.result.directions]
                elif case['rotation']!=0:
                    status='unsupported_motion_no_world_frame_steering'
                elif method=='covariance_music_v010':
                    result=CovarianceMUSICLocalizer().estimate(packet,array)
                    est=[float(np.rad2deg(d.azimuth_rad)) for d in result.directions]
                elif binding is None:
                    status='odas_unavailable_no_binding'
                else:
                    result=ODASOfflineBackend(binding).process(packet,array)
                    est=[float(np.rad2deg(d.azimuth_rad)) for d in result.directions]
            except (ValueError,RuntimeError,OSError) as exc:
                status=f'error:{type(exc).__name__}:{str(exc)[:160]}'
            errors=match(truth,est) if status=='ok' else []
            rows.append({'case':case['name'],'method':method,'status':status,
                         'rotation_deg':case['rotation'],'room':case['room'],
                         'level_db':case['level'],'truth_deg':json.dumps(truth),
                         'estimates_deg':json.dumps(est),'tp':len(errors),
                         'fp':len(est)-len(errors),'fn':len(truth)-len(errors) if status=='ok' else 0,
                         'exact_count':status=='ok' and len(est)==len(truth) and len(errors)==len(truth)})
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case','method']);writer.writeheader();writer.writerows(rows)
    report={'seed_base':909000,'paired_recordings':True,
            'warnings':['Only independently installed ODAS can produce ODAS benchmark results',
                        'ODAS and baseline MUSIC are not evaluated on moving-array cases',
                        'ODAS uses last potential JSON frame, not an entire-recording time-integrated result',
                        'ODAS configuration geometry/clock alignment must be audited externally',
                        'synthetic benchmark; no physical-microphone accuracy claims'],
            'methods':{method:summarize(rows,method) for method in METHODS}}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'cases.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('benchmark_results/v010_odas'))
    parser.add_argument('--limit',type=int);parser.add_argument('--odas-binding',type=Path)
    args=parser.parse_args()
    run(args.output,args.limit,args.odas_binding)
