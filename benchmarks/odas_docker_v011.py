"""Paired static-window benchmark; never fabricate ODAS numbers.

python benchmarks/odas_docker_v011.py --limit 3
python benchmarks/odas_docker_v011.py --limit 3 --enable-odas
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'benchmarks'))
from motion_stress_v09 import cases,_pcm,OFFSETS,FS,match
from onda_hearing.contracts import AudioPacket,MicrophoneArray
from onda_hearing.spatial.srp import SRPCandidateLocalizer
from onda_hearing.spatial.odas_docker import DockerODASRunner,DockerODASOptions


def run(output:Path,*,limit:int=3,enable_odas:bool=False,image:str='onda-odas:dev'):
    if limit<1:raise ValueError('limit must be positive')
    output.mkdir(parents=True,exist_ok=True)
    (output/'mixtures').mkdir(exist_ok=True)
    rows=[]
    chosen=[c for c in cases() if c['rotation']==0][:limit]
    if not chosen:raise ValueError('no static cases available')
    array=MicrophoneArray.from_positions(OFFSETS,array_id='benchmark_array',frame_id='body',geometry_id='v09')
    odas=DockerODASRunner(DockerODASOptions(image=image)) if enable_odas else None
    for i,case in enumerate(chosen):
        pcm,pose,truth=_pcm(case,909000+i)
        packet=AudioPacket.from_pcm(pcm,FS,start_ns=0,clock_id='sim',array=array)
        np.savez_compressed(output/'mixtures'/f'{i:03d}.npz',pcm=pcm,truth_deg=np.asarray(truth),microphones_m=array.positions_m)
        for method in ('srp_phat','odas_docker'):
            status='ok';est=[];frames=0;detail=''
            try:
                if method=='srp_phat':
                    result=SRPCandidateLocalizer().estimate(packet,array)
                    est=[float(np.rad2deg(d.azimuth_rad)) for d in result.result.directions]
                elif odas is None:
                    status='not_run_by_default'
                else:
                    result=odas.run(packet,array)
                    frames=len(result.frames)
                    # Deliberately final-frame vs full-window comparison is marked diagnostic only.
                    # Save every real frame so future versions can score matched time windows.
                    frame_file=output/f'odas_{i:03d}_frames.json'
                    frame_file.write_text(json.dumps([{'time_s':f.time_s,'angles_deg':f.directions_deg,'energies':f.energies} for f in result.frames],indent=2))
                    est=list(result.frames[-1].directions_deg)
                    detail='final_frame_only_not_temporally_equivalent'
            except Exception as exc:
                status='unavailable_or_error';detail=f'{type(exc).__name__}: {str(exc)[:200]}'
            matched=match(truth,est) if status=='ok' else []
            rows.append(dict(case=case['name'],method=method,status=status,details=detail,
                        truth_deg=json.dumps(truth),estimate_deg=json.dumps(est),frames=frames,
                        tp=len(matched),fp=len(est)-len(matched),fn=len(truth)-len(matched) if status=='ok' else 0))
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'paired_pcm':True,'odas_executed':enable_odas,
             'temporal_comparability':'ODAS final frame versus Onda full window: diagnostic ONLY; not an algorithm ranking',
             'methods':{name:{'evaluated':sum(r['status']=='ok' for r in rows if r['method']==name),
                        'errors':sum(r['status']=='unavailable_or_error' for r in rows if r['method']==name)}
                        for name in ('srp_phat','odas_docker')}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('benchmark_results/v011_odas'))
    p.add_argument('--limit',type=int,default=3);p.add_argument('--enable-odas',action='store_true');p.add_argument('--image',default='onda-odas:dev')
    a=p.parse_args();run(a.output,limit=a.limit,enable_odas=a.enable_odas,image=a.image)
