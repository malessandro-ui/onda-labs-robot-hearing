#!/usr/bin/env python3
"""Dependency-light sanity demo: direct + finite specular reflection in same 4ch PCM."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from onda_hearing.acoustic_scene import (Pose3,Binding,Schedule,Emitter,MicArray,
                                         AudioAsset,RendererConfig)
from onda_hearing.acoustic_scene.physical import (TransmissionMaterial,
    AcousticSurface,AcousticGeometry,GeometryAwareAcousticRenderer)
from onda_hearing.acoustic_scene.unified_physics import (
    ReflectionMaterial, UnifiedGeometryAcousticRenderer, UnifiedPhysicsConfig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('runs/v020_unified_static'))
    args=p.parse_args()
    sr=16000
    t=np.arange(sr*2,dtype=float)/sr
    x=(.06*np.sin(2*np.pi*440*t)+.03*np.sin(2*np.pi*3000*t)).astype(np.float32)
    src=Emitter('phone','phone',Binding(world=Pose3((1.,0.,1.))),Schedule(onset_s=0.))
    mics=MicArray(Binding(world=Pose3((2.,0.,1.))),
                  ((0.,0.,0.),(0.,.035,0.),(0.,-.035,0.),(.035,0.,0.)))
    tl=TransmissionMaterial((125.,8000.),(60.,60.),'synthetic_wall_opaque')
    wall=AcousticSurface.quad('wall',((-3,2,0),(5,2,0),(5,2,3),(-3,2,3)),tl)
    geom=AcousticGeometry((wall,))
    reflection=ReflectionMaterial((125.,8000.),(.2,.2),(.02,.02),'synthetic_reflection')
    cfg=RendererConfig(sample_rate=sr,noise_std=0.)
    unified=UnifiedGeometryAcousticRenderer(cfg,mics,[src],{'phone':AudioAsset('phone',x,sr)},
        geom,{'wall':reflection},unified_config=UnifiedPhysicsConfig())
    direct=GeometryAwareAcousticRenderer(cfg,mics,[src],{'phone':AudioAsset('phone',x,sr)},geom)
    y=[];d=[]
    for cursor in range(0,len(x),256):
        n=min(256,len(x)-cursor)
        y.append(unified.render(cursor,n));d.append(direct.render(cursor,n))
    y=np.concatenate(y);d=np.concatenate(d);wet=y-d
    args.out.mkdir(parents=True,exist_ok=True)
    for name,signal in (('unified_4ch.wav',y),('direct_4ch.wav',d),
                        ('reflections_only_4ch.wav',wet)):
        wavfile.write(str(args.out/name),sr,signal.astype(np.float32))
    metrics={'status':'PASS' if float(np.sqrt(np.mean(wet.astype(float)**2)))>1e-4 else 'FAIL',
        'path_length_direct_m':1.,'path_length_reflection_m':math.sqrt(17.),
        'delay_direct_ms':1000/343.,'delay_reflection_ms':1000*math.sqrt(17.)/343.,
        'reflection_rms':float(np.sqrt(np.mean(wet.astype(float)**2))),
        'pcm_channels':4,'max_abs':float(np.max(abs(y))),
        'physical_accuracy':'synthetic demo, not measured material validation',
        'renderer':unified.telemetry()}
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2));print('OUTPUT',args.out)
    return int(metrics['status']!='PASS')
if __name__=='__main__':
    raise SystemExit(main())
