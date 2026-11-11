#!/usr/bin/env python3
"""Native MuJoCo moving-door acoustic acceptance: dynamic hinge -> geometry -> 4ch PCM.

No robot-perception model or game-engine audio playback required. All WAVs are
raw *simulated microphone samples*. Door dynamics are integrated by MuJoCo,
and the acoustic surface is rebuilt from data.geom_xpos and data.geom_xmat
before each corresponding audio block. Does not modify MuJoCo or Onda assets.

Installation: pip install mujoco (only if not already present). The rest runs
with existing Onda v0.17/v0.19 + this v0.20 add-on.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import wave

import numpy as np

from onda_hearing.acoustic_scene import (Pose3, Binding, Schedule, Emitter,
                                         MicArray, AudioAsset, RendererConfig, load_wav)
from onda_hearing.acoustic_scene.physical import (
    TransmissionMaterial, AcousticSurface, AcousticGeometry,
    GeometryAwareAcousticRenderer, PhysicalConfig)
from onda_hearing.acoustic_scene.physical_adapters import surface_from_mujoco_box
from onda_hearing.acoustic_scene.unified_physics import (
    ReflectionMaterial, UnifiedPhysicsConfig, UnifiedGeometryAcousticRenderer)

SR = 16000
# AUDIO: synthetic coefficients for demonstration, NOT measured material data.
BAND = (125.,500.,1000.,4000.,8000.)
WALL_TL = TransmissionMaterial(BAND,(42.,48.,55.,64.,70.),'DEMO_SYNTHETIC_WALL_UNCALIBRATED')
DOOR_TL = TransmissionMaterial(BAND,(6.,10.,16.,29.,35.),'DEMO_SYNTHETIC_DOOR_UNCALIBRATED')
REFLECT = ReflectionMaterial(BAND,(.15,.17,.22,.29,.37),(.02,.04,.07,.12,.17),
                             'DEMO_SYNTHETIC_REFLECTION_UNCALIBRATED')

# Hinge at the left jamb. At 0 radians the panel spans y=-0.5..+0.5 in
# separating wall x=0. At +90 degrees it swings toward negative x and
# leaves the central x-axis sound path unblocked.
MJCF = '''<mujoco model="onda_unified_moving_door_v020">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="3.0 2.4 0.05" contype="0" conaffinity="0"/>
    <geom name="side_pos" type="box" pos="0 2.35 1.25" size="3.0 0.05 1.25" contype="0" conaffinity="0"/>
    <geom name="side_neg" type="box" pos="0 -2.35 1.25" size="3.0 0.05 1.25" contype="0" conaffinity="0"/>
    <geom name="partition_neg" type="box" pos="0 -1.45 1.25" size="0.03 0.95 1.25" contype="0" conaffinity="0"/>
    <geom name="partition_pos" type="box" pos="0 1.45 1.25" size="0.03 0.95 1.25" contype="0" conaffinity="0"/>
    <geom name="lintel" type="box" pos="0 0 2.25" size="0.03 0.5 0.25" contype="0" conaffinity="0"/>
    <body name="robot" pos="1.45 0.12 1.35"><site name="robot_mic_center" pos="0 0 0" size=".015"/></body>
    <body name="phone" pos="-1.45 -0.20 1.20"><site name="phone_sound" pos="0 0 0" size=".015"/></body>
    <body name="door" pos="0 -0.5 0">
      <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.57" limited="true" damping="25"/>
      <geom name="door_panel" type="box" pos="0 0.5 1" size="0.025 0.50 1.0"
            mass="12" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="door_motor" joint="door_hinge" kp="160" ctrllimited="true" ctrlrange="0 1.50"/>
  </actuator>
</mujoco>'''


def face_from_box(model,data,name,axis,positive,material):
    """Exact world-space outer face of a MuJoCo box as a thin reflective quad."""
    import mujoco
    index = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
    if index<0:
        raise KeyError(name)
    if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f'Expected box: {name}')
    half=np.asarray(model.geom_size[index],dtype=np.float64)
    center=np.asarray(data.geom_xpos[index],dtype=np.float64)
    rotation=np.asarray(data.geom_xmat[index],dtype=np.float64).reshape(3,3)
    other=[a for a in (0,1,2) if a != axis]
    pts=[]
    for u,v in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        p=np.zeros(3,dtype=float)
        p[axis]=half[axis]*(1 if positive else -1)
        p[other[0]]=u*half[other[0]]
        p[other[1]]=v*half[other[1]]
        pts.append(tuple(center+rotation@p))
    return AcousticSurface.quad(f'mujoco:{name}:face',tuple(pts),material)


def synth_audio(seconds):
    t=np.arange(int(seconds*SR),dtype=float)/SR
    # Contains low and high tones for meaningful spectral-band comparisons.
    # This is NOT a natural telephone recording.
    x=.065*np.sin(2*np.pi*440*t)+.032*np.sin(2*np.pi*3000*t)
    return AudioAsset('phone',x.astype(np.float32),SR)


def wav_float32(path,audio):
    # IEEE FLOAT WAV format 3 via scipy; no normalization or clipping.
    from scipy.io import wavfile
    wavfile.write(str(path),SR,np.asarray(audio,dtype=np.float32))


def band_level(signal,freq,t_start,t_end):
    x=np.asarray(signal,dtype=np.float64)[int(t_start*SR):int(t_end*SR)]
    n=np.arange(len(x))
    if len(x) < 256:
        raise ValueError('invalid analysis interval')
    # Lock-in amplitude estimate with Hann window to reject non-integer cycles.
    w=np.hanning(len(x))
    amp=2.*abs(np.sum(w*x*np.exp(-2j*np.pi*freq*n/SR)))/np.sum(w)
    return float(amp)


def run(args):
    import mujoco
    model=mujoco.MjModel.from_xml_string(MJCF)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    if model.nu!=1:
        raise RuntimeError('expected exactly one door actuator')
    timestep=float(model.opt.timestep)
    step_samples=int(round(timestep*SR))
    if not np.isclose(step_samples/SR,timestep,atol=1e-12):
        raise RuntimeError('audio and physics clocks cannot be aligned exactly')
    if args.chunk_samples%step_samples:
        raise ValueError(f'chunk_samples must be divisible by {step_samples}')
    length=int(round(args.duration_s*SR))
    if length%args.chunk_samples:
        raise ValueError('duration must contain an integer number of audio chunks')
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'door_hinge')
    qid=int(model.jnt_qposadr[jid])
    robot_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    phone_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'phone')
    robot_center=tuple(map(float,data.xpos[robot_id]))
    source_center=tuple(map(float,data.xpos[phone_id]))
    mic_local=((0.,0.,0.),(0.,.035,0.),(0.,-.035,0.),(.035,0.,0.))
    mic=MicArray(Binding(world=Pose3(robot_center)),mic_local)
    emitter=Emitter('telephone','phone',Binding(world=Pose3(source_center)),
                    Schedule(onset_s=0.,loop=True))
    source_asset=load_wav(args.wav,'phone') if args.wav else synth_audio(args.duration_s)
    if source_asset.sample_rate != SR:
        raise ValueError('input WAV must be 16kHz mono PCM')
    # The physical scene geometry comes from the native MuJoCo model/data.
    static=[]
    for name in ('partition_neg','partition_pos','lintel'):
        static.append(surface_from_mujoco_box(model,data,name,WALL_TL))
    static.extend((face_from_box(model,data,'floor',2,True,WALL_TL),
                   face_from_box(model,data,'side_pos',1,False,WALL_TL),
                   face_from_box(model,data,'side_neg',1,True,WALL_TL)))
    door=surface_from_mujoco_box(model,data,'door_panel',DOOR_TL)
    geo=AcousticGeometry(tuple(static)+(door,))
    refs={s.surface_id:REFLECT for s in static if ':face' in s.surface_id}
    cfg=RendererConfig(sample_rate=SR,noise_std=0.,seed=7,clip_output=False)
    combined=UnifiedGeometryAcousticRenderer(
        cfg,mic,[emitter],{'phone':source_asset},geo,refs,
        unified_config=UnifiedPhysicsConfig(
            max_specular_order=args.order,
            geometry_step_samples=args.chunk_samples,
            transition_samples=min(320,args.chunk_samples*2),
            max_path_states=512,
        ))
    direct=GeometryAwareAcousticRenderer(
        cfg,mic,[emitter],{'phone':source_asset},geo,
        physical_config=PhysicalConfig(geometry_step_samples=args.chunk_samples,
                                      transition_samples=min(320,args.chunk_samples*2)))
    started_wall=time.perf_counter()
    chunks=[];dry_chunks=[];angles=[];path_counts=[]
    for offset in range(0,length,args.chunk_samples):
        t=offset/SR
        # Geometry is captured at the SAME simulation time as audio block start.
        panel=surface_from_mujoco_box(model,data,'door_panel',DOOR_TL)
        geo.put(panel)
        chunks.append(combined.render(offset,args.chunk_samples))
        dry_chunks.append(direct.render(offset,args.chunk_samples))
        angle=float(data.qpos[qid])
        angles.append((t,angle))
        path_counts.append(combined.active_specular_paths)
        # Advance native MuJoCo dynamics to the NEXT audio frame boundary.
        for _ in range(args.chunk_samples//step_samples):
            sim_t=float(data.time)
            data.ctrl[0]=0. if sim_t<1. else (1.5 if sim_t<2.9 else 0.)
            mujoco.mj_step(model,data)
    physics_and_render_wall_s=time.perf_counter()-started_wall
    samples=np.concatenate(chunks)
    direct_samples=np.concatenate(dry_chunks)
    assert samples.shape == (length,4)
    assert np.isfinite(samples).all()
    assert np.isfinite(direct_samples).all()
    assert len(angles) == length//args.chunk_samples
    closed_angle=max(a for t,a in angles if t<.9)
    open_angle=max(a for t,a in angles if 2.1<t<2.9)
    movement=open_angle-closed_angle
    closed_window=(.45,.8)
    open_window=(2.3,2.65)
    atten={}
    for freq in (440.,3000.):
        closed_amp=band_level(samples[:,0],freq,*closed_window)
        open_amp=band_level(samples[:,0],freq,*open_window)
        atten[str(freq)]=20*math.log10(max(open_amp,1e-12)/max(closed_amp,1e-12))
    wet= samples.astype(np.float64)-direct_samples.astype(np.float64)
    wet_rms=float(np.sqrt(np.mean(wet**2)))
    total_rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
    report={
        'suite':'onda_v020_native_mujoco_moving_door_unified_pcm',
        'result':'PASS' if movement > .75 and wet_rms > 1.e-5 else 'FAIL',
        'physics_claim':'geometric_specular_quasistatic_not_calibrated',
        'simulator':'native MuJoCo hinged door with integrated position actuator',
        'door_initial_angle_rad':angles[0][1],
        'door_max_closed_angle_rad':closed_angle,
        'door_max_open_angle_rad':open_angle,
        'door_motion_span_rad':movement,
        'open_minus_closed_band_db':atten,
        'four_channel_pcm_rms':total_rms,
        'physics_and_render_wall_s':physics_and_render_wall_s,
        'simulated_seconds_per_wall_second':args.duration_s/max(physics_and_render_wall_s,1.e-12),
        'reflection_only_pcm_rms':wet_rms,
        'peak_abs_pcm':float(np.max(abs(samples))),
        'active_specular_paths_max':max(path_counts),
        'audio_samples_per_channel':len(samples),
        'material_provenance':'DEMO_SYNTHETIC_UNCALIBRATED',
        'renderer':combined.telemetry(),
        'limitations':['door samples quasi-static each audio chunk',
                       'no diffraction','no door seal leakage','no late diffuse reverb',
                       'no validated measured RIR or acoustic impedance'],
    }
    args.output.mkdir(parents=True,exist_ok=True)
    wav_float32(args.output/'mujoco_door_unified_4ch_float.wav',samples)
    wav_float32(args.output/'mujoco_door_direct_only_4ch_float.wav',direct_samples)
    wav_float32(args.output/'mujoco_door_reflections_only_4ch_float.wav',wet)
    with (args.output/'mujoco_door_angles.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_s','hinge_angle_rad','active_specular_paths'])
        writer.writerows((t,a,n) for (t,a),n in zip(angles,path_counts))
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('OUTPUT:',args.output)
    return int(report['result']!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/v020_mujoco_moving_door'))
    p.add_argument('--wav',type=Path,default=None,help='optional 16k mono PCM telephone WAV')
    p.add_argument('--order',type=int,choices=(0,1,2),default=1)
    p.add_argument('--duration-s',type=float,default=5.12)
    p.add_argument('--chunk-samples',type=int,default=256)
    args=p.parse_args()
    if not 5. <= args.duration_s <= 10.:p.error('duration-s must be 5..10')
    if not 32 <= args.chunk_samples <= 1024:p.error('chunk-samples must be 32..1024')
    return run(args)

if __name__=='__main__':
    raise SystemExit(main())
