#!/usr/bin/env python3
"""Native MuJoCo moving phone -> v0.17 acoustic PCM -> v0.16 spatial + BEATs.

A test harness, not a trainable policy. Simulated acoustic propagation is direct-path.
Spatial truth is used ONLY by evaluator, never passed to policy observation.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from onda_v017_native_common import (MIC_LOCAL,Accumulator,atomic_json,make_brain,
    make_hearing,validate_metrics,truth_bearing)

MJCF='''<mujoco model="onda_scene">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_head" pos="0 0 0">
      <geom type="sphere" size="0.08"/>
    </body>
    <body name="phone" mocap="true" pos="2 0 0">
      <geom type="box" size="0.07 0.1 0.03"/>
    </body>
  </worldbody>
</mujoco>'''

def run(args):
    import mujoco
    from onda_hearing.acoustic_scene import (Binding,Emitter,Schedule,MicArray,RendererConfig,
        AcousticSceneRenderer,PoseTimeline,MuJoCoPoseSnapshot,load_wav)
    from onda_hearing.streaming.core import Pose
    sr=16000
    sound=load_wav(args.wav,'phone')
    if sound.sample_rate!=sr:raise ValueError('expected mono 16kHz WAV')
    model=mujoco.MjModel.from_xml_string(MJCF);data=mujoco.MjData(model)
    source=MuJoCoPoseSnapshot(model,data)
    timeline=PoseTimeline(maxlen=8192,max_hold_s=.02)
    microphones=MicArray(Binding(object_path='robot_head'),MIC_LOCAL)
    renderer=AcousticSceneRenderer(RendererConfig(sample_rate=sr,noise_std=args.noise_std,seed=73),
        microphones,[Emitter('ringing_phone','phone',Binding(object_path='phone'),
            Schedule(onset_s=.15,loop=True))],{'phone':sound},timeline)
    brain=make_brain(args)
    hearing=make_hearing(brain,MIC_LOCAL)
    acc=Accumulator();last_metrics={};started=time.monotonic();cursor=0;phone_y=[]
    def capture():
        for name in ('robot_head','phone'):
            timeline.append(name,float(data.time),source.capture('body',name))
    def truth(capture_ns):
        t=capture_ns/1e9
        try:return truth_bearing(timeline.pose_at('robot_head',t),timeline.pose_at('phone',t))
        except (ValueError,KeyError):return None
    try:
        mujoco.mj_forward(model,data);capture()
        initial=timeline.pose_at('robot_head',0.)
        hearing.push_pose(Pose(0,initial.xyz,initial.quat_xyzw))
        total=int(round(args.duration_s*sr))
        while cursor<total:
            n=min(args.chunk_samples,total-cursor)
            until=(cursor+n)/sr
            # Record enough future pose states to render at the current reception time.
            while data.time<until+.003:
                data.mocap_pos[0]=[2.,.65*math.sin(1.1*data.time),0.]
                mujoco.mj_step(model,data)
                capture()
                phone_y.append(float(source.capture('body','phone').xyz[1]))
            pcm=renderer.render(cursor,n)
            if pcm.shape!=(n,4) or not np.isfinite(pcm).all():raise AssertionError('bad rendered audio')
            end_ns=round((cursor+n)*1e9/sr)
            current=timeline.pose_at('robot_head',until)
            hearing.push_pose(Pose(end_ns,current.xyz,current.quat_xyzw))
            hearing.push_audio(pcm,cursor)
            cursor+=n
            _,_,last_metrics=acc.observe(hearing,end_ns,truth,7+len(brain.labels))
            # A controlled demo should give the asynchronous BEATs worker CPU/GPU time.
            # Source-clock and publication-wall-clock measures remain distinct.
            if args.pace:
                remaining=cursor/sr-(time.monotonic()-started)
                if remaining>0:time.sleep(min(remaining,.1))
        # Bounded drain: workers may still be processing the latest inference window.
        deadline=time.monotonic()+args.drain_s
        while time.monotonic()<deadline and (not acc.semantic_unique or len(acc.spatial_unique)<10):
            time.sleep(.03)
            _,_,last_metrics=acc.observe(hearing,round(cursor*1e9/sr),truth,7+len(brain.labels))
        _,_,last_metrics=acc.observe(hearing,round(cursor*1e9/sr),truth,7+len(brain.labels))
        errors=validate_metrics(last_metrics,acc,min_spatial=10,max_p75=args.max_angular_error_deg)
        motion_span=max(phone_y)-min(phone_y) if phone_y else 0.
        if args.expected_label is not None and acc.labels.get(args.expected_label,0.)<args.min_label_score:
            errors.append(f'expected_label_not_recognized: {args.expected_label}')
        if motion_span<.25:errors.append('native_mocap_source_did_not_move_enough')
        if renderer.telemetry()['clipped_samples'] or renderer.telemetry()['overrange_samples']:
            errors.append('rendered_PCM_clipping_or_overrange')
        report={'suite':'native MuJoCo object-bound unified hearing','result':'PASS' if not errors else 'FAIL',
                'limitations':['Analytic direct-path renderer; no wall occlusion or full room acoustics',
                   'Acoustic event scores apply to full mono window, not uniquely to localized source',
                   'Synthetic ring is not a natural telephone semantic benchmark'],
                'duration_sim_s':cursor/sr,'elapsed_wall_s':time.monotonic()-started,
                'samples_per_channel':cursor,'audio_channels':4,'source_y_motion_span_m':motion_span,
                'scene_poses':'native MuJoCo mocap body via MuJoCoPoseSnapshot',
                'emitter_binding':'phone','microphone_binding':'robot_head',
                'renderer_telemetry':renderer.telemetry(),
                'observation':acc.report(),'streaming_telemetry':last_metrics,'failures':errors}
        return report
    finally:
        if not hearing.close():raise RuntimeError('unified workers did not shut down cleanly')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav',required=True)
    p.add_argument('--checkpoint',default=None);p.add_argument('--labels',default=None)
    p.add_argument('--beats-dir',default=None);p.add_argument('--device',default='cuda')
    p.add_argument('--duration-s',type=float,default=8.)
    p.add_argument('--chunk-samples',type=int,default=512)
    p.add_argument('--noise-std',type=float,default=.005)
    p.add_argument('--max-angular-error-deg',type=float,default=15.)
    p.add_argument('--expected-label',default=None,help='Optional semantic ground-truth gate')
    p.add_argument('--min-label-score',type=float,default=.15)
    p.add_argument('--drain-s',type=float,default=6.)
    p.add_argument('--pace',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--out',type=Path,default=Path('docs/validation/v017_native/mujoco_unified.json'))
    args=p.parse_args()
    if not 4<=args.duration_s<=180 or not 64<=args.chunk_samples<=16000:p.error('invalid duration/chunk')
    if not 0<=args.noise_std<=.1:p.error('invalid noise')
    if not 0<=args.min_label_score<=1:p.error('invalid minimum semantic score')
    try: result=run(args)
    except BaseException as e:
        result={'suite':'native MuJoCo object-bound unified hearing','result':'FAIL',
                'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}
    atomic_json(args.out,result)
    print('ONDA V017 MUJOCO:',result['result'],result.get('failures',result.get('error')))
    print('ONDA V017 MUJOCO: SAVED',args.out)
    return int(result['result']!='PASS')
if __name__=='__main__':sys.exit(main())
