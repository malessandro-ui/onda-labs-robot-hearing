#!/usr/bin/env python3
"""Headless native Isaac Lab moving USD prim -> v0.17 renderer -> unified PCM hearing.

Use Isaac Lab's actual RigidObject scene tensors; verifies corresponding USD
prims exist. Does not claim PhysX produces audible audio. Finite harness hard
exit is solely for known Kit shutdown stalls on this tested Shadeform image.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from onda_v017_native_common import (MIC_LOCAL,Accumulator,atomic_json,make_brain,
    make_hearing,validate_metrics,truth_bearing)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav',required=True)
    p.add_argument('--checkpoint',default=None);p.add_argument('--labels',default=None)
    p.add_argument('--beats-dir',default=None);p.add_argument('--device',default='cuda')
    p.add_argument('--duration-sim-s',type=float,default=8.)
    p.add_argument('--max-angular-error-deg',type=float,default=15.)
    p.add_argument('--expected-label',default=None,help='Optional semantic ground-truth gate')
    p.add_argument('--min-label-score',type=float,default=.15)
    p.add_argument('--out',type=Path,default=Path('docs/validation/v017_native/isaaclab_unified.json'))
    p.add_argument('--shutdown',choices=['hard','graceful'],default='hard')
    args=p.parse_args()
    if not 4<=args.duration_sim_s<=120:p.error('duration must be 4..120 sim seconds')
    if not 0<=args.min_label_score<=1:p.error('invalid minimum semantic score')
    payload={'suite':'native Isaac Lab scene-object unified hearing','result':'FAIL',
             'limitations':['Analytic Onda direct-path rendering, not physically simulated airborne sound',
                 'Cartpole with native kinematic scene objects, not a humanoid',
                 'Simulation and wall-time clocks are different',
                 'Semantic classifications apply to whole audio windows, not identified spatial tracks']}
    app=None; hearing=None;code=1
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True).app
        import gymnasium as gym
        import torch
        import omni.usd
        import isaaclab_tasks
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
        from onda_hearing.acoustic_scene import (Binding,Emitter,Schedule,MicArray,
            RendererConfig,AcousticSceneRenderer,Pose3,PoseTimeline,load_wav)
        from onda_hearing.streaming.core import Pose
        sr=16000
        sound=load_wav(args.wav,'phone')
        if sound.sample_rate!=sr:raise ValueError('expected mono 16kHz WAV')
        cfg=CartpoleEnvCfg();cfg.scene.num_envs=1;cfg.scene.env_spacing=4.
        cfg.episode_length_s=max(40.,args.duration_sim_s+30.)
        obj_cfg=lambda radius: sim_utils.SphereCfg(radius=radius,
             rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
             mass_props=sim_utils.MassPropertiesCfg(mass=.1))
        cfg.scene.hearing_rig=RigidObjectCfg(
             prim_path='{ENV_REGEX_NS}/OndaV017Head',spawn=obj_cfg(.07),
             init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,1.)))
        cfg.scene.phone=RigidObjectCfg(
             prim_path='{ENV_REGEX_NS}/OndaV017Phone',spawn=obj_cfg(.045),
             init_state=RigidObjectCfg.InitialStateCfg(pos=(2.,.4,1.)))
        env=gym.make('Isaac-Cartpole-v0',cfg=cfg).unwrapped
        obs,_=env.reset()
        original_shape=tuple(obs['policy'].shape)
        rig=env.scene['hearing_rig'];phone=env.scene['phone']
        root=rig.data.root_pos_w[0].detach().cpu().numpy().copy()
        stage=omni.usd.get_context().get_stage()
        rigpath='/World/envs/env_0/OndaV017Head'
        phonepath='/World/envs/env_0/OndaV017Phone'
        if not stage.GetPrimAtPath(rigpath).IsValid() or not stage.GetPrimAtPath(phonepath).IsValid():
            raise AssertionError('native USD microphone or phone prim missing')
        timeline=PoseTimeline(maxlen=8192,max_hold_s=.03)
        array=MicArray(Binding(object_path=rigpath),MIC_LOCAL)
        render=AcousticSceneRenderer(RendererConfig(sample_rate=sr,noise_std=.005,seed=81),
               array,[Emitter('ringing_phone','phone',Binding(object_path=phonepath),
                      Schedule(onset_s=.15,loop=True))],{'phone':sound},timeline)
        brain=make_brain(args)
        hearing=make_hearing(brain,MIC_LOCAL)
        acc=Accumulator();last_metrics={};cursor=0;phone_y=[];rig_yaw=[]
        start_wall=time.monotonic();dt=env.step_dt
        def current_pose(obj):
            pos=obj.data.root_pos_w[0].detach().cpu().numpy()
            q=obj.data.root_quat_w[0].detach().cpu().numpy() # WXYZ in Isaac
            return Pose3(tuple(map(float,pos)),tuple(map(float,q[[1,2,3,0]])))
        def append(t):
            timeline.append(rigpath,t,current_pose(rig))
            timeline.append(phonepath,t,current_pose(phone))
        def truth(ts):
            t=ts/1e9
            try:return truth_bearing(timeline.pose_at(rigpath,t),timeline.pose_at(phonepath,t))
            except (KeyError,ValueError):return None
        append(0.)
        start_pose=timeline.pose_at(rigpath,0.)
        hearing.push_pose(Pose(0,start_pose.xyz,start_pose.quat_xyzw))
        steps=math.ceil(args.duration_sim_s/dt)
        for k in range(steps):
            t=(k+1)*dt
            a=.30*math.sin(.8*t)
            q=torch.tensor([[math.cos(a/2),0.,0.,math.sin(a/2)]],device=env.device,dtype=torch.float32)
            rig_pos=torch.tensor(np.asarray(root,dtype=np.float32).reshape(1,3),device=env.device)
            rig.write_root_pose_to_sim(torch.cat((rig_pos,q),dim=1))
            phone_pos=np.array(root,dtype=np.float32);phone_pos[0]+=2.;phone_pos[1]+=.7*math.sin(.7*t)
            phone_xyz=torch.tensor(phone_pos.reshape(1,3),device=env.device)
            # Phone orientation is fixed: WXYZ identity.
            fixed_q=torch.tensor([[1.,0.,0.,0.]],device=env.device)
            phone.write_root_pose_to_sim(torch.cat((phone_xyz,fixed_q),dim=1))
            actions=torch.zeros((1,env.action_manager.total_action_dim),device=env.device)
            obs,reward,terminated,truncated,extras=env.step(actions)
            if tuple(obs['policy'].shape)!=original_shape:raise AssertionError('policy observation changed')
            if not torch.isfinite(reward).all():raise AssertionError('nonfinite rewards')
            if abs(float(current_pose(phone).xyz[1])-float(phone_pos[1]))>.03:
                raise AssertionError('native phone position did not follow commanded trajectory')
            append(t)
            phone_y.append(current_pose(phone).xyz[1])
            rig_yaw.append(2*math.degrees(math.atan2(current_pose(rig).quat_xyzw[2],current_pose(rig).quat_xyzw[3])))
            end=round(t*sr);n=end-cursor
            if n<=0:raise AssertionError('invalid sample advance')
            pcm=render.render(cursor,n)
            if pcm.shape!=(n,4) or not np.isfinite(pcm).all():raise AssertionError('invalid PCM')
            cap_ns=round(end*1e9/sr)
            pose=current_pose(rig)
            hearing.push_pose(Pose(cap_ns,pose.xyz,pose.quat_xyzw))
            hearing.push_audio(pcm,cursor)
            cursor=end
            _,_,last_metrics=acc.observe(hearing,cap_ns,truth,7+len(brain.labels))
            # Attach only sensor-derived features to the policy-facing observation.
            policy_obs=dict(obs)
            policy_obs['onda_hearing']=torch.as_tensor(acc.last_policy_vector.reshape(1,-1),device=env.device)
            if tuple(policy_obs['onda_hearing'].shape)!=(1,7+len(brain.labels)):
                raise AssertionError('unexpected unified policy observation shape')
            if k%60==0:print('ONDA V017 ISAAC:',round(t,2),'sim s',
                len(acc.spatial_unique),'spatial',len(acc.semantic_unique),'semantic',flush=True)
            # Keep semantic evaluation paced; source-clock still uses simulation sample indices.
            remaining=t-(time.monotonic()-start_wall)
            if remaining>0:time.sleep(min(remaining,.06))
        deadline=time.monotonic()+6.
        while time.monotonic()<deadline and (len(acc.semantic_unique)<1 or len(acc.spatial_unique)<10):
            time.sleep(.03)
            _,_,last_metrics=acc.observe(hearing,round(cursor*1e9/sr),truth,7+len(brain.labels))
        _,_,last_metrics=acc.observe(hearing,round(cursor*1e9/sr),truth,7+len(brain.labels))
        failures=validate_metrics(last_metrics,acc,min_spatial=10,max_p75=args.max_angular_error_deg)
        phone_span=max(phone_y)-min(phone_y) if phone_y else 0.
        rig_span=max(rig_yaw)-min(rig_yaw) if rig_yaw else 0.
        if args.expected_label is not None and acc.labels.get(args.expected_label,0.)<args.min_label_score:
            failures.append(f'expected_label_not_recognized: {args.expected_label}')
        if phone_span<.25:failures.append('USD_scene_phone_object_did_not_move_enough')
        if rig_span<5.:failures.append('USD_scene_microphone_rig_did_not_rotate')
        if render.telemetry()['clipped_samples'] or render.telemetry()['overrange_samples']:
            failures.append('rendered_PCM_clipping_or_overrange')
        payload.update(result='PASS' if not failures else 'FAIL',failures=failures,
          sim_duration_s=cursor/sr,wall_duration_s=time.monotonic()-start_wall,
          steps=steps,task='Isaac-Cartpole-v0',gpu=torch.cuda.get_device_name(0),
          scene_prims=[rigpath,phonepath],source_y_motion_span_m=phone_span,rig_yaw_span_deg=rig_span,
          source_pose='native RigidObject root pose; matching USD prim exists',
          microphone_pose='native RigidObject root pose; matching USD prim exists',
          renderer_telemetry=render.telemetry(),observation=acc.report(),streaming_telemetry=last_metrics)
        code=0 if not failures else 1
        print('ONDA V017 ISAAC: VERDICT',payload['result'],failures,flush=True)
    except BaseException as e:
        payload.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        print('ONDA V017 ISAAC: ERROR\n'+payload['traceback'],flush=True)
    finally:
        if hearing is not None:
            closed=hearing.close()
            payload['worker_clean_shutdown']=bool(closed)
            if not closed:
                payload['result']='FAIL';payload.setdefault('failures',[]).append('unclean_shutdown');code=1
        atomic_json(args.out,payload)
        print('ONDA V017 ISAAC: SAVED',args.out,flush=True)
        if args.shutdown=='hard':
            print('ONDA V017 ISAAC: FINITE KIT HARNESS HARD EXIT',flush=True)
            sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if app is not None:app.close()
    return code
if __name__=='__main__':sys.exit(main())
