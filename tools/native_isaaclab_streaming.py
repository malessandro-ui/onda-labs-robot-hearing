#!/usr/bin/env python3
"""Native Isaac Lab two-env end-to-end PCM streaming acceptance.

Uses real Isaac Lab robot/source pose tensors, an explicit analytic microphone
renderer, and a PCM-only streaming localizer. Does not synthesize acoustic audio
from PhysX itself. Does not change NVIDIA or Onda source files.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def angular_error(a,b):
    return abs((a-b+180)%360-180)


def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.tmp')
    with tmp.open('w') as fp:
        json.dump(payload,fp,indent=2,allow_nan=False)
        fp.flush();os.fsync(fp.fileno())
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=180)
    p.add_argument('--out',type=Path,default=Path('/ephemeral/onda-nvidia/onda-v015/docs/validation/native_isaaclab_streaming_v016/results.json'))
    p.add_argument('--shutdown',choices=['hard','graceful'],default='hard')
    args=p.parse_args()
    if args.steps<80: p.error('--steps must be >=80')
    output={'suite':'Onda PCM-first streaming native Isaac Lab validation','result':'FAIL',
            'limitations':['Reference analytic point-source renderer; Isaac Lab has no physical acoustic propagation in this test',
                           'Cartpole root stands in for microphone frame; not a calibrated humanoid head',
                           'Python-thread prototype, not hard real-time or multi-source tracking']}
    code=1
    app=None
    streams=[]
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True).app
        import gymnasium as gym
        import torch
        import isaaclab_tasks
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
        from onda_hearing.streaming import StreamConfig,Pose,StreamingHearing,GCCPHATSRP
        from onda_hearing.streaming.simulated_capture import render_point_source,DEFAULT_ARRAY_METERS
        cfg=CartpoleEnvCfg()
        cfg.scene.num_envs=2
        cfg.scene.env_spacing=4.
        cfg.episode_length_s=30.
        cfg.scene.sound_source=RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/OndaPCMEmitter',
            spawn=sim_utils.SphereCfg(radius=.05,
                   rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                   collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                   mass_props=sim_utils.MassPropertiesCfg(mass=.1)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5,0.,1.)))
        env=gym.make('Isaac-Cartpole-v0',cfg=cfg).unwrapped
        obs,_=env.reset()
        original_policy_shape=tuple(obs['policy'].shape)
        robot=env.scene['robot']
        rp=robot.data.root_pos_w.detach().cpu().numpy().copy()
        rq=robot.data.root_quat_w.detach().cpu().numpy().copy()
        angle=np.deg2rad([38.,-38.])
        fixed=rp.copy()
        fixed[:,0]+=1.5*np.cos(angle)
        fixed[:,1]+=1.5*np.sin(angle)
        src=env.scene['sound_source']
        src_p=torch.as_tensor(fixed,dtype=torch.float32,device=env.device)
        src.write_root_pose_to_sim(torch.cat((src_p,src.data.root_quat_w.clone()),dim=-1))
        rng=np.random.default_rng(42)
        mono=rng.standard_normal(int(args.steps*env.step_dt*16000)+16000).astype(np.float32)
        setup=StreamConfig(window_samples=2048,hop_samples=512,buffer_seconds=4.,
                           max_observation_age_ms=500,max_pose_gap_ms=120)
        for _ in range(2):
            streams.append(StreamingHearing(setup,0,GCCPHATSRP(DEFAULT_ARRAY_METERS)))
        for i,s in enumerate(streams):
            s.push_pose(Pose(0,tuple(map(float,rp[i])),tuple(map(float,rq[i,[1,2,3,0]]))))
        previous_sample=0
        valid=[0,0]
        errors=[[],[]]
        policy_ms=[]
        audio_push_ms=[]
        nonblocking_read_ms=[]
        for step in range(args.steps):
            actions=torch.zeros((2,env.action_manager.total_action_dim),device=env.device)
            start=time.perf_counter_ns()
            obs,reward,terminated,truncated,extras=env.step(actions)
            policy_ms.append((time.perf_counter_ns()-start)/1e6)
            assert tuple(obs['policy'].shape)==original_policy_shape
            if bool((terminated|truncated).any().item()):
                raise AssertionError('unexpected native episode reset in streaming acceptance')
            rp=robot.data.root_pos_w.detach().cpu().numpy().copy()
            rq=robot.data.root_quat_w.detach().cpu().numpy().copy()
            # Geometry remains in the native scene; no privileged source locations in observations.
            current_source=src.data.root_pos_w.detach().cpu().numpy().copy()
            if not np.allclose(current_source,fixed,atol=.001):
                raise AssertionError('native source moved')
            t_ns=round((step+1)*env.step_dt*1e9)
            end_sample=round((step+1)*env.step_dt*setup.sample_rate)
            frames=end_sample-previous_sample
            for i,s in enumerate(streams):
                q=tuple(map(float,rq[i,[1,2,3,0]]))
                s.push_pose(Pose(t_ns,tuple(map(float,rp[i])),q))
                pcm=render_point_source(mono,previous_sample,frames,setup.sample_rate,rp[i],q,current_source[i])
                st=time.perf_counter_ns()
                s.push_audio(pcm,previous_sample)
                audio_push_ms.append((time.perf_counter_ns()-st)/1e6)
            previous_sample=end_sample
            # Separate, nonblocking observation reads; policy never receives source pose.
            hearing_policy_features=[]
            for i,s in enumerate(streams):
                start=time.perf_counter_ns()
                features,hearing,metadata=s.policy_features(t_ns)
                hearing_policy_features.append(features)
                nonblocking_read_ms.append((time.perf_counter_ns()-start)/1e6)
                assert features.shape==(5,) and features.dtype==np.float32
                if hearing.valid:
                    valid[i]+=1
                    # Compare world-frame estimator against true source direction.
                    reference=math.degrees(math.atan2(current_source[i,1]-rp[i,1],current_source[i,0]-rp[i,0]))
                    error=angular_error(hearing.bearing_world_deg,reference)
                    errors[i].append(error)
            policy_hearing=torch.as_tensor(np.stack(hearing_policy_features),dtype=torch.float32,device=env.device)
            augmented_obs=dict(obs)
            augmented_obs['onda_hearing']=policy_hearing
            assert augmented_obs['onda_hearing'].shape==(2,5)
            assert tuple(augmented_obs['policy'].shape)==original_policy_shape
            if step in (0,39,79,args.steps-1):
                print('ONDA STREAM NATIVE: step',step+1, 'valid',valid,
                      'processed',[s.poll(t_ns)[1]['windows_processed'] for s in streams],flush=True)
        # Permit final worker publication; bounded wait is for test verification only.
        deadline=time.monotonic()+2.0
        while time.monotonic()<deadline and any(not x for x in errors):
            time.sleep(.02)
            for i,s in enumerate(streams):
                hearing,_=s.poll(round(args.steps*env.step_dt*1e9))
                if hearing.valid:
                    valid[i]+=1
                    ref=math.degrees(math.atan2(fixed[i,1]-rp[i,1],fixed[i,0]-rp[i,0]))
                    errors[i].append(angular_error(hearing.bearing_world_deg,ref))
        telemetry=[s.poll(round(args.steps*env.step_dt*1e9))[1] for s in streams]
        p75=[float(np.percentile(e,75)) if e else None for e in errors]
        for i in range(2):
            assert valid[i]>=5, f'env{i}: fewer than five valid acoustic observations'
            assert p75[i] is not None and p75[i]<=10, f'env{i}: angular error >10deg'
            assert telemetry[i]['audio_discontinuities']==0
            assert telemetry[i]['overwritten_audio_samples']==0
            assert telemetry[i]['worker_error'] is None
        summary=lambda seq:{'p50_ms':float(np.percentile(seq,50)),'p95_ms':float(np.percentile(seq,95))}
        output.update(result='PASS',task='Isaac-Cartpole-v0',gpu=torch.cuda.get_device_name(0),
                      torch=torch.__version__,num_envs=2,steps=args.steps,
                      valid_read_counts=valid,p75_angular_error_deg=p75,
                      telemetry=telemetry,
                      timing={'native_policy_physics_only':summary(policy_ms),
                              'pcm_ingestion':summary(audio_push_ms),
                              'nonblocking_policy_read':summary(nonblocking_read_ms)},
                      caveat='A step-to-step update is not a continuous hardware audio capture test.')
        code=0
        print('ONDA STREAM NATIVE: VALID BEARINGS PASS',valid,p75,flush=True)
        print('ONDA STREAM NATIVE: FINAL VERDICT PASS',flush=True)
    except BaseException as exc:
        output.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print('ONDA STREAM NATIVE: FAIL\n'+output['traceback'],flush=True)
    finally:
        shutdown_results=[s.close(timeout_s=5.) for s in streams]
        output['worker_clean_shutdown']=all(shutdown_results)
        atomic_json(args.out,output)
        print('ONDA STREAM NATIVE: SAVED',args.out,flush=True)
        if args.shutdown=='hard':
            print('ONDA STREAM NATIVE: FINITE HARNESS HARD EXIT (NOT PRODUCTION)',flush=True)
            sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if app is not None: app.close()
    return code


if __name__=='__main__':sys.exit(main())
