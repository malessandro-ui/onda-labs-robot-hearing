#!/usr/bin/env python3
"""Native Isaac Lab v0.16 streaming motion soak with strict bounded telemetry.

Simulation-clock duration and real elapsed time are separate. Never use sim
capture timestamps to assert wall-clock end-to-end capture latency.

The kinematic microphone rig physically rotates in USD; a stationary analytic
source is rendered as 4-channel PCM. Not a sound-propagation PhysX simulation.
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
from onda_soak_common import (LatencyHistogram, Checkpoints, CyclicSource, angle_error,
                              atomic_json, rss_bytes, sample_timestamp_ns, yaw_at_seconds,
                              yaw_xyzw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration-sim-s',type=float,default=180.)
    p.add_argument('--checkpoint-s',type=float,default=60.)
    p.add_argument('--overload-ms',type=float,default=0.,help='Artificial slow estimator during two simulation-time intervals')
    p.add_argument('--out',type=Path,default=Path('docs/validation/streaming_v016_soak/native_motion_soak.json'))
    p.add_argument('--shutdown',choices=['hard','graceful'],default='hard')
    args=p.parse_args()
    if not 10<=args.duration_sim_s<=3600 or args.checkpoint_s<=0:p.error('duration must be 10-3600 sim seconds')
    payload={'suite':'Onda native Isaac Lab streaming rotating-rig soak','result':'FAIL',
             'limitations':['Analytic point-source PCM from native rig/source geometry, no physical acoustic propagation',
                            'Kinematic rigid microphone rig, not articulated robot head',
                            'Simulation time differs from wall time; latency clocks are not interchangeable',
                            'Python worker, single-source planar localizer, not production hardware']}
    app=None
    streams=[]
    checkpoint=None
    code=1
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True).app
        import gymnasium as gym
        import torch
        import isaaclab_tasks
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
        from onda_hearing.streaming import StreamConfig, Pose, StreamingHearing, GCCPHATSRP
        from onda_hearing.streaming.simulated_capture import DEFAULT_ARRAY_METERS
        from threading import Event
        slow_phase=Event()
        class OptionalSlowEstimator:
            def __init__(self, real): self.real=real
            def estimate(self, pcm):
                if slow_phase.is_set(): time.sleep(args.overload_ms/1000.)
                return self.real.estimate(pcm)
        cfg=CartpoleEnvCfg()
        cfg.scene.num_envs=2
        cfg.scene.env_spacing=4.
        cfg.episode_length_s=max(args.duration_sim_s+30.,60.)
        sphere=sim_utils.SphereCfg(radius=.05,
              rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
              mass_props=sim_utils.MassPropertiesCfg(mass=.1))
        cfg.scene.hearing_rig=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/OndaMovingMicRig',
                                            spawn=sphere,
                                            init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,1.)))
        cfg.scene.sound_source=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/OndaWorldFixedSound',
             spawn=sim_utils.SphereCfg(radius=.04,
                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                  collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                  mass_props=sim_utils.MassPropertiesCfg(mass=.1)),
             init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5,0,1.)))
        env=gym.make('Isaac-Cartpole-v0',cfg=cfg).unwrapped
        obs,_=env.reset()
        baseline_policy_shape=tuple(obs['policy'].shape)
        rig=env.scene['hearing_rig']
        src=env.scene['sound_source']
        bases=rig.data.root_pos_w.detach().cpu().numpy().copy()
        srcfixed=bases.copy()
        bearings=np.array([38.,-38.])
        srcfixed[:,0]+=1.5*np.cos(np.deg2rad(bearings))
        srcfixed[:,1]+=1.5*np.sin(np.deg2rad(bearings))
        desired_source=torch.as_tensor(srcfixed,device=env.device,dtype=torch.float32)
        src.write_root_pose_to_sim(torch.cat([desired_source,src.data.root_quat_w.clone()],axis=1))
        config=StreamConfig(sample_rate=16000,channels=4,window_samples=2048,hop_samples=512,
                            buffer_seconds=4.,max_pose_gap_ms=120,max_observation_age_ms=350.)
        for i in range(2):
            s=StreamingHearing(config,0,OptionalSlowEstimator(GCCPHATSRP(DEFAULT_ARRAY_METERS)))
            s.push_pose(Pose(0,tuple(map(float,bases[i])),yaw_xyzw(0)))
            streams.append(s)
        mono=CyclicSource()
        policy_duration=LatencyHistogram()
        ingestion_duration=LatencyHistogram()
        reading_duration=LatencyHistogram()
        unique_valid=[0,0];valid_reads=[0,0];accuracy_errors=[0,0]
        angular_error_hist=[LatencyHistogram(bin_ms=0.1,max_ms=180),LatencyHistogram(bin_ms=.1,max_ms=180)]
        sim_capture_age_hist=[LatencyHistogram(),LatencyHistogram()]
        previously_seen=[-1,-1]
        last_checkpoint=0
        policy_deadline_misses=0
        rss_start=rss_bytes();rss_warmup=None;rss_peak=rss_start
        checkpoint=Checkpoints(args.out.with_suffix('.checkpoints.jsonl'))
        planned_steps=math.ceil(args.duration_sim_s/env.step_dt)
        previous_sample=0
        wall_start=time.monotonic()
        for step in range(planned_steps):
            sim_seconds=(step+1)*env.step_dt
            yaw=yaw_at_seconds(sim_seconds)
            if args.overload_ms > 0 and any(
                a*args.duration_sim_s <= sim_seconds < a*args.duration_sim_s + min(8., max(.75,args.duration_sim_s*.025))
                for a in (.25,.65)):
                slow_phase.set()
            else:
                slow_phase.clear()
            a=math.radians(yaw)/2
            desired_quat=torch.as_tensor([[math.cos(a),0.,0.,math.sin(a)]]*2,
                                         dtype=torch.float32,device=env.device)
            rig.write_root_pose_to_sim(torch.cat([torch.as_tensor(bases,device=env.device,dtype=torch.float32),
                                                  desired_quat],axis=1))
            start=time.perf_counter_ns()
            actions=torch.zeros((2,env.action_manager.total_action_dim),device=env.device)
            obs,rewards,term,trunc,extra=env.step(actions)
            policy_ms=(time.perf_counter_ns()-start)/1e6
            policy_duration.add(policy_ms)
            if policy_ms>env.step_dt*1000:policy_deadline_misses+=1
            if tuple(obs['policy'].shape)!=baseline_policy_shape:
                raise AssertionError('native policy observations changed')
            # Cartpole can reset due to its own task termination. This is not a
            # PCM reset; we track its frequency and retain the independent rig.
            if not torch.isfinite(rewards).all():raise AssertionError('nonfinite rewards')
            robot_source=src.data.root_pos_w.detach().cpu().numpy().copy()
            if not np.allclose(robot_source,srcfixed,atol=.002):
                raise AssertionError('world-fixed source drifted')
            rp=rig.data.root_pos_w.detach().cpu().numpy().copy()
            rq=rig.data.root_quat_w.detach().cpu().numpy().copy()
            target_yaw=np.degrees(2*np.arctan2(rq[:,3],rq[:,0]))
            if any(angle_error(float(x),yaw)>1.5 for x in target_yaw):
                raise AssertionError(f'rig yaw mismatch {target_yaw} versus {yaw}')
            end_sample=round(sim_seconds*config.sample_rate)
            frames=end_sample-previous_sample
            timestamp=sample_timestamp_ns(0,end_sample,config.sample_rate)
            for i,s in enumerate(streams):
                xyzw=tuple(map(float,rq[i,[1,2,3,0]]))
                s.push_pose(Pose(timestamp,tuple(map(float,rp[i])),xyzw))
                midpoint_yaw=yaw_at_seconds((previous_sample+end_sample)/(2*config.sample_rate))
                pcm=mono.render(previous_sample,frames,config.sample_rate,rp[i],
                                yaw_xyzw(midpoint_yaw),robot_source[i],DEFAULT_ARRAY_METERS)
                t=time.perf_counter_ns()
                s.push_audio(pcm,previous_sample)
                ingestion_duration.add((time.perf_counter_ns()-t)/1e6)
                t=time.perf_counter_ns()
                features,hearing,metric=s.policy_features(timestamp)
                reading_duration.add((time.perf_counter_ns()-t)/1e6)
                if features.shape!=(5,) or not np.isfinite(features).all():
                    raise AssertionError('invalid policy hearing features')
                if not hearing.valid and np.any(features):
                    raise AssertionError('stale hearing leaked to policy')
                if hearing.valid:
                    valid_reads[i]+=1
                    if hearing.source_sample_end!=previously_seen[i]:
                        unique_valid[i]+=1
                        previously_seen[i]=hearing.source_sample_end
                        error=angle_error(hearing.bearing_world_deg,bearings[i])
                        angular_error_hist[i].add(error)
                        if error>10:accuracy_errors[i]+=1
                    age_ms=(timestamp-hearing.capture_time_ns)/1e6
                    if age_ms>=0:sim_capture_age_hist[i].add(age_ms)
            previous_sample=end_sample
            if sim_seconds//args.checkpoint_s>last_checkpoint or step==planned_steps-1:
                last_checkpoint=int(sim_seconds//args.checkpoint_s)
                metrics=[s.poll(timestamp)[1] for s in streams]
                memory=rss_bytes()
                if rss_warmup is None and sim_seconds>=min(5,args.duration_sim_s/4):rss_warmup=memory
                if memory is not None and rss_warmup is not None:
                    rss_peak=max(rss_peak or 0,memory)
                checkpoint.append({'sim_s':round(sim_seconds,3),
                                  'wall_s':round(time.monotonic()-wall_start,3),
                                  'valid_reads':valid_reads.copy(), 'unique_valid':unique_valid.copy(),
                                  'angular_p75_deg':[h.percentile(75) for h in angular_error_hist],
                                  'rss_bytes':memory, 'telemetry':metrics})
                print('ONDA NATIVE SOAK:',round(sim_seconds,1),'sim s',
                      'valid',valid_reads,'unique',unique_valid,flush=True)
        final_ts=sample_timestamp_ns(0,previous_sample,config.sample_rate)
        metrics=[s.poll(final_ts)[1] for s in streams]
        p75=[h.percentile(75) for h in angular_error_hist]
        payload.update(task='Isaac-Cartpole-v0',gpu=torch.cuda.get_device_name(0),
                       torch=torch.__version__,simulation_duration_s=round(planned_steps*env.step_dt,3),
                       wall_duration_s=round(time.monotonic()-wall_start,3),
                       simulation_steps=planned_steps,envs=2,overload_ms=args.overload_ms,valid_reads=valid_reads,
                       unique_valid_decisions=unique_valid,p75_angular_error_deg=p75,
                       errors_over_10_deg=accuracy_errors,telemetry=metrics,
                       simulator_policy_ms=policy_duration.summary(),
                       pcm_ingest_ms=ingestion_duration.summary(),
                       policy_read_ms=reading_duration.summary(),
                       simulation_capture_age_ms=[h.summary() for h in sim_capture_age_hist],
                       deadline_misses_16_67ms=policy_deadline_misses,
                       rss_start_bytes=rss_start,rss_warmup_bytes=rss_warmup,
                       rss_peak_after_warmup_bytes=rss_peak)
        failures=[]
        for i,metric in enumerate(metrics):
            if unique_valid[i]<max(5,args.duration_sim_s*0.4):failures.append(f'env{i}_insufficient_detections')
            if p75[i] is None or p75[i]>10:failures.append(f'env{i}_bearing_p75')
            if accuracy_errors[i]>max(1,unique_valid[i]*.05):failures.append(f'env{i}_bearing_outliers')
            if metric['audio_discontinuities']!=0:failures.append(f'env{i}_audio_discontinuity')
            if metric['ring_overrun_recoveries']!=0:failures.append(f'env{i}_ring_overrun')
            if metric['pose_misses']!=0:failures.append(f'env{i}_pose_miss')
            if metric['worker_error'] is not None:failures.append(f'env{i}_worker_error')
        if rss_peak is not None and rss_warmup is not None and rss_peak-rss_warmup>128*1024*1024:
            failures.append('rss_growth_after_warmup_exceeded_128MiB')
        payload.update(failures=failures,result='PASS' if not failures else 'FAIL')
        code=0 if not failures else 1
        print('ONDA NATIVE SOAK: VERDICT',payload['result'],failures,flush=True)
    except BaseException as exc:
        payload.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print('ONDA NATIVE SOAK: ERROR',payload['traceback'],flush=True)
    finally:
        if checkpoint:checkpoint.close()
        shutdown_results=[s.close(timeout_s=5.) for s in streams]
        payload['worker_clean_shutdown']=all(shutdown_results)
        if not payload['worker_clean_shutdown']:
            payload['result']='FAIL';payload.setdefault('failures',[]).append('unclean_worker_shutdown');code=1
        atomic_json(args.out,payload)
        print('ONDA NATIVE SOAK: SAVED',args.out,flush=True)
        if args.shutdown=='hard':
            print('ONDA NATIVE SOAK: FINITE HARNESS HARD EXIT (KNOWN KIT STALL)',flush=True)
            sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if app:app.close()
    return code


if __name__=='__main__':sys.exit(main())
