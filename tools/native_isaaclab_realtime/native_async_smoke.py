#!/usr/bin/env python3
"""Native Isaac Lab asynchronous integration smoke test; not an accuracy gate.

An intentional CPU reference overload may drop acoustic frames. The acceptance
criterion is that it FAILS CLOSED, leaves original policy/reward untouched,
and publishes metrics without blocking on manager.advance().
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('/ephemeral/onda-nvidia/onda-v015'))
    parser.add_argument('--steps', type=int, default=90)
    parser.add_argument('--pace-ms', type=float, default=0.0)
    parser.add_argument('--out', type=Path, default=Path('/ephemeral/onda-nvidia/onda-v015/docs/validation/native_isaaclab_realtime_v015/async_smoke.json'))
    parser.add_argument('--shutdown', choices=['hard','graceful'], default='hard')
    args = parser.parse_args()
    if args.steps < 10:
        parser.error('steps must be >=10')
    app = None
    wrapper = None
    payload = {'suite':'Onda v0.15 bounded async native Isaac Lab smoke',
               'result':'FAIL', 'notes':[
                   'Python thread decouples blocking manager.advance but not the GIL or GPU staging.',
                   'Drops/resynchronizations are expected if source throughput exceeds CPU capacity.',
                   'No real-time throughput claim; no robot policy uses hearing automatically.',
               ]}
    exit_code = 1
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True).app
        import gymnasium as gym
        import isaaclab_tasks
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
        from onda_hearing.simulators.async_isaac_lab import OndaAsyncIsaacLabEnvWrapper, AsyncHearingConfig
        sys.path.insert(0, str(args.repo/'examples'))
        from v015_isaac_lab_task_integration import attach_onda
        cfg = CartpoleEnvCfg()
        cfg.scene.num_envs = 2
        cfg.scene.env_spacing = 4.0
        cfg.episode_length_s = 30.0
        cfg.scene.sound_source = RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/OndaFixedSource',
            spawn=sim_utils.SphereCfg(
                radius=.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=.1)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5,0,1.)),
        )
        env = gym.make('Isaac-Cartpole-v0', cfg=cfg).unwrapped
        rng=np.random.default_rng(42)
        audio=(.2*rng.standard_normal(16_000 * 4)).astype(np.float32)
        synchronous = attach_onda(env, array_asset='robot',source_asset='sound_source',
                                  source_audio=audio,max_cpu_envs=2)
        wrapper = OndaAsyncIsaacLabEnvWrapper(
            env, synchronous.manager, synchronous.poses,
            config=AsyncHearingConfig(queue_capacity=2,max_staleness_ms=250.))
        obs, _ = wrapper.reset()
        assert obs['onda_hearing'].shape == (2,5)
        assert torch.isfinite(obs['onda_hearing']).all()
        initial_policy_shape=tuple(obs['policy'].shape)
        pos=env.scene['robot'].data.root_pos_w.clone()
        angle=torch.deg2rad(torch.tensor([38.,-38.],device=env.device))
        srcpos=pos.clone()
        srcpos[:,0] += 1.5*torch.cos(angle)
        srcpos[:,1] += 1.5*torch.sin(angle)
        src=env.scene['sound_source']
        src.write_root_pose_to_sim(torch.cat((srcpos,src.data.root_quat_w.clone()),dim=-1))
        # Re-seed at changed pose after reset, without ever touching manager on producer.
        wrapper.worker.submit(wrapper.poses.read(env),reset_all=True)
        policy_ms=[]
        cpu_stage_submit_ms=[]
        invalid=0
        any_valid=0
        for i in range(args.steps):
            actions=torch.zeros((2,env.action_manager.total_action_dim),device=env.device)
            start=time.perf_counter_ns()
            obs,reward,terminated,truncated,extras=wrapper.step(actions)
            policy_ms.append((time.perf_counter_ns()-start)/1e6)
            assert tuple(obs['policy'].shape) == initial_policy_shape
            hearing=obs['onda_hearing']
            assert hearing.shape==(2,5) and hearing.dtype==torch.float32
            assert torch.isfinite(hearing).all()
            any_valid+=int((hearing[:,3]>0.5).sum().item())
            if wrapper.last_hearing_metadata['stale']:
                assert not hearing.any(), 'Stale acoustic result leaked to policy'
                invalid+=1
            if args.pace_ms > 0:
                time.sleep(args.pace_ms / 1000.0)
            if i in (0,9,29,59,args.steps-1):
                print('ONDA ASYNC: step',i+1,wrapper.last_hearing_metadata,flush=True)
        # Let the current worker item finish so the test can inspect published state.
        time.sleep(.25)
        _, telem=wrapper.worker.read()
        assert telem['processed_frames']>=1, 'Worker never processed an acoustic batch'
        assert telem['worker_errors']==0, 'Worker exception'
        if args.pace_ms > 0:
            assert any_valid > 0, 'Paced worker never produced valid hearing'
            assert telem['dropped_frames'] == 0, 'Paced worker still overloaded'
        payload.update(diagnostic_pace_ms=args.pace_ms)
        payload.update(result='PASS', steps=args.steps, policy_step_ms_p50=float(np.percentile(policy_ms,50)),
                       policy_step_ms_p95=float(np.percentile(policy_ms,95)),
                       acoustic_observations_valid_count=any_valid,
                       stale_read_steps=invalid, telemetry=telem,
                       gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                       warning='PASS checks isolation/fail-closed behavior, not 60 Hz acoustic throughput')
        exit_code=0
        print('ONDA ASYNC: NATIVE SMOKE PASS',flush=True)
    except BaseException as exc:
        payload.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print('ONDA ASYNC: FAIL',payload['traceback'],flush=True)
    finally:
        if wrapper is not None:
            try:
                payload['worker_clean_shutdown']=wrapper.worker.close(timeout_s=5)
            except BaseException as exc:
                payload['worker_shutdown_error']=repr(exc)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.out.with_suffix('.json.tmp')
        with tmp.open('w') as fp:
            json.dump(payload,fp,indent=2,allow_nan=False)
            fp.flush();os.fsync(fp.fileno())
        os.replace(tmp,args.out)
        print('ONDA ASYNC: SAVED',args.out,flush=True)
        if args.shutdown=='hard':
            print('ONDA ASYNC: FINITE TEST HARNESS HARD EXIT; NOT FOR PRODUCTION',flush=True)
            sys.stdout.flush();sys.stderr.flush()
            os._exit(exit_code)
        if app is not None:
            app.close()
    return exit_code

if __name__ == '__main__':
    sys.exit(main())
