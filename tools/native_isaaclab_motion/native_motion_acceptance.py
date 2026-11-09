#!/usr/bin/env python3
"""Native Onda v0.15 / Isaac Lab 2.3.1 rotating-array acceptance.

This is a finite-run integration test, NOT production runtime software.
A kinematic, non-colliding rigid mic rig rotates in Isaac Lab while a distinct
emitter rigid body remains fixed in WORLD coordinates. Onda receives poses via
its existing IsaacLabSceneReader, renders PCM and infers bearing acoustically.
Ground truth is independently computed ONLY for test assertions/reporting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from rotation_math import (angular_error_deg, bearing_from_features,
                           local_bearing_deg, normalize_degrees, yaw_quaternion_wxyz)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def save_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(document, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_pcm():
    """Deterministic, 300-3800Hz broadband source at 16 kHz; no direction labels."""
    rng = np.random.default_rng(20261109)
    values = rng.standard_normal(16_000 * 8).astype(np.float32)
    spec = np.fft.rfft(values)
    hz = np.fft.rfftfreq(values.size, 1/16_000)
    spec[(hz < 300) | (hz > 3800)] = 0.0
    values = np.fft.irfft(spec, n=values.size).astype(np.float32)
    values *= 0.18 / max(1.0, float(np.std(values)) * 3)
    return values


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('/ephemeral/onda-nvidia/onda-v015'))
    p.add_argument('--out', type=Path, default=Path('/ephemeral/onda-nvidia/onda-v015/docs/validation/native_isaaclab_motion_v015'))
    p.add_argument('--hold-steps', type=int, default=90)
    p.add_argument('--ramp-steps', type=int, default=25)
    p.add_argument('--tail-steps', type=int, default=35)
    p.add_argument('--angle-tolerance', type=float, default=10.0)
    p.add_argument('--shutdown', choices=('hard', 'graceful'), default='hard')
    return p.parse_args()


def main():
    args = args_parse()
    ensure(args.hold_steps >= 65, 'hold-steps must be >=65 for PCM convergence')
    ensure(args.ramp_steps >= 10, 'ramp-steps must be >=10 for real incremental motion')
    ensure(15 <= args.tail_steps <= args.hold_steps - 20, 'invalid tail-steps')
    ensure(0 < args.angle_tolerance <= 20, 'invalid angle tolerance')
    ensure((args.repo/'examples/v015_isaac_lab_task_integration.py').is_file(), 'Onda SDK example not found')
    output = args.out / 'results.json'
    report = {
        'suite': 'Onda v0.15 native Isaac Lab rotating microphone rig acceptance',
        'started_utc': utc_now(), 'result': 'INCOMPLETE', 'test_revision': '2-world-and-local-frames',
        'environment': {}, 'phases': [], 'timing': {},
        'limitations': [
            'Virtual four-microphone rig; geometry not calibrated against physical robot.',
            'The rigid rig follows commanded kinematic poses, not a driven robot joint.',
            'Onda CPU acoustic renderer and SRP; GPU-parallel throughput not demonstrated.',
            'A test-runner hard exit bypasses the known Kit close() hang when requested.',
            'Phase changes use gradual yaw ramps followed by stable evaluation windows.',
        ]
    }
    exit_code = 1
    app = None
    env = None
    print('ONDA MOTION: STARTING NATIVE ISAAC LAB', flush=True)
    try:
        # Required: launch Kit BEFORE importing native simulation/task modules.
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True).app
        import gymnasium as gym
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # noqa: F401 - registers Isaac-Cartpole-v0
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
        sys.path.insert(0, str(args.repo/'examples'))
        from v015_isaac_lab_task_integration import attach_onda

        cfg = CartpoleEnvCfg()
        cfg.scene.num_envs = 2
        cfg.scene.env_spacing = 4.0
        cfg.episode_length_s = 30.0
        cfg.scene.mic_rig = RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/OndaMicRig',
            spawn=sim_utils.SphereCfg(
                radius=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 2)),
        )
        cfg.scene.sound_source = RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/OndaFixedEmitter',
            spawn=sim_utils.SphereCfg(
                radius=0.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0, 2)),
        )
        env = gym.make('Isaac-Cartpole-v0', cfg=cfg).unwrapped
        ensure(env.num_envs == 2 and str(env.device).startswith('cuda'), 'Expected two CUDA Isaac Lab environments')
        report['environment'] = {
            'task': 'Isaac-Cartpole-v0', 'num_envs': 2,
            'device': str(env.device), 'step_dt_s': float(env.step_dt),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
        }
        robot = env.scene['robot']
        rig = env.scene['mic_rig']
        emitter = env.scene['sound_source']
        wrapped = attach_onda(env, array_asset='mic_rig', source_asset='sound_source',
                              source_audio=source_pcm(), max_cpu_envs=2)
        initial_obs, _ = wrapped.reset()
        ensure('policy' in initial_obs and initial_obs['onda_hearing'].shape == (2,5),
               'Policy/hearing contract missing on reset')
        original_policy_shape = tuple(initial_obs['policy'].shape)

        # Use native environment origins, then keep emitter poses WORLD-FIXED.
        origins = robot.data.root_pos_w.detach().clone()
        targets_deg = np.array([38.0, -38.0], dtype=np.float64)
        angles = torch.tensor(targets_deg, device=env.device, dtype=torch.float32)
        emit_pos = origins.clone()
        emit_pos[:, 0] += 1.5 * torch.cos(torch.deg2rad(angles))
        emit_pos[:, 1] += 1.5 * torch.sin(torch.deg2rad(angles))
        initial_emitter_quat = emitter.data.root_quat_w.detach().clone()
        emitter.write_root_pose_to_sim(torch.cat([emit_pos, initial_emitter_quat], dim=-1))
        fixed_emit_pos = emit_pos.detach().cpu().numpy().copy()
        report['environment']['fixed_emitters_world_xyz'] = fixed_emit_pos.tolist()
        report['environment']['mic_rig_world_xyz'] = origins.detach().cpu().tolist()
        report['environment']['target_world_angles_deg'] = targets_deg.tolist()

        def write_rig_yaw(degrees):
            q = yaw_quaternion_wxyz(degrees)
            qt = torch.tensor(q, dtype=origins.dtype, device=env.device)
            qs = qt.unsqueeze(0).repeat(2, 1)
            rig.write_root_pose_to_sim(torch.cat((origins, qs), dim=-1))

        write_rig_yaw(0.0)
        wrapped.manager.reset()
        wrapped.manager.seed(*wrapped.poses.read(env))
        ensure(np.allclose(wrapped.manager.read(), 0), 'Seed unexpectedly produced observations')
        print('ONDA MOTION: WORLD-FIXED EMITTERS', fixed_emit_pos.tolist(), flush=True)

        # Independent Onda-only timing of manager.advance, without SDK patching.
        original_advance = wrapped.manager.advance
        acoustic_ms = []
        def timed_advance(*a, **kw):
            start = time.perf_counter_ns()
            result = original_advance(*a, **kw)
            acoustic_ms.append((time.perf_counter_ns()-start)/1e6)
            return result
        wrapped.manager.advance = timed_advance

        def action_zero():
            return torch.zeros((2, env.action_manager.total_action_dim), device=env.device, dtype=torch.float32)

        policy_ms = []
        yaw_schedule = (0.0, 30.0, -30.0, 0.0)
        previous_yaw = 0.0
        for phase_id, target_yaw in enumerate(yaw_schedule):
            # Physically rotate the native rigid body over many steps.
            # During the ramp, acoustic history remains alive: no manual resets.
            if phase_id:
                for k in range(1, args.ramp_steps+1):
                    yaw = previous_yaw + (target_yaw-previous_yaw)*k/args.ramp_steps
                    write_rig_yaw(yaw)
                    _, _, term, trunc, _ = wrapped.step(action_zero())
                    ensure(not bool((term|trunc).any().item()),
                           f'Native environment reset during yaw ramp phase {phase_id}')
            else:
                write_rig_yaw(target_yaw)
            previous_yaw = target_yaw
            print(f'ONDA MOTION: PHASE {phase_id} TARGET YAW {target_yaw:+.1f} deg', flush=True)
            data = [[], []]
            observed_yaw = []
            fresh_audio_decisions = [0, 0]
            for step in range(args.hold_steps):
                t0 = time.perf_counter_ns()
                obs, reward, term, trunc, extra = wrapped.step(action_zero())
                policy_ms.append((time.perf_counter_ns()-t0)/1e6)
                ensure(not bool((term|trunc).any().item()),
                       f'Native environment reset in phase {phase_id}, step {step+1}')
                feat = obs['onda_hearing']
                ensure(feat.shape == (2,5) and feat.dtype == torch.float32 and
                       feat.device.type == 'cuda' and bool(torch.isfinite(feat).all()),
                       f'Hearing contract violation, phase {phase_id} step {step+1}')
                ensure(tuple(obs['policy'].shape) == original_policy_shape,
                       'Native policy observation tensor shape changed')
                if step >= args.hold_steps-args.tail_steps:
                    # Prove the pipeline made NEW acoustic decisions in this
                    # phase, rather than merely replaying a cached bearing.
                    for env_id, frame in enumerate(wrapped.manager.frames()):
                        if frame is not None and frame.new_decision:
                            fresh_audio_decisions[env_id] += 1
                    rig_pos, rig_q, scene_sources = wrapped.poses.read(env)
                    rig_pos = rig_pos.detach().cpu().numpy()
                    rig_q = rig_q.detach().cpu().numpy()
                    src_pos = scene_sources['sound'].detach().cpu().numpy()
                    drift = float(np.max(np.linalg.norm(src_pos-fixed_emit_pos, axis=1)))
                    ensure(drift <= 0.001, f'Stationary emitter moved {drift:.6f} m')
                    features = feat.detach().cpu().numpy()
                    for i in range(2):
                        actual_q = rig_q[i]
                        # Compare actual quaternion orientation; q and -q are equivalent.
                        planned_q = yaw_quaternion_wxyz(target_yaw)
                        dot = float(abs(np.dot(actual_q, planned_q)))
                        dot = min(1.0, max(-1.0, dot))
                        orientation_err = math.degrees(2*math.acos(dot))
                        ensure(orientation_err <= 1.5,
                               f'Native mic_rig pose failed to rotate: env{i}, err={orientation_err:.2f}deg')
                        observed_yaw.append(orientation_err)
                        # v0.15 policy features encode WORLD azimuth (not mic-local).
                        # Compare WORLD-to-WORLD first. Derive local bearing only for
                        # evaluation by rotating the *estimated world ray* using the
                        # actual Isaac Lab microphone quaternion. Never feed this
                        # derived answer back into Onda or the policy.
                        delta_w = src_pos[i] - rig_pos[i]
                        reference_world = normalize_degrees(
                            math.degrees(math.atan2(delta_w[1], delta_w[0])))
                        reference_local = local_bearing_deg(
                            rig_pos[i], actual_q, src_pos[i])
                        if features[i,3] == 1.0:
                            estimate_world = bearing_from_features(features[i])
                            estimate_world_ray = rig_pos[i] + np.array([
                                math.cos(math.radians(estimate_world)),
                                math.sin(math.radians(estimate_world)), 0.0])
                            estimate_local = local_bearing_deg(
                                rig_pos[i], actual_q, estimate_world_ray)
                            data[i].append({'step':step+1,
                                'reference_deg':reference_local,
                                'estimate_deg':estimate_local,
                                'error_deg':angular_error_deg(estimate_local,reference_local),
                                'world_reference_deg':reference_world,
                                'world_estimate_deg':estimate_world,
                                'world_error_deg':angular_error_deg(estimate_world,reference_world),
                                'evidence':float(features[i,2]),
                                'age_s':float(features[i,4])})
            phase_result = {'phase_id':phase_id, 'target_yaw_deg':target_yaw,
                            'hold_steps':args.hold_steps, 'ramp_steps':0 if phase_id==0 else args.ramp_steps,
                            'environments':[], 'max_rig_orientation_error_deg':max(observed_yaw)}
            for i in range(2):
                samples = data[i]
                ensure(fresh_audio_decisions[i] >= 2,
                       f'Phase {phase_id} env{i}: not enough fresh acoustic decisions '
                       f'({fresh_audio_decisions[i]})')
                ensure(len(samples) >= max(12,args.tail_steps//3),
                       f'Phase {phase_id} env{i}: only {len(samples)} valid detections in final {args.tail_steps} steps')
                errors = [x['error_deg'] for x in samples]
                world_errors = [x['world_error_deg'] for x in samples]
                p75 = float(np.percentile(errors,75))
                median = float(np.median(errors))
                world_p75 = float(np.percentile(world_errors,75))
                ensure(p75 <= args.angle_tolerance,
                       f'Phase {phase_id} env{i}: P75 local-frame angular error {p75:.2f}deg exceeds {args.angle_tolerance}deg')
                ensure(world_p75 <= args.angle_tolerance,
                       f'Phase {phase_id} env{i}: P75 world-frame angular error {world_p75:.2f}deg exceeds {args.angle_tolerance}deg')
                phase_result['environments'].append({'env_id':i, 'valid_tail_samples':len(samples),
                    'expected_local_bearing_last_deg':samples[-1]['reference_deg'],
                    'inferred_local_bearing_last_deg':samples[-1]['estimate_deg'],
                    'expected_world_bearing_last_deg':samples[-1]['world_reference_deg'],
                    'inferred_world_bearing_last_deg':samples[-1]['world_estimate_deg'],
                    'median_error_deg':median, 'p75_error_deg':p75,
                    'world_p75_error_deg':world_p75,
                    'fresh_acoustic_decisions_in_tail':fresh_audio_decisions[i],
                    'mean_evidence':float(np.mean([x['evidence'] for x in samples]))})
            report['phases'].append(phase_result)
            # Persist each PASS phase, so a later failure doesn't erase evidence.
            save_json_atomic(output,report)
            print('ONDA MOTION: PHASE PASS', phase_id, phase_result['environments'], flush=True)

        # Prevent a degenerate fixed-bearing implementation from passing.
        for i in range(2):
            first = report['phases'][0]['environments'][i]['inferred_local_bearing_last_deg']
            rotated = report['phases'][1]['environments'][i]['inferred_local_bearing_last_deg']
            motion_change = angular_error_deg(first,rotated)
            ensure(motion_change >= 15,
                   f'Env{i}: apparent hearing direction barely changed after 30deg array rotation ({motion_change:.1f}deg)')
            restored = report['phases'][-1]['environments'][i]['inferred_local_bearing_last_deg']
            ensure(angular_error_deg(first,restored) <= args.angle_tolerance,
                   f'Env{i}: bearing did not return to baseline after returning to yaw=0')

        report['timing'] = {
            'policy_step_including_physx_onda_python_ms_p50':float(np.median(policy_ms)),
            'policy_step_including_physx_onda_python_ms_p95':float(np.percentile(policy_ms,95)),
            'onda_manager_advance_including_cpu_staging_ms_p50':float(np.median(acoustic_ms)),
            'onda_manager_advance_including_cpu_staging_ms_p95':float(np.percentile(acoustic_ms,95)),
            'number_of_timed_policy_steps':len(policy_ms),
            'number_of_timed_onda_advances':len(acoustic_ms),
            'notes':('Manager.advance includes PCM generation, SRP, and GPU-to-CPU staging; '
                     'it does not include scene reading, policy tensor conversion, or Kit startup. '
                     'No GPU synchronization is imposed; these are wall-clock CPU-side call durations.')
        }
        report['result']='PASS'
        print('ONDA MOTION: ALL PHASES PASSED', flush=True)
        print('ONDA MOTION: TIMINGS', report['timing'], flush=True)
        exit_code=0
    except BaseException as exc:
        report['result']='FAIL'
        report['failure']={'error_type':type(exc).__name__, 'message':str(exc),
                           'traceback':traceback.format_exc()}
        print('ONDA MOTION: FAIL\n'+traceback.format_exc(), flush=True)
    finally:
        report['completed_utc']=utc_now()
        report['shutdown_mode']=args.shutdown
        try:
            save_json_atomic(output,report)
            print('ONDA MOTION: SAVED',output,flush=True)
        except BaseException:
            traceback.print_exc()
            exit_code=1
        if args.shutdown=='hard':
            # Only in this single-purpose subprocess; never in training/SDK runtime.
            print('ONDA MOTION: TEST-HARNESS HARD EXIT (KNOWN KIT SHUTDOWN STALL)',flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(exit_code)
        try:
            if env is not None: env.close()
            if app is not None: app.close()
        except BaseException:
            traceback.print_exc()
            exit_code=1
    return exit_code


if __name__=='__main__':
    sys.exit(main())
