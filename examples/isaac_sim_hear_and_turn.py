"""Onda v0.14 Isaac Sim live articulated-head hearing experiment.

Requires an EXISTING USD robot with an initialized revolute head-yaw joint,
+Z positive yaw, an array attachment prim, and a stationary source prim.
The renderer uses the source prim position; the controller never sees it.

Run under Isaac Sim 5.1's own Python, not system Python:
 ./python.sh /path/to/onda-robot-hearing/examples/isaac_sim_hear_and_turn.py \
   --usd /path/to/robot.usd --robot-prim /World/Robot \
   --array-prim /World/Robot/Head/MicrophoneArray \
   --head-joint head_yaw --source-prim /World/Beacon --strict

The stage must include the Beacon prim and a functional head position drive.
This is a real-physics validation harness, not a prevalidated universal robot asset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import wave

import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.control import wrap_degrees
from onda_hearing.experiments import IsaacHearingLoop, HearingLoopConfig
from onda_hearing.simulators import IsaacSimSceneAdapter, IsaacSingleArticulationYaw


def excitation(seconds: float, fs: int = 16000) -> np.ndarray:
    n = int(seconds * fs)
    rng = np.random.default_rng(202613)
    samples = rng.normal(size=n)
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(n, 1 / fs)
    spectrum[(frequencies < 300) | (frequencies > 3800)] = 0
    signal = np.fft.irfft(spectrum, n=n)
    return (0.085 * signal / max(1.0, float(np.std(signal)) * 3)).astype(np.float32)


def write_wav(path: Path, pcm: np.ndarray, fs: int) -> None:
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(pcm.shape[1])
        out.setsampwidth(2)
        out.setframerate(fs)
        out.writeframes(np.round(np.clip(pcm, -1.0, 1.0) * 32767).astype('<i2').tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--usd', type=Path, required=True)
    ap.add_argument('--robot-prim', required=True)
    ap.add_argument('--array-prim', required=True)
    ap.add_argument('--head-joint', required=True)
    ap.add_argument('--source-prim', required=True)
    ap.add_argument('--duration', type=float, default=2.4)
    ap.add_argument('--mode', choices=['sound', 'silence'], default='sound')
    ap.add_argument('--output-dir', type=Path, default=Path('benchmark_results/v014_isaac_hear_and_turn'))
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()
    if not args.usd.is_file():
        ap.error(f'USD file missing: {args.usd}')
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 15:
        ap.error('duration must be finite between 1 and 15 seconds')

    # Isaac APIs must be imported after SimulationApp initialization.
    from isaacsim import SimulationApp
    # Onda arguments have already been parsed. Do not forward them to Kit.
    import sys
    sys.argv = [sys.argv[0]]

    print("ONDA: Initializing Isaac Sim", flush=True)
    app = SimulationApp({
        'headless': args.headless,
        'fast_shutdown': True,
    })
    print("ONDA: Isaac Sim initialized", flush=True)
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from onda_hearing.runtime import ScheduledSource

        world = World(stage_units_in_meters=1.0)
        dt = 0.002
        world.set_simulation_dt(physics_dt=dt, rendering_dt=dt)
        add_reference_to_stage(usd_path=str(args.usd.resolve()), prim_path='/World')
        robot = SingleArticulation(prim_path=args.robot_prim, name='onda_isaac_robot')
        world.scene.add(robot)
        world.reset()
        robot.initialize()

        stage = get_current_stage()
        pose_provider = IsaacSimSceneAdapter(stage, array_prim=args.array_prim,
                                               source_prims={'beacon': args.source_prim})

        print("ONDA DIAGNOSTIC: articulation path =", args.robot_prim, flush=True)
        print("ONDA DIAGNOSTIC: DOF names =", robot.dof_names, flush=True)
        print("ONDA DIAGNOSTIC: DOF count =", robot.num_dof, flush=True)

        from pxr import UsdPhysics
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                print("ONDA DIAGNOSTIC: USD joint =", prim.GetPath(), flush=True)
                print("  body0 =", joint.GetBody0Rel().GetTargets(), flush=True)
                print("  body1 =", joint.GetBody1Rel().GetTargets(), flush=True)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                print("ONDA DIAGNOSTIC: articulation root =", prim.GetPath(), flush=True)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print("ONDA DIAGNOSTIC: rigid body =", prim.GetPath(), flush=True)

        if args.head_joint not in robot.dof_names:
            raise RuntimeError(
                f"Joint {args.head_joint!r} not registered. "
                f"Available DOFs: {robot.dof_names}"
            )

        actuator = IsaacSingleArticulationYaw(robot, joint_name=args.head_joint)
        array = MicrophoneArray.from_positions(
            [[0.05, 0.03, 0.0], [-0.045, 0.035, 0.0],
             [-0.05, -0.038, 0.0], [0.042, -0.047, 0.0]],
            array_id='head_array', frame_id='robot/head', geometry_id='asym90mm_v014')
        waveform = excitation(args.duration + 0.5)
        if args.mode == 'silence':
            waveform[:] = 0.0
        loop = IsaacHearingLoop(
            array=array, sources=(ScheduledSource('beacon', waveform, 0),),
            poses=pose_provider, actuator=actuator,
            config=HearingLoopConfig(physics_hz=500, audio_rate_hz=16000))
        loop.reset()
        observations = []
        eval_rows = []
        initial_yaw = actuator.get_joint_yaw_rad()
        steps = round(args.duration / dt)
        for k in range(1, steps + 1):
            loop.before_step(k)
            world.step(render=not args.headless)
            row = loop.after_step(k)
            if row is None:
                continue
            observations.append(row)
            # EVALUATION ONLY: this snapshot and its source position never reach
            # IsaacHearingLoop, selector or actuator except via audio rendering.
            if args.mode == 'sound':
                truth = pose_provider.snapshot(row.time_ns)
                delta = (np.asarray(truth.source_positions_m['beacon'])
                         - np.asarray(truth.array_position_m))
                true_world_deg = math.degrees(math.atan2(delta[1], delta[0]))
                err = wrap_degrees(math.degrees(row.joint_yaw_rad) - true_world_deg)
            else:
                true_world_deg = None
                err = None
            eval_rows.append({'time_s': row.time_ns / 1e9,
                              'true_world_deg': true_world_deg,
                              'alignment_error_deg': err})

        pcm = loop.microphone_pcm()
        expected_samples = round(args.duration * 16000)
        if pcm.shape != (expected_samples, array.channels):
            raise RuntimeError(f'PCM duration mismatch: {pcm.shape} vs {expected_samples}')
        if not observations:
            raise RuntimeError('No observations returned')
        final_yaw = math.degrees(observations[-1].joint_yaw_rad)
        max_movement = max(abs(math.degrees(x.joint_yaw_rad - initial_yaw)) for x in observations)
        final_error = None if args.mode == 'silence' else abs(eval_rows[-1]['alignment_error_deg'])
        passes = ((args.mode == 'sound' and loop.accepted_decisions > 0 and final_error <= 10)
                  or (args.mode == 'silence' and loop.accepted_decisions == 0 and max_movement <= 2))
        summary = {
            'status': 'PASS' if passes else 'FAIL', 'simulator': 'Isaac Sim',
            'sdk_version': '0.14.0a0', 'mode': args.mode,
            'final_head_yaw_deg': final_yaw,
            'final_alignment_error_deg': final_error,
            'max_head_movement_deg': max_movement,
            'perception_decision_count': loop.new_decisions,
            'accepted_decision_count': loop.accepted_decisions,
            'audio_samples': len(pcm), 'audio_channels': array.channels,
            'source_truth_exposed_to_controller': False,
            'note': 'Single-source direct-path demo. No native Isaac results claimed until run.'
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / 'telemetry.csv').open('w', newline='') as f:
            columns = [*asdict(observations[0]), 'time_s', 'true_world_deg', 'alignment_error_deg']
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            for observation, truth in zip(observations, eval_rows):
                writer.writerow({**asdict(observation), **truth})
        write_wav(args.output_dir / 'microphones.wav', pcm, 16000)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        if args.strict and not passes:
            raise SystemExit('Isaac hearing-to-motion acceptance gate failed')
    finally:
        app.close()


if __name__ == '__main__':
    main()
