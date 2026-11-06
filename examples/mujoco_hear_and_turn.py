"""Actual MuJoCo head yaw driven ONLY by Onda-estimated microphone bearings.

Install: python -m pip install -e '.[mujoco,dev]'
Run:     python examples/mujoco_hear_and_turn.py --strict
Negative control: python examples/mujoco_hear_and_turn.py --mode silence --strict

This is a research demo: no source IDs, no semantics, no guaranteed accuracy.
The ground truth is calculated ONLY in the evaluation/logger path after control.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import wave

import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.runtime import HearingRuntime, HearingWindowBuffer, ScheduledSource
from onda_hearing.simulators import MuJoCoSceneAdapter
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig
from onda_hearing.control import AcousticBearing, HeadYawConfig, HeadYawController, wrap_degrees
from onda_hearing.experiments import (
    AcousticOnlySelector, AcousticSelectionConfig, EvaluationRow,
    evaluate_episode, save_episode,
)

FS = 16000
PHYSICS_DT_S = 0.002
SCENE_DT_STEPS = 10
PERCEPTION_PERIOD_STEPS = 50  # 100 ms, audio window is 256 ms
SOUND_AZIMUTH_DEG = 38.0

XML = '''<mujoco model="onda_acoustic_control">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="head" pos="0 0 1.0">
      <joint name="head_yaw" type="hinge" axis="0 0 1"
             range="-1.570796 1.570796" limited="true" damping="3" armature="0.015"/>
      <geom type="sphere" size="0.08" mass="1" rgba="0.18 0.58 0.75 1"/>
      <site name="head_array" pos="0.0 0.0 0.0" size="0.01"/>
    </body>
    <site name="beacon" pos="1.182 0.924 1.0" size="0.018"/>
  </worldbody>
  <actuator>
    <position name="head_servo" joint="head_yaw" kp="70"/>
  </actuator>
</mujoco>'''


def excitation(seconds: float, fs: int = FS) -> np.ndarray:
    """Seeded broadband beacon, rather than narrowband tone unsuitable for TDOA."""
    n = int(seconds * fs)
    rng = np.random.default_rng(202613)
    samples = rng.normal(0, 1, size=n)
    spec = np.fft.rfft(samples)
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 300) | (f > 3800)] = 0
    signal = np.fft.irfft(spec, n=n)
    return (0.085 * signal / max(1.0, float(np.std(signal)) * 3)).astype(np.float32)


def write_pcm(path: Path, samples: np.ndarray, fs: int) -> None:
    pcm = np.asarray(samples, dtype=np.float32)
    with wave.open(str(path), 'wb') as file:
        file.setnchannels(pcm.shape[1])
        file.setsampwidth(2)
        file.setframerate(fs)
        file.writeframes(np.round(np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())


def run_episode(*, duration_s: float = 2.4, mode: str = 'sound',
                source_azimuth_deg: float = 38.0, output_dir: str | Path = 'benchmark_results/v013_hear_and_turn',
                strict: bool = False) -> dict:
    """Live MuJoCo integration; raises ImportError when MuJoCo is not installed."""
    import mujoco

    if mode not in ('sound', 'silence') or not math.isfinite(duration_s) or duration_s < 1.0 or duration_s > 15 or not math.isfinite(source_azimuth_deg) or not -75 <= source_azimuth_deg <= 75:
        raise ValueError('invalid mode/duration')
    bearing_rad = math.radians(source_azimuth_deg)
    xyz = (1.5 * math.cos(bearing_rad), 1.5 * math.sin(bearing_rad))
    scene_xml = XML.replace('pos="1.182 0.924 1.0"', f'pos="{xyz[0]:.10f} {xyz[1]:.10f} 1.0"')
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MuJoCoSceneAdapter(model, data, array_site='head_array',
                                 source_sites={'beacon': 'beacon'})
    array = MicrophoneArray.from_positions(
        [[0.05,0.03,0],[-0.045,0.035,0],[-0.05,-0.038,0],[0.042,-0.047,0]],
        array_id='head_array', frame_id='robot/head', geometry_id='asym90mm_v013')
    audio = excitation(duration_s + 0.5)
    if mode == 'silence':
        audio = np.zeros_like(audio)
    runtime = HearingRuntime(array, (ScheduledSource('beacon', audio, 0),), sample_rate_hz=FS)
    buffer = HearingWindowBuffer(array, sample_rate_hz=FS,
                                 window_samples=4096, capacity_samples=8192)
    localizer = MotionAwareSRPLocalizer(MotionSRPConfig(
        block_samples=320, hop_samples=160, angular_resolution_deg=2,
        max_candidates=3, min_relative_score=.65, min_prominence=.012))
    selector = AcousticOnlySelector(localizer, AcousticSelectionConfig(
        min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
    controller = HeadYawController(HeadYawConfig(max_setpoint_speed_deg_s=125,
                                                min_evidence_score=.08,
                                                max_observation_age_s=.3))
    controller.reset(initial_joint_rad=float(data.qpos[0]))
    initial = adapter.snapshot(0)
    runtime.advance(initial)
    buffer.push_pose(initial)
    rows: list[EvaluationRow] = []
    audio_segments = []
    last_decision = None
    last_bearing: AcousticBearing | None = None
    last_reason = 'not_yet_observed'
    last_command = None
    steps = round(duration_s / PHYSICS_DT_S)
    for step in range(1, steps + 1):
        # The controller can access its own yaw and the current joint state.
        # The base of this deliberately minimal model has zero world yaw.
        now_ns = round(step * PHYSICS_DT_S * 1e9)
        command = controller.command(now_ns=now_ns, base_world_yaw_rad=0.,
                                     dt_s=PHYSICS_DT_S)
        last_command = command
        data.ctrl[0] = command.target_joint_rad
        mujoco.mj_step(model, data)
        if step % SCENE_DT_STEPS != 0:
            continue
        current_ns = round(float(data.time) * 1e9)
        snap = adapter.snapshot(current_ns)
        packet = runtime.advance(snap)
        buffer.push_pose(snap)
        if packet is not None:
            buffer.push_audio(packet.audio)
            audio_segments.append(packet.audio.pcm)
        if step % PERCEPTION_PERIOD_STEPS == 0 and buffer.ready:
            last_decision = selector.process(buffer.window(), array)
            last_reason = last_decision.reason
            if last_decision.bearing is not None:
                last_bearing = last_decision.bearing
                controller.set_bearing(last_bearing)
            else:
                # Ambiguous / unsupported detection: no further movement.
                controller.set_bearing(None)
                last_bearing = None
        measured_head_deg = math.degrees(float(data.qpos[0]))
        # Evaluation-only section. Never pass `snap.source_positions_m` to
        # localizer, selector, head controller, or the motor command.
        if mode == 'sound':
            delta = (np.array(snap.source_positions_m['beacon'])
                     - np.array(snap.array_position_m))
            true_bearing_deg = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            alignment_error = wrap_degrees(measured_head_deg - true_bearing_deg)
        else:
            true_bearing_deg = None
            alignment_error = None
        rows.append(EvaluationRow(
            time_s=float(data.time), head_yaw_deg=measured_head_deg,
            command_yaw_deg=math.degrees(command.target_joint_rad),
            acoustic_bearing_world_deg=None if last_bearing is None else last_bearing.world_azimuth_deg,
            acoustic_score=None if last_bearing is None else last_bearing.evidence_score,
            control_reason=command.reason, acoustic_reason=last_reason,
            ground_truth_world_deg=true_bearing_deg,
            ground_truth_alignment_error_deg=alignment_error,
        ))
    pcm = np.concatenate(audio_segments, axis=0)
    if pcm.shape != (round(duration_s * FS), array.channels):
        raise RuntimeError(f'audio sample continuity failed: {pcm.shape}')
    summary = evaluate_episode(rows)
    summary.update({
        'sdk_version': '0.14.0a0', 'simulator': f'mujoco {mujoco.__version__}',
        'sample_rate_hz': FS, 'microphone_channels': array.channels,
        'audio_frames': len(pcm), 'audio_chunks': len(audio_segments),
        'experiment_duration_s': duration_s,
        'evaluation_only_source_azimuth_deg': source_azimuth_deg if mode == 'sound' else None,
        'source_truth_exposed_to_controller': False,
        'perception_backend': 'motion_srp_v08',
        'notes': ['single stationary source only',
                  'direct-path audio and deterministic broadband excitation',
                  'no multisource tracking, semantics, or robot hardware qualification'],
    })
    out = Path(output_dir)
    save_episode(out, rows, summary)
    write_pcm(out / 'microphones.wav', pcm, FS)
    if strict and summary['status'] != 'PASS':
        raise RuntimeError(f"auditory control acceptance gate failed: {json.dumps(summary)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sound','silence'], default='sound')
    parser.add_argument('--duration', type=float, default=2.4)
    parser.add_argument('--source-bearing', type=float, default=38.0, help='evaluation scene source azimuth, degrees')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--strict', action='store_true', help='fail on acceptance-gate miss')
    args = parser.parse_args()
    output_dir = args.output_dir or f'benchmark_results/v013_hear_and_turn/{args.mode}'
    print(json.dumps(run_episode(duration_s=args.duration, mode=args.mode,
                                 source_azimuth_deg=args.source_bearing, output_dir=output_dir, strict=args.strict), indent=2))


if __name__ == '__main__':
    main()
