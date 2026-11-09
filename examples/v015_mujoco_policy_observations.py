"""Native MuJoCo -> simulated four-microphone PCM -> Onda spatial policy features.

This example does NOT command the robot. It demonstrates observation delivery
for arbitrary customer-defined policies. Simulator source truth is used for
acoustic rendering and optional final evaluation ONLY.

Run:
  python examples/v015_mujoco_policy_observations.py --mode sound --source-bearing 38 --strict
  python examples/v015_mujoco_policy_observations.py --mode silence --strict
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig
from onda_hearing.policy import WindowedHearingPipeline, WindowedPipelineConfig
from onda_hearing.runtime import HearingRuntime, ScheduledSource
from onda_hearing.simulators import MuJoCoSceneAdapter
from onda_hearing.simulators.mujoco_policy import WindowedMuJoCoPolicyBridge
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig


FS = 16000
DT = 0.002
XML = '''<mujoco model="onda_policy_observation">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="head" pos="0 0 1.0">
      <joint name="head_yaw" type="hinge" axis="0 0 1"
             range="-1.570796 1.570796" limited="true" damping="3" armature="0.015"/>
      <geom type="sphere" size="0.08" mass="1"/>
      <site name="head_array" pos="0 0 0" size="0.01"/>
    </body>
    <site name="beacon" pos="1.182 0.924 1.0" size="0.018"/>
  </worldbody>
</mujoco>'''


def excitation(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(202613)
    n = round(seconds * FS)
    signal = rng.normal(size=n)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1. / FS)
    spectrum[(freqs < 300) | (freqs > 3800)] = 0
    signal = np.fft.irfft(spectrum, n=n)
    return (0.085 * signal / max(1., float(np.std(signal)) * 3)).astype(np.float32)


def run_episode(*, mode: str = 'sound', bearing_deg: float = 38.,
                duration_s: float = 2.4, strict: bool = False) -> dict:
    import mujoco
    if mode not in ('sound', 'silence'):
        raise ValueError('invalid mode')
    if not math.isfinite(bearing_deg) or abs(bearing_deg) > 75:
        raise ValueError('bearing must be within ±75°')
    if not 1. <= duration_s <= 15. or not math.isfinite(duration_s):
        raise ValueError('invalid duration')

    x = 1.5 * math.cos(math.radians(bearing_deg))
    y = 1.5 * math.sin(math.radians(bearing_deg))
    xml = XML.replace('pos="1.182 0.924 1.0"', f'pos="{x:.10f} {y:.10f} 1.0"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = MuJoCoSceneAdapter(model, data,
                                 array_site='head_array', source_sites={'beacon': 'beacon'})
    array = MicrophoneArray.from_positions(
        [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
        array_id='head_array', frame_id='robot/head', geometry_id='asym90mm_v013')
    signal = excitation(duration_s + 0.5)
    if mode == 'silence':
        signal[:] = 0.
    runtime = HearingRuntime(array, (ScheduledSource('beacon', signal, 0),), sample_rate_hz=FS)
    selector = AcousticOnlySelector(
        MotionAwareSRPLocalizer(MotionSRPConfig(
            block_samples=320, hop_samples=160, angular_resolution_deg=2,
            max_candidates=3, min_relative_score=.65, min_prominence=.012)),
        AcousticSelectionConfig(min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
    bridge = WindowedMuJoCoPolicyBridge(runtime, adapter, selector,
                                         pipeline_config=WindowedPipelineConfig(perception_period_ns=100_000_000))
    bridge.observe(0)
    accepted = []
    samples = 0
    last = None
    # Customer policies would read frame.features here. No robot action is prescribed.
    for step in range(1, round(duration_s / DT) + 1):
        mujoco.mj_step(model, data)
        if step % 10 != 0:
            continue
        frame = bridge.observe(round(float(data.time) * 1e9))
        samples += frame.audio_samples
        last = frame
        if frame.new_decision and frame.features[3] == 1.:
            accepted.append(float(math.degrees(math.atan2(frame.features[0], frame.features[1]))))
    if last is None:
        raise RuntimeError('no sensor frames')
    if mode == 'sound':
        error = (accepted[-1] - bearing_deg + 180) % 360 - 180 if accepted else None
        passed = bool(accepted and abs(error) <= 10.)
    else:
        error = None
        passed = not accepted and bridge.pipeline.accepted == 0
    passed = passed and samples == round(duration_s * FS)
    summary = dict(status='PASS' if passed else 'FAIL', mode=mode,
                   simulator='mujoco ' + mujoco.__version__,
                   duration_s=duration_s, microphone_channels=array.channels,
                   audio_samples=samples, decisions=bridge.pipeline.decisions,
                   accepted_decisions=bridge.pipeline.accepted,
                   last_valid_bearing_deg=accepted[-1] if accepted else None,
                   evaluation_only_bearing_deg=bearing_deg if mode == 'sound' else None,
                   evaluation_only_error_deg=error,
                   last_features=last.features.tolist(),
                   source_truth_exposed_to_policy=False,
                   policy_actions_executed=0)
    if strict and not passed:
        raise RuntimeError('v0.15 native policy observation gate failed: ' + json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sound', 'silence'), default='sound')
    parser.add_argument('--source-bearing', type=float, default=38.)
    parser.add_argument('--duration', type=float, default=2.4)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    summary = run_episode(mode=args.mode, bearing_deg=args.source_bearing,
                          duration_s=args.duration, strict=args.strict)
    if args.output_dir:
        dest = Path(args.output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
