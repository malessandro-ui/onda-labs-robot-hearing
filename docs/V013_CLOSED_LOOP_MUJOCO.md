# v0.13 — MuJoCo Hear-and-Turn: engineering and acceptance notes

## Scope and interfaces

This version adds a bounded audio-window buffer, an acoustic-evidence selector,
a head-yaw setpoint controller, a MuJoCo experiment, and evaluation-only logs.
It is an **experimental single-source signal-driven control demonstration**,
not a production control system or a benchmark for unknown source count.

```text
MuJoCo mjData site poses + scheduled source audio
  -> SceneSnapshot -> HearingRuntime -> DirectPathRenderer
  -> AudioPacket -> HearingWindowBuffer + timestamped array pose history
  -> MotionAwareSRPLocalizer (world-frame angular candidate)
  -> AcousticOnlySelector (score, ambiguity, temporal stability)
  -> AcousticBearing (world angle, capture end timestamp, evidence score)
  -> HeadYawController (base yaw, joint limits, slew-limited target)
  -> real MuJoCo position actuator -> next mj_step()
```

Only the acoustic renderer and **evaluation-only** logger receive simulator
sound-source coordinates. The localizer receives PCM, fixed microphone geometry,
and array pose samples. The controller receives an `AcousticBearing` and the
robot's own base orientation. No simulator source position, source ID, or target
angle reaches the head controller. This separation is checked structurally and
by control-invariance tests; it is not a security boundary against malicious
Python plugins.

## Run on a Mac with MuJoCo

```bash
python -m pip install -e '.[mujoco,dev]'
python -m pytest -q
python examples/mujoco_hear_and_turn.py --strict
python examples/mujoco_hear_and_turn.py --mode silence --strict
python examples/mujoco_hear_and_turn.py --source-bearing -38 --strict \
  --output-dir benchmark_results/v013_hear_and_turn/opposite
```

Each run writes `microphones.wav`, `telemetry.csv`, and `summary.json` in its
selected output directory. `--strict` exits with a nonzero error when the
single-source final yaw error is over 10 degrees, no validated acoustic
observation was accepted, or the silence run moves more than 2 degrees.

`telemetry.csv` records elapsed time, estimated **world** bearing and score,
head joint angle, commanded yaw, control/selection reasons, and evaluation-only
true bearing/alignment error. The reported world angle comes from PCM through
motion-aware SRP and is a **hypothesis, not calibrated source identity**.

## Timing

- 500-Hz physics loop (2 ms), 50-Hz audio pose snapshots (20 ms).
- 16-kHz microphone PCM; 320 frames are emitted per nominal 20-ms scene step.
- 256-ms rolling acoustic window, evaluated every 100 ms.
- Motion-aware steering uses 20-ms FFT blocks and 10-ms analysis hops.
- Clock: common simulator nanoseconds; no robot IMU or clock drift modeled.
- Controller applies a 125 degree/second **command slew limit** and ±90 degree
  target joint limits. This is *not* a guaranteed physical velocity or torque
  bound, and position actuator hardware would require additional safety logic.

## Acoustic excitation

A seeded 300–3800 Hz broadband beacon is intentionally used rather than a
single sinusoidal telephone ring. Pure tones frequently do not constrain TDOA
well enough for reliable localization. It is a *localization probe*, not a
validated speech/telephone event recognition workload. A single stationary
source at ±38 degrees is used. The direct-path free-field renderer still lacks
measured microphone response, realistic reverberation and actuator ego-noise.

## Acceptance gates

1. **Sound:** estimated bearing comes only from PCM + the robot pose; head
   rotates toward the source, final absolute error at most 10 degrees, and at
   least one accepted acoustic observation.
2. **Silence:** no accepted bearings and no head movement greater than 2 degrees.
3. **Opposite source:** moving the simulated sound to −38 degrees produces a
   corresponding negative head rotation; source coordinates are not supplied
   directly to control.
4. Sample count must match duration × sample rate; no packet gap or clock
   discontinuity is silently accepted.
5. Deterministic synthetic API stand-in tests and optional real-MuJoCo test.

## Tests and honest limitation

The core regression suite is run independently of MuJoCo. The native MuJoCo
acceptance test is **skipped** when `mujoco` is not installed. Successful tests
with the stand-in do NOT prove real MuJoCo behavior. Actual `--strict` runs
on macOS with MuJoCo 3.x are the gate for native integration.

There is no guarantee the strict gate will pass every configuration. Report
failures in the generated logs rather than modifying the source's truth or
quietly expanding tolerances. No source separation, multiple-source identity,
robotic navigation, semantics, persistent world tracks, measured physical
acoustics, or physical hardware certification is claimed.

## Suggested v0.14 work

After native MuJoCo confirmation: port this same audio-derived loop to a
headless Isaac Sim 5.x articulated USD scene using the existing Isaac adapter;
then implement Isaac Lab SensorBase integration with batchable, fixed-shape
observations, validity masks, per-env resets, explicit device placement, and
throughput/memory benchmarks. Performance and leakage tests precede Kit UI.
