# v0.14.0-alpha — Isaac Sim hearing-to-motion bridge

## Proven vs. pending

**Proven on the user's Mac:** Onda v0.13's native MuJoCo strict single-source test: +38° ground truth, +36° final yaw, 2° error, 157 tests passing after specifying `<compiler angle="radian"/>`.

**Proven in the v0.14 development environment:** the inherited tests plus the new simulator-neutral Isaac hearing loop, Isaac action wrapper, negative controls, and test of the MuJoCo unit regression. The Isaac test plant is synthetic, not NVIDIA PhysX.

**Not yet proven:** running the example inside an installed Isaac Sim, the experimental minimal USD scene's articulation physics, USD prim organization in a customer's robot, performance with GPU-batched Isaac Lab envs, real-time audio, audio-visual grounding, real-robot calibration, moving-source/multisource tracking. Do not call a passing synthetic test native Isaac verification.

## API boundary

- `IsaacSimSceneAdapter` reads world poses from USD. Source positions are supplied only to the acoustic renderer.
- `IsaacSingleArticulationYaw` reads a named joint and calls `SingleArticulation.apply_action(ArticulationAction(...))` in **radians**, never the instantaneous `set_joint_positions` method.
- `IsaacHearingLoop` manages sample-accurate hearing packets, a bounded audio buffer, motion-aware SRP, conservative selection, and the joint setpoint controller. It is a single-robot research harness, not a production safety or Isaac Lab batched sensor.
- `before_step(k)` must be invoked before physics update `k`, and `after_step(k)` after it. `k` starts at 1; `reset()` must precede step 1. Audio and pose timelines use exact integer nanosecond timestamps derived from the configured physics frequency. Reset the **physics world first**, then reset the Onda loop.
- The user-supplied robot must be z-up, have a single yaw hinge rotating around positive Z, a valid position-controlled drive, and a stationary base with zero world yaw. Dynamic-base transform handling is future work.

## Required runtime

Isaac Sim requires a supported NVIDIA GPU / software environment. Do not attempt to install Isaac Sim into the ordinary MacBook CPU-only SDK virtual environment. In an Isaac Sim 5.1 installation, install the SDK into Isaac's Python (`./python.sh -m pip install -e /path/to/onda-robot-hearing`); then run under Isaac's `./python.sh`.

### 1. Construct experimental USD scene

```bash
./python.sh /path/to/onda-robot-hearing/examples/isaac_make_demo_stage.py \
  --output /tmp/onda_isaac_head.usd
```

The generated USD is a preliminary minimal single-joint PhysX scene; first verify the articulation drives correctly before relying on the result. For a production Isaac robot you should instead supply its existing USD articulation asset.

### 2. Run live closed loop

```bash
./python.sh /path/to/onda-robot-hearing/examples/isaac_sim_hear_and_turn.py \
  --usd /tmp/onda_isaac_head.usd \
  --robot-prim /World/Robot/Head \
  --array-prim /World/Robot/Head/MicrophoneArray \
  --head-joint head_yaw --source-prim /World/Beacon \
  --headless --strict \
  --output-dir /tmp/onda-isaac-sound
```

The script saves `microphones.wav`, `telemetry.csv`, and `summary.json` whether or not the `--strict` condition passes (unless it fails before recording). It does not fudge physics errors.

### 3. Negative control: silence

```bash
./python.sh /path/to/onda-robot-hearing/examples/isaac_sim_hear_and_turn.py \
  --usd /tmp/onda_isaac_head.usd \
  --robot-prim /World/Robot/Head \
  --array-prim /World/Robot/Head/MicrophoneArray \
  --head-joint head_yaw --source-prim /World/Beacon \
  --headless --mode silence --strict \
  --output-dir /tmp/onda-isaac-silence
```

### 4. Regression test on Mac

```bash
python -m pip install -e '.[mujoco,dev]'
python -m pytest -q
python examples/mujoco_hear_and_turn.py --strict
python examples/mujoco_hear_and_turn.py --mode silence --strict
python examples/mujoco_hear_and_turn.py --source-bearing -38 --strict \
   --output-dir benchmark_results/v014_mujoco_opposite
```

## Acceptance gate

For clean single-source cases, at least one **new accepted localization decision** (not a retained telemetry row), final head alignment error <=10°, no incorrect coordinate-frame or actuator unit assumptions, and exact expected PCM sample count. In silence, zero accepted decisions and <=2° of joint motion. Opposite source direction should reverse the commanded movement. All tests must use microphone-derived estimates; no source position may leak to the motor controller. Results are scenario-specific, not general accuracy claims.

## Next milestone

Once native Isaac Sim works with realistic robot USD assets, add a configurable Omniverse/Isaac Lab sensor interface and fixed-shape batched observations, with independent reset and measured throughput. Do not equate 1-robot Python sequential SRP performance with thousands of GPU environments.
