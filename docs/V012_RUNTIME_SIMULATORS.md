# Onda Robot Hearing v0.12: runtime and simulator adapters

## Status

This is an **experimental, offline-reference** shared hearing runtime with two thin
pose adapters. All existing 121 v0.11.1 regression tests are retained. The new
unit tests exercise timing, waveforms, resets, trajectory interpolation, and
simulator stand-ins; running the real Isaac Sim/real MuJoCo binaries is a separate
integration qualification gate. No real-time latency guarantee is made.

## Architecture and trust boundary

Simulator adapter -> immutable `SceneSnapshot` (positions, orientation, timestamp)
-> `HearingRuntime` -> `DirectPathRenderer` -> `AudioPacket` -> optional
`DirectionEstimator`/`SpatialBackend` -> `HearingObservation`.

Only the renderer has access to source positions. Perception receives **only** the
PCM packet and *fixed local* microphone-array calibration. Results from the
experimental source localizers are hypotheses, not certified source identities.
The default runtime emits an empty `SpatialResult` (`no_perception`) rather than
inventing detections. Scene ground truth is retained outside the returned
observation.

## Time and frame conventions

- `SceneSnapshot.time_ns` is a nonnegative integer in a **single simulation clock**.
  The caller must supply all scene snapshots using that clock. `clock_id` is
  explicit on every output `AudioPacket`.
- First `advance()` establishes the timeline epoch and yields no packet.
  Subsequent calls must strictly increase time; `reset()` is required after
  simulator rewind/restart. Pausing means **do not call advance with the same
  timestamp**.
- Cumulative output frame count is `floor((time_ns-epoch_ns)*fs / 1e9)`.
  This avoids accumulating floating-point rounding error from repeated physics
  steps. Zero-output steps still update the pose history.
- Output intervals come from cumulative sample boundaries. If simulation steps
  subdivide audio samples, only completed samples are emitted. The final sample
  count at a given time is independent of physics step partitioning.
- The runtime uses absolute source waveform times; source start times are
  expressed as integer nanoseconds in the simulation clock.
- Right-handed world XYZ in meters; local-array-to-world orientation is an
  active XYZW quaternion. Array `MicrophoneArray` positions are in its local frame.
- The acoustic renderer uses retarded source emission time and interpolation of
  timestamped poses. Simulation pose histories need sufficient coverage for
  propagation delays. Interpolation/constant-pose approximation can still be
  inaccurate under abrupt motion and poorly sampled trajectories.

## Optional simulator integrations

**MuJoCo:** `MuJoCoSceneAdapter(model, data, array_site='...', source_sites=...)`
reads live site `xpos` and `xmat`. Call `mujoco.mj_forward` or `mujoco.mj_step`
before sampling. It copies state and never retains views into mutable `mjData`.
The included `examples/mujoco_live_hearing.py` is the first executable live
simulation example; requires `pip install -e '.[mujoco]'`.

**Isaac Sim:** `IsaacSimSceneAdapter(stage, array_prim='...', source_prims=...)`
uses USD `XformCache` and extracts the current prim world transform. It must be
instantiated inside a compatible Isaac Sim/Kit Python runtime with `pxr`. The
caller owns `SimulationApp`, stage loading, physics stepping, and clocks. USD
stage units and up-axis must be **meters / Z-up** (or translated explicitly by
the application before entering Onda). The adapter does not currently enforce
USD `metersPerUnit` or up-axis; the caller must verify these. There is no
Omniverse Kit extension or Isaac Lab `SensorBase` implementation yet.

## Known limitations and explicit qualification gates

1. `DirectPathRenderer` is a scalar free-field reference. It does not model
   moving-room RIRs, occlusion/diffraction, microphone directivity, calibrated
   frequency response, or actuator ego-noise. Static measured RIR utilities exist
   separately but are not automatically combined with moving free-field audio.
2. Source waveforms must be mono samples at the selected sample rate (no automatic
   resampling). Source definitions are fixed within an episode. Modifying source
   signals dynamically is unsupported.
3. The renderer is Python/NumPy and re-renders each chunk using pose histories.
   This is not qualified for large multi-environment GPU training or hard
   real-time execution. `AcousticConfig.max_samples` limits a **single** advance,
   but currently pose history grows for the entire episode. For long production
   episodes introduce bounded interpolation history and a streaming convolution
   engine with explicit propagation-delay coverage and performance benchmarks.
4. The renderer never silently normalizes or clips. If a mixture exceeds ±1,
   advance raises; lower source gain or implement an explicitly calibrated gain
   stage upstream of a production output path.
5. The adapter obtains scene **positions** for sound source placement only.
   A virtual source can be scheduled using WAV content through caller-provided
   `ScheduledSource`. Sound directivity, spatial source extent, semantic labeling,
   and moving-room reflection acoustics are still future work.
6. Physics tests use synthetic reference motions. Real MuJoCo and Isaac Sim
   smoke tests and benchmark source-recovery tests remain release gates.
7. ODAS Docker is a separate offline benchmark integration, **not** a low-latency
   Isaac Lab backend. ODAS candidate filtering/time alignment remain under study.
8. Persistent world-frame tracking and reliable source separation are not
   established by this release.

## Minimal pattern

```python
array = MicrophoneArray.from_positions([...], array_id='head', frame_id='robot/head', geometry_id='head_v1')
source = ScheduledSource('phone', waveform_1d, start_ns=0, amplitude=.02)
hearing = HearingRuntime(array, (source,), sample_rate_hz=16000)
hearing.advance(adapter.snapshot(0))
# Then, after each physics step:
output = hearing.advance(adapter.snapshot(sim_time_ns))
if output is not None:
    audio = output.audio        # real multichannel PCM
    observation = output.observation  # empty unless perception backend supplied
```

## Source release hygiene

The public SDK should publish code licenses and third-party notices only after
review. `pyproject.toml` intentionally does not invent a license choice.
