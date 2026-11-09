# Onda v0.16: continuous PCM and native motion soak validation

## Scope and clock-domain correctness

This add-on **does not modify Onda's existing v0.16 SDK**. It tests the pre-existing
`StreamingHearing` PCM ring, worker, source-time alignment and policy read. It
uses an independent 60 Hz policy-reader thread and an explicitly wall-clock-paced
16 kHz, four-channel PCM producer (50 chunks/s). The default 1,800 s run takes
**30 real minutes**. A second, optional Isaac Lab test advances **simulation
seconds**, not necessarily the same wall-clock duration.

The wall-clock experiment uses one shared `time.monotonic_ns()` clock for PCM
source timestamps and poses. Its `capture_to_publication` metric is computed
from `observation.publish_mono_ns - observation.capture_time_ns`, for a single
synthetic microphone rig with source clock paced to actual wall time. It is **not
hardware ADC-to-output latency**. The native test instead reports *simulated*
capture age, because Isaac Lab advances its own simulation clock; the two
numbers must not be conflated.

## Acceptance gates

The 30-minute run checks:

- Every scheduled audio sample is supplied with consecutive absolute indices;
  no declared gaps; **zero `audio_discontinuities`**.
- **Zero `ring_overrun_recoveries`**: no unread inference window lost due to
  insufficient retention capacity. `overwritten_audio_samples` is expected to
  increase after four seconds and means *normal eviction of old PCM*. This is
  not an error by itself. `inference_windows_skipped` may grow during overload
  because the worker coalesces obsolete inference jobs while retaining PCM.
- Valid acoustic observations, unique sample-indexed decisions, world-bearing
  error statistics, no stale/invalid policy leakage, and no worker failure.
- Multiple deliberate **95 ms slow-estimator bursts** that affect only the
  worker, at about 25% and 65% through the run, while audio ingestion and
  policy reads continue. These are synthetic overloads, not CPU scheduling
  isolation or simulated physical robot noise.
- P50/P95/P99 **capture-to-publication**, first valid read, PCM ingestion,
  nonblocking policy read, and current resident memory trend. Exact individual
  samples are not stored: fixed-width bounded histograms give ~0.1 ms quantiles
  with explicit overflow, not unbounded Python lists.
- A per-minute atomic checkpoint JSONL stream with `fsync`, final atomic JSON,
  and nonzero exit on failed gates.

Default quality-of-service gates: P99 capture-to-publication ≤300 ms; P99
policy-read ≤10 ms; ≤5% producer/reader deadline misses (lag >two periods),
post-warmup RSS growth ≤64 MiB; world-bearing errors >10° for no more than 5%
of unique valid decisions. These are **experimental acceptance thresholds**, not
supported robot contracts or certified deadlines. Host scheduling and power
limits may cause tests to fail. Do not loosen thresholds to force a PASS.

The native test creates two Isaac Lab environments with a kinematic rotating
microphone rig and separate world-fixed sources. It verifies rig yaw with actual
rig quaternions, continuous PCM, per-environment world bearings and physics
policy/reward integrity. It reports sim-time separately from wall-time. Because
Kit shutdown previously stalled, only the **finite native validation harness**
uses a hard process exit after persisting results.

## Run

On the validated development branch:

```bash
cd /ephemeral/onda-nvidia/onda-v015
source /ephemeral/onda-nvidia/env_isaaclab/bin/activate
python -m pytest -q tests/test_streaming_pcm.py tests/test_streaming_soak.py

# Quick end-to-end, real elapsed 30 seconds, not certification:
PYTHONPATH="$PWD/tools:$PWD/src" python tools/soak_streaming_v016.py \
  --duration-s 30 --checkpoint-s 10 \
  --output docs/validation/streaming_v016_soak/wall_clock_quick.json

# Full 30 minutes of actual time; prints every-minute checkpoints:
bash tools/run_soak_validation.sh

# Native Isaac Lab: 180 *simulation* seconds (two envs, rotating rig):
bash tools/run_native_soak.sh

# Optional native 1800 simulation seconds, deliberately slow inferences:
ONDA_NATIVE_SOAK_DURATION_S=1800 ONDA_NATIVE_SOAK_OVERLOAD_MS=95 \
  bash tools/run_native_soak.sh
```

Inspect `docs/validation/streaming_v016_soak/*.json` and the `.checkpoints.jsonl`
files. Do not commit `.log` or hundreds of MB of generated audio. The soak does
not write raw PCM files.

## Limits and follow-up

This is synthetic single-source broadband input with an analytic point-source
renderer. No Isaac Lab audio propagation, environmental reflections, physical
microphone clock synchronization, ROS 2, humanoid actuation, ego noise, speech,
or multi-source tracking is validated. The Python-thread implementation is
best effort: it does not provide deterministic deadline guarantees, isolation
from the GIL, or safety certification. A genuine production acceptance suite
must include raw device capture, external clock diagnostics, long hardware
runs, fault injection (device reset, timestamps, IMU outage), and P99.9
requirements for specific robot configurations.
