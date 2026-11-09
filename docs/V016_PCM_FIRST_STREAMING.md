# Onda v0.16 experimental PCM-first streaming architecture

## Motivation and evidence

Prior native validation: v0.15 stationary and motion test gates passed at approximately 2° error using a CPU reference acoustic renderer. The initial threaded pose-per-policy-step wrapper reported 12 dropped updates, 6 resyncs, and zero valid observations. A deliberately paced diagnostic yielded 132 valid observations, zero dropped frames, and 2 startup resets. **Pacing is not a throughput improvement.**

## Data contracts

- Audio: finite, numeric `[frames, channels]` PCM at fixed `sample_rate`; monotonically increasing absolute `start_sample`. Gaps are detected; overlaps are rejected.
- Clock: `capture_start_ns` and `Pose.time_ns` are the same monotonic nanosecond domain. Sample capture times derived from indices; fractional sample-to-time roundoff is deterministic.
- Pose: world XYZ meters; array quaternion **XYZW** (Isaac Lab uses **WXYZ**, so convert at the boundary), maximum interpolation interval enforced, no extrapolation.
- Output: `HearingObservation` includes capture midpoint time, monotonic publication time, source sample interval, epoch, validity, confidence, local/world azimuth, and failure/rejection reason.
- Robot policy: a nonblocking copy of `[sin(local),cos(local),confidence,valid,age_s]`; stale and otherwise invalid results are zeroed. Metadata is not automatically fed to policy.

## Scheduling and overload behavior

1. Ingest all PCM in a finite-capacity preallocated circular ring. Overwrites are counted in **audio samples**; discontinuities are explicit and create a new audio epoch.
2. A separate worker processes windows of length `window_samples` at `hop_samples` increments. When behind, it **coalesces obsolete inference windows** and takes the most recent complete contiguous window. It does not splice samples and does not reset the audio epoch due only to inference demand.
3. Worker looks up the pose at **audio-window midpoint**, using SLERP for orientation. If there is no valid pose bracket, no bearing is published for that window; no extrapolation.
4. Worker may be busy or fail. The policy never waits for it: reads are O(1) and either see a valid fresh cached estimate or an explicit invalid/stale state. `worker_error` must be monitored.
5. The ring is bounded. If consumers fall behind its capacity, old PCM is overwritten, and unavailable inference windows are skipped. This is **not equivalent to lossless archival capture**. For evidence-grade audio retention, connect an independent recording sink.

## What is and is not optimized

A new vectorized, single-dominant-source planar GCC-PHAT/SRP reference estimator is provided, with a conservative absolute-coherence gate to reject independent microphone noise in basic tests and benchmarked independently. It is useful to investigate a faster algorithm, but it is not semantically or acoustically equivalent to the existing Onda v0.15 pipeline. The old `MotionAwareSRPLocalizer` must still be benchmarked on the same test suite before this backend can replace it. Do not advertise a speedup of the existing acoustic model based on these microbenchmarks.

## Native validation acceptance

The native runner creates Isaac Lab `Isaac-Cartpole-v0` with 2 environments and a rigid stationary source. It synthesizes 4-channel PCM from native pose tensors and a seeded mono waveform; source poses are used only by the renderer and evaluation. The worker sees **PCM only**, not the source ground-truth pose. It must produce at least 5 valid read observations per environment, each with 75th-percentile world-frame angular error at most 10°, no PCM discontinuity or overwrite, no worker exception, unchanged original policy observation shape, and a clean worker shutdown.

## Follow-up release gates

- Static and rotating arrays under a common independent audio-recording fixture, including local/world azimuth checks and timing alignment.
- Sustained simulated-time and wall-time throughput, at 1/2/4/8 environments, with per-stage P50/P95/P99 and overrun counts.
- Thread contention and worker-process isolation; CPU affinity, process priority, GC effects, buffer contention.
- Real microphone drivers with sample-count-derived timestamps, device reset and drift handling, body-to-mic calibration and IMU/ROS2 pose synchronization.
- Reverberant and multi-source tests; hardware ego-noise evaluation; backpressure and loss policies defined per hardware and application.
