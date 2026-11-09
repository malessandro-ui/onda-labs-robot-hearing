# Onda v0.15: acoustic throughput and asynchronous control integration

**Status:** experimental performance/architecture add-on. Not part of v0.14 public API, not certified for real-time use.

## Baseline and goals

The validated native motion reference measured median `manager.advance()` 154.82 ms, P95 197.27 ms across two CPU-rendered environments; full Isaac Lab policy step P50 164.48 ms, P95 205.34 ms. The task's nominal policy interval is 16.667 ms. A 60 Hz robot policy must **not** synchronously wait for every full CPU acoustic computation.

We separate two problems:

1. **Throughput:** reduce acoustic compute per unit of audio by profiling independently and selecting a profile only when accuracy remains within bounds.
2. **Scheduling:** deliver cached hearing observations without blocking robot control on SRP; explicitly invalidate stale data and handle overload.

These are not interchangeable. Async scheduling doesn't create capacity. Optimizing acoustic window cadence risks latency and missed transient sounds; evaluate both.

## Design decisions

- **Single worker owner:** the acoustic manager, renderer, selector, clock, and buffer are owned by one thread. Caller only sees copied NumPy results/tensor observations.
- **Bounded input:** maximum 2 batches by default. `put_nowait` prevents accumulation of unlimited waiting scene snapshots. If full, drop pending items, increment epoch, zero published features, and use the newest pose to establish a new Onda audio history.
- **No invented audio:** dropped intervals never result in the manager jumping its episode clock and pretending that all intervening PCM arrived. Reconnect via `manager.reset()` / `seed()`.
- **Partial auto-reset:** after native Isaac Lab post-step reset, the worker resets those environment pipelines, advances unaffected environments and seeds reset environments from their new poses.
- **Epoch-fenced output:** a stale worker computation cannot publish into a new epoch.
- **Fail closed:** no fresh result, excess simulation lag, wall-clock expiry, or partial reset = invalid hearing (`valid=0`) instead of an obsolete bearing.
- **Failure propagation:** worker exceptions surface as errors in the calling path. They are not replaced by success data.
- **Telemetry:** dropped frames, resynchronizations, queue occupancy, processed/submitted frames, last acoustic cost, staging cost, age and lag.

## Where latency still occurs

The producer calls `IsaacLabSceneReader.read` and stages per-field GPU tensors to NumPy. Staging synchronizes GPU work and costs wall time. Creating GPU hearing observation tensors also consumes time. The GIL remains shared across Python threads. These costs must be profiled during an actual robot-control loop. A separate process with bounded IPC would be the next isolation level; even then, staging and copies must be measured.

## Measurement philosophy

- Do not compare noisy single iterations. Use P50/P95, warmup and a stable seed, then run native motion and silence gates.
- Distinguish `manager.advance()` wall time (full rendering+SRP+staging) from actual per-audio-window inference time and from end-to-end source-to-actuator latency.
- Report dropped frames, fresh-decision rate, staleness and queue depth next to throughput. A fast control loop outputting all-zero hearing is **not successful hearing**.
- The reference 10° angle threshold is an acceptance gate, not a claim about all rooms, sources, noise, microphone placements or microphone array calibration.
- This add-on does **not** change sampling, array geometry, learned semantics or source tracking algorithms.

## Release gate for robotics customers

Before advertising real-time robot hearing, require validated audio capture jitter, timestamp synchronization to robot pose/IMU, continuous rotations, transient response latency, mixed sounds and ego noise, reset/reconnect behavior, long-duration stress, memory stability, failsafe behavior and calibrated hardware. Define the advertised platform, maximum environment count, supported update rate and P95/P99 latency as measured SLOs. Run motion accuracy again for any optimized profile.
