# Onda Labs Robot Hearing — Architecture and Engineering Report

**Status:** Architecture baseline + runnable alpha foundation, November 2026.
**Commercial license:** deliberately undecided.
**Design target:** robotics teams that need real acoustic sensing in simulation, an upgrade path to deployed microphones, and reliable policy-facing observations.

## Executive decision

**One SDK, two simulator adapters, any physically valid microphone geometry.** Onda's core must be simulator-neutral; NVIDIA Isaac Sim and MuJoCo provide body/source poses, asset links, and simulation clocks. The propagation engine generates actual multi-channel PCM, which is then passed to perception. An Isaac Lab/Omniverse extension is an optional convenience integration, not a separate SDK or inference implementation.

Onda should also expose a **free/local baseline** and an interchangeable **paid perception backend** based on the existing BEATs + ODAS + Onda Motion v7 work. The paid service should support edge runtime and streaming API. It must not be a mandatory dependency of the SDK's baseline simulation capabilities.

## Product boundaries

1. **Core and contracts:** validated array geometry, timestamped PCM, coordinate frames, source observations, explicit uncertainty/status, model provenance.
2. **Acoustic simulation:** customer WAV upload, arbitrary overlapping sources, propagation, room responses, directivity, calibrated noise, source schedules, reproducible seeds.
3. **Perception:** semantic event identity and embeddings, DOA/TDOA, multi-source candidate localization, source separation (when feasible), voice/noise handling, persistent source tracks, array/robot motion compensation.
4. **Audio-visual fusion interface:** audio rays + uncertainty, calibrated camera/robot transforms, user-provided detections/depth/VLM objects, probabilistic association with `unmatched` output.
5. **Policy observation adapters:** Isaac Lab, Gymnasium, ROS 2, VLA clients, user-configured event-to-text descriptions. The SDK **must not** autonomously issue manipulation commands.

## Critical physical distinction

Audio source detection, direction estimation, multi-source localization, waveform separation, and tracking are **five distinct operations**. Seeing two spectral peaks does not prove two physical sound sources; estimating two DOA candidates does not separate their waveforms; persistent track IDs do not prove stable acoustic identification. A Kalman filter stabilizes estimated state but cannot repair bad acoustic observations. ODAS is a reference for jointly using SSL, source tracking, and separation-related processing, but every claim must be tested under overlap, reverberation, and robot motion.

Maintain separate observation forms:

- `DirectionCandidate`: frame, time, normalized direction, score, covariance/uncertainty, method, validity.
- `SourceTrack`: persistent ID, lifecycle (tentative/confirmed/coasting/lost), direction/position estimate with uncertainty, association evidence, timestamps.
- `SeparatedAudio`: optional mono/binaural stream with source/track association confidence, quality diagnostics, and sample-clock metadata.
- `SoundEvent`: semantic class confidence, onset/offset, embedding, and association to source track when justified.

If a model cannot separate two sources, report unresolved/ambiguous—not fake two independent streams.

## Cross-simulator core

One `PoseProvider` protocol returns named entity transforms at explicit simulation times. `AcousticScene` holds array specifications, sound source instances, and acoustic environment. Isaac Sim adapter reads USD world transforms; MuJoCo adapter reads body/site transforms. Both feed the same propagation engine; both yield the same PCM contract. Simulator truth (source identity, exact location) is exported separately for reward computation and benchmarking, not to perception.

For Isaac Sim 5.1, initialize `SimulationApp` **before** importing Kit-dependent APIs. Support a standalone script and separate Kit extension under `extensions/`, with version-pinned CI. Isaac Lab observation managers require strict [env, max_sources, features] shapes, availability masks, episode resets, and bounded update latency. MuJoCo uses `site_xpos`/`site_xmat`; copy transforms because `mjData` is mutable. Do not assume visuals/meshes imply acoustic materials.

## Microphone specification and calibration

Do not hardcode four microphones: require any supported set of per-channel 3D coordinates, orientations, directivity, frequency responses, gains, and clock metadata. Validate channel map and geometry rank; a single microphone cannot provide array TDOA; a linear array is ambiguous; a planar array cannot uniquely recover elevation everywhere. Model array-to-robot calibration and timestamped head/IMU/joint poses; raw gyro readings are not world pose. Pose compensation should precede world-frame tracking; provide robust rotation and clock failure detection.

## Multiple WAV source architecture

Use source-local dry audio where available. Each sound has source object link, 3D pose history, onset/stop schedule, gain, playback behavior, asset metadata, and a seeded variation configuration. The mix at microphone *m* is the sum over active sources of each waveform convolved with its time-varying source-to-microphone propagation response plus sensor/electronics noise. All per-channel received PCM must be synthesized before perception; never pass source class or exact position into a perception estimate.

A real-world sound library should store SHA256, provenance, sample format, license, clip class, dry/reverberant status, split, and commercial eligibility. FSD50K requires clip-level license review and is not automatically a set of dry isolated sound sources.

## Acoustic reality gap and hardware variability

Simulation fidelity tiers:

- A0: direct path, fractional delay and distance attenuation (the runnable foundation implements static A0).
- A1: physically constrained randomized room acoustic responses, appropriate ISM/RIR providers.
- A2: measured multichannel room impulse responses, hardware-characterized microphone responses, recorded robot ego-noise.
- A3: calibrated geometry/path acoustics and moving wavefield support, explicitly qualified per backend.

Avoid naive "randomize everything". Use plausible distributions conditional on environment, hardware type, and source. Separate: (a) environmental acoustic sources with spatial coherence, (b) channel electronics self-noise/gain/phase/ADC distortion, and (c) ego-noise conditioned on joint velocities, motor state, fan state, footfalls, and structural vibration. Record every randomization seed and parameters. Evaluate each domain-randomization setting against held-out real recordings. An RT60 range by itself is not a calibrated room model.

## Streaming production interfaces

`AudioBlock`: frame-major float32 PCM, frame ID, channel map, sample rate, capture start/stop, clock ID, sequence ID, drop count. Use fixed chunk sizes and bounded backpressure. Semantic BEATs windows are longer than DOA update windows; make them asynchronous with causal alignment and event history. Provide model version, threshold version, source tracking state, and inference-time vs capture-time metadata. No future audio is allowed in an online policy observation. Repeated ring events should remain tied to persistent phone hypotheses where justified.

Proprietary BEATs/ODAS/Motion v7 lives in a separate package/service implementing the same backend interface. Free DSP algorithms must remain genuinely usable. Cloud gRPC streams and REST batch jobs may be offered alongside local edge inference; do not require cloud for robot motion response.

## Audio-to-vision grounding

Audio supplies ray/bearing with uncertainty, not a 3D object pose or distance unless independently estimated. Visual detection supplies image boxes/masks, possibly depth/3D. Use capture-time transforms between array, head, base, odom, camera and map. Evaluate candidate object likelihood in an audio ray volume, include semantic compatibility and consistency over time, and preserve `unmatched`/ambiguous results. The customer owns policy decisions (orient, navigate, grasp); configurable descriptions such as "ringing phone at +30 degrees" are outputs, not mandatory action prompts. An event-to-command rule engine must be explicitly opted into by the integrator.

## Engineering acceptance criteria

| Gate | Required proof |
|---|---|
| G0 - foundation | Arbitrary microphone geometry, multi-WAV mix, deterministic delay math, zero ground-truth leakage, unit tests |
| G1 - spatial | Single-source DOA accuracy versus synthetic truth across azimuth/SNR and geometries, calibrated confidence |
| G2 - overlapping | Multi-source recall/precision, count accuracy, angular error, separation quality and source-to-track binding under overlap |
| G3 - motion | Time-aligned pose compensation and track continuity through head/body rotation, ID-switch rate, dropped packets |
| G4 - room | Measured vs synthetic RIR comparisons and documented reverberation/occlusion limits |
| G5 - adapters | Matching output between Isaac and MuJoCo for matched static scene trajectories, reset correctness |
| G6 - policy | Audio-guided tasks with sound-muted and sound-swapped controls, no privileged truth leakage |
| G7 - real hardware | Calibrated array, live capture, sustained uptime, latency percentiles, real room noise and real overlapping sources |

For sustained deployment, add stress tests, bounded memory, native-process restart behavior, observability, deterministic model loading, versioned wire schemas, security review, and compatibility tests. Claim production readiness only after G7 on customer-representative hardware.

## Code review lessons from PatrizioAcquadro/isaac-audio-sensors

The supplied review archive (commit `dc7ed2b`, two commits after v3.0.0) shows a useful separation of `core/simulation.py`, `core/perception.py`, analytic propagation modules, waveform I/O, effects chains, `isaac/sensor.py`, Kit extension, CLI, and integration/release tests. Study their typed signal blocks, plugin boundaries, source scheduling, array geometry and release checks. Do not copy giant simulator sensor orchestrators verbatim. The maintained multisource localizer explicitly does not maintain persistent source identity; the v3.0.0 pathway's nominal 750 ms temporal context and geometry/sampling fallbacks mean it is not sufficient for Onda's multisource/motion requirements. Onda must independently implement and benchmark source separation and persistent tracking rather than interpreting candidate peaks as separated tracks.

## Scope of included source release

The packaged `0.1.0a0` code implements geometry validation, immutable PCM blocks, static direct-path fractional-delay mixing of multiple sources, a planar signal-only GCC-PHAT angular search, a conservative bearing-only ID tracker, WAV utilities, MuJoCo pose readout helper, CLI, and tests. It deliberately **does not** promise reliable multisource separation, ODAS/BEATs/Motion v7 integration, Isaac/MuJoCo end-to-end adapters, calibrated RIR/noise, autonomous audio-visual grounding or production robustness. These must be built and validated through G1–G7.

## Research and implementation references

- Isaac Audio Sensors: https://github.com/PatrizioAcquadro/isaac-audio-sensors
- Isaac Sim 5.1 workflows: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/workflows.html
- Isaac Sim standalone Python: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/python_scripting/manual_standalone_python.html
- MuJoCo official Python API: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo site/world frames: https://mujoco.readthedocs.io/en/stable/overview.html
- Pyroomacoustics RIR: https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html
- ODAS: https://github.com/introlab/odas
