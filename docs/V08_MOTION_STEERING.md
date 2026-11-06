# v0.8 engineering note — pose-conditioned world-frame SRP

## Status
Experimental, offline, planar, direct-path synthetic benchmark. It is **not** a deployment-ready multisource tracker, separator, source-count classifier, or certified motion compensation stack. The v0.7 APIs remain intact.

## Design
`MotionAwareSRPLocalizer.estimate(audio, body_array, poses, pose_clock_id='sim')` uses body-frame microphone offsets (meters), with world-from-array rotation sampled at each short-window midpoint. For every block it computes PHAT-whitened GCC for microphone pairs and scans candidate world azimuths using world-rotated pair baselines. It aggregates spatial responses across blocks before circular peak detection. Neither ground-truth source positions nor their count enter inference.

**Frames:** `body_array.positions_m` is in `array.frame_id`, not world coordinates. `PoseTrajectory` holds `world_from_array` rotation. Results have `frame_id='world'`. Do not apply `to_world()` to these already-world observations. Source ranges cannot be inferred from azimuth alone.

**Clocks:** the source `AudioPacket.capture` clock must exactly equal `pose_clock_id`; clock offsets and drift must be explicitly resolved upstream. Pose interpolation requires bracketing samples with a maximum nearest-neighbor time gap, unless a single fixed-pose trajectory is explicitly used. No extrapolation is allowed for time-varying poses. Recording time base is nanoseconds; pose trajectory times are seconds in the same time domain.

**Window assumptions:** array pose and travel time variations within one FFT block are approximated by the midpoint pose. The renderer uses continuously time-varying positions; the estimator does not yet model continuous per-sample acoustic steering. High rotation speeds, Doppler, head/body flex, incorrect mic geometry, or clock drift can cause degradation. PHAT peaks are not independent-source proofs. Full 3D direction/elevation, near-field propagation, room multipath steering and waveform separation are out of scope.

**Why separate tracking?** `MotionAwareSRPLocalizer` emits independent candidate observations. It must not assign source IDs. Source identity is a different inference problem requiring temporal data association, uncertainty, occlusions, track lifecycle and evaluation for false confirmations and ID switches. A world-frame tracker is not included in the v0.8 acceptance claim.

## Benchmark
Run `python benchmarks/motion_steering_v08.py`. It writes reproducible PCM, body offsets, pose rotations and truth to `benchmark_results/v08_motion_steering/mixtures/` and reports per-case paired results in `results.csv`, plus `summary.json` and scene definitions. Both algorithms receive the same PCM. The **static** baseline gets the world geometry corresponding to the midpoint pose, while the new method gets body geometry + the entire time-aligned pose history. This is an explicitly different information condition, not a model-only apples-to-apples comparison.

12 scenarios: 0°, 45°, 90°, and 180° rotation, each with two fixed, distinct world sources and angular separations of 45°, 90°, or 120°, level differences of 0, -6 or -12 dB. At 16 kHz, 200 ms, 4-microphone planar array, 15° matching tolerance. Scoring uses maximum one-to-one source matching, with TP/FP/FN and exact candidate count. This is a small curated deterministic benchmark; **generalization cannot be inferred**. The audio is free-field; moving reflections, sensor dynamics, actuator ego noise and measured room RIRs are absent.

## Numerical limitations and next release gates
- Add synchronised hardware microphone + IMU/robot-odometry capture and **actual** clock calibration.
- Independently validate moving acoustic propagation against measured signals and retarded-time analytical cases.
- Stress orientation rate (°/s), source angle, spectral profile, level imbalance, microphone geometry, source motion, signal dropout, reverberation, noise and multi-second sequences.
- Evaluate true negatives, false candidates, missed sources, worst-case angular error, stable source counts, latency and resource usage.
- Only then introduce world-frame source tracking with identity switches, false-track confirmations, re-identification and ground-truth crossing tests.
- Test backward compatibility and reproducible build on supported Python versions, including Python 3.14 used on macOS.

## New modules
- `src/onda_hearing/spatial/motion_srp.py`
- `benchmarks/motion_steering_v08.py`
- `tests/test_motion_srp_v08.py`

All v0.7 interfaces and tests are retained. Previous `SRPCandidateLocalizer` and `SubbandSRPLocalizer` remain optional alternatives.
