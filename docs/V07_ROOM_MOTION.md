# v0.7 — Room and motion validation (alpha)

## Scope and physical assumptions

This release adds an **additive** `onda_hearing.rooms` module and `onda_hearing.motion` module, preserving all v0.6 public imports. The RIR engine implements direct and first-order specular image-source paths for a rectangular room. Every wall has octave-band **energy** absorption at 125, 250, 500, 1000, 2000, 4000, 8000 Hz; its approximate pressure-reflection magnitude is `sqrt(1-alpha)`. Interpolation between bands and a windowed FIR approximate frequency-dependent reflections. The model does **not** model wall impedance phase, diffuse scattering, diffraction, air absorption, higher-order reflections, source/receiver directivity, or changing RIRs during motion. It is not a calibrated real room simulation.

`MeasuredRIR` is an immutable `[taps, source, channel]` container with sample-rate, source order, microphone order, and provenance. `save`/`load` store structured non-pickled NPZ. Rendering uses **dry source PCM** and the full RIR including direct propagation. Explicit sample-rate or channel/source-order mismatches raise errors. An actual recorded measurement must come with physical channel ordering, geometry, excitation/deconvolution method, sample rate, and trustworthy timing calibration. The included example material values and synthetic RIRs are **not measured data**.

The `TimedBearing`/`to_world` contract uses a measured **array-frame** bearing and a same-clock array-to-world pose at the **actual capture time**. It rejects mismatched frames/clocks, pose extrapolation, and stale bracketing pose samples. A single static pose is treated as a fixed transform. It outputs **azimuth only** and does not infer range, elevation, or persistent track ID. Correct physical use requires array-to-robot extrinsics and synchronized capture/pose clock domains. `PoseTrajectory` rotation/position interpolation is inherited from v0.4.

## Reproducible benchmark

```bash
python benchmarks/room_motion_v07.py
```

Outputs `benchmark_results/v07_room_motion/{cases.json,results.csv,summary.json,mixtures/*.npz}`. Both the existing v0.5 SRP-PHAT and v0.6 subband SRP read the **same saved PCM** for each case; truth labels are used only for scoring. One-to-one angular matching has a 15° acceptance tolerance. Each window has 3200 samples (200 ms at 16 kHz). There are 27 cases: 12 separation/level mixtures, 9 first-order-room-reflection cases (absorption 0.15/0.5/0.85 with 1/2/3 sources), 3 signal-to-noise scenarios, and 3 **within-window continuous rotation** cases (45/90/180°).

In motion cases, direct-path PCM uses time-varying microphone positions and rotations. Each baseline, which assumes a fixed array, receives the midpoint world-coordinate geometry: this deliberately evaluates model mismatch caused by within-window rotation. The returned directions are already in the world-frame basis; do not rotate them a second time. **World-frame source-ID continuity is not benchmarked.** In room cases sources and receivers are stationary, so the room simulation is independent of the motion path. There is no moving-RIR propagation in v0.7.

### Development-environment result (one deterministic benchmark seed)

| Metric | SRP-PHAT v0.5 | Subband SRP v0.6 |
|--|--:|--:|
| True positives | 41 | 41 |
| False positives | 5 | 15 |
| False negatives | 10 | 10 |
| Precision | 89.13% | 73.21% |
| Recall | 80.39% | 80.39% |
| Exact candidate-count rate | 55.56% | 66.67% |

**Continuous rotation is the largest failure category.** Static-array methods are not motion-aware beamformers. The SDK makes no real-world accuracy, calibrated RT60, robust track-identity, source separation, or production latency claim. The per-condition reports, error cases, and full mixtures are retained so no false-positive tradeoffs are hidden. These statistics are dependent on particular source signals, geometry, thresholds and synthetic test choices. They should not be used as broad performance estimates.

## Follow-up qualification gates

1. Convolve with *externally recorded and calibrated* multi-source RIRs, record and disclose measurement methods, mic sensitivities, environment geometry and microphone channel order.
2. Quantify frequency-response, arrival-time, RT60 decay and C50/DRR against physical recordings (not simply RT60 configuration values).
3. Add time-varying room acoustics with validated reflection trajectories, Doppler/relative source motion tests, and array/channel calibration effects.
4. Implement timestamp-aware **time-varying steering**, motion-compensated localization and an independent world-frame tracker; benchmark ID switches and false tracks during continuous rotation.
5. Report uncertainty, per-room and per-hardware slices, confidence intervals across many randomized scenes, compute costs and latency before suggesting production readiness.
