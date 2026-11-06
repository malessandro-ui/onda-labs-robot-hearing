# Onda Robot Hearing v0.9 — Motion localization stress test

## Scope and truth-in-advertising

This release adds a paired, deterministic **diagnostic benchmark**, not a new validated source-count estimator and not a production world-frame tracker. The default motion-aware SRP estimator remains unchanged from v0.8. All 75 original tests still pass, alongside eight new tests.

The benchmark synthesizes two stationary world sources using independent band-limited Gaussian signals. It samples 13 controlled cases covering rapid 90°/180°/270° rotation, 0.8-second recordings, close source bearings (30°), 20 dB source imbalance, additive noise, synthetic yaw perturbations, and a known 5 ms clock offset with correctly aligned and deliberately uncorrected comparisons. A static **first-order shoebox RIR** scenario is included. **Moving-room reflections are not simulated**: the room-RIR case has a fixed array. The RIR is numerical and not a measured room.

Every estimator configuration receives the *identical stored PCM* per case and the same noisy or corrected pose history. The four variants are the v0.8 settings; 1.5° angular grid; 160-sample blocks/80-sample hop; and lower candidate thresholds. No ground-truth source bearings enter localization.

## Results — development environment

| Variant | TP | FP | FN | Precision | Recall | Exactly resolved cases |
|---|---:|---:|---:|---:|---:|---:|
| v0.8 default | 17 | 1 | 9 | 94.44% | 65.38% | 4/13 |
| Finer 1.5° scan | 17 | 1 | 9 | 94.44% | 65.38% | 4/13 |
| Shorter FFT blocks | 17 | 1 | 9 | 94.44% | 65.38% | 4/13 |
| Relaxed candidate selection | 19 | 1 | 7 | 95.00% | 73.08% | 5/13 |

A result is a TP when a candidate direction matches a truth direction within 15° and the match is unique. `exact_count_rate` requires **all** true sources matched and no extra candidates, not merely equal output length. Cases may produce zero candidates. Precision/recall aggregate over all cases. Results do not prove source-count identification, independence, track identity, calibrated pose error, or actual waveform separation.

## Diagnostics and reproducibility

```
python -m pip install -e '.[dev]'
python -m pytest -q
python benchmarks/motion_stress_v09.py
```

Outputs under `benchmark_results/v09_motion_stress/`:

- `cases.json`: all scenario metadata and parameter values.
- `mixtures/*.npz`: immutable input PCM plus ground truth and pose series used in evaluations. The benchmark *writes* these with `np.savez_compressed`.
- `results.csv`: per-case candidates, true/false positive counts, source misses, rejection reason, matched angular error, algorithm settings.
- `summary.json`: aggregate and case-by-case metrics.

## Clock semantics

`ClockAlignment.pose_to_audio_offset_s` means `audio_timestamp = raw_pose_timestamp + offset`. Incorrect clock IDs are rejected. Only a known **constant offset** is handled: there is no clock drift estimation, latency calibration, IMU integration, or robot-state-estimator interface. The uncorrected case deliberately violates temporal alignment and is tagged as such in the case name. A caller must supply clock calibration from a trusted source.

## Pose-noise semantics

`perturb_poses` applies deterministic, independently seeded Gaussian yaw and XYZ perturbations at each pose knot. It is a **synthetic diagnostic**, not a sampled MEMS IMU noise/drift/bias model. Pose uncertainty is not propagated into probabilistic DOA covariance.

## Known limitations and next gates

- Reflections for moving arrays are not modeled; cannot infer motion performance in actual reverberant rooms.
- The array is a four-microphone reference rectangle, with two stationary sources. Arbitrary geometry is supported in the SDK contracts but not comprehensively qualified here.
- The estimator assumes a far-field planar wavefront within each analysis block and a known, synchronized rigid-array trajectory.
- No real microphone recordings, real pose timestamps, or physical room RIRs have been compared.
- Persistent world-source tracking is intentionally **not** part of this milestone. Before it becomes appropriate, test source-count recovery across larger held-out motion, echo, and SNR configurations, with a declared false-positive budget and independent recording corpus.
- Do not choose production thresholds based solely on this 13-case benchmark. Use held-out calibration and evaluation splits.
