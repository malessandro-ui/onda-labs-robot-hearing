# v0.6 — Paired multisource stress benchmark

## Scope

Adds `SubbandSRPLocalizer` as an independent **experimental** frequency-diverse SRP-PHAT candidate backend alongside the v0.5 SRP-PHAT baseline. It computes normalized per-frequency-band spatial responses and combines them through a high quantile. This is a *candidate generator*, not source separation, independently verified source-count estimation, or calibrated confidence.

`benchmarks/spatial_challenge_v06.py` creates 53 deterministic cases: 7 isolated sources and one silence trial; 30 two-source cases with 10°–180° angular separation and 0, −10, −20 dB level offset; 4 three-source scenes; 3 SNR conditions (20, 10, 0 dB); 3 synthetic echo conditions; and 5 fixed-orientation rotation conditions. Every estimator reads the exact same saved `mixtures/###.npz` (`pcm`, `positions_m`, `sample_rate_hz`). The evaluation ground truth is not delivered to the estimators.

## Metrics

Source hypotheses are one-to-one associated with truth within a 15-degree angular gate; two nearby true sources cannot receive credit from a single reported peak. `precision=TP/(TP+FP)`, `recall=TP/(TP+FN)`, exact count is number of reported candidate directions equals number of true sources. This count metric does not verify correct source identity. Raw CSV contains each scene and algorithm's predictions.

Results from the local full run (seed=21610, 53 scenes, 16kHz, 200ms, four-mic example geometry):

| Method | TP | FP | FN | Precision | Recall | Exact count |
|---|---:|---:|---:|---:|---:|---:|
| v0.5 global SRP-PHAT | 65 | 6 | 36 | 0.9155 | 0.6436 | 0.3019 |
| v0.6 subband SRP | 72 | 27 | 29 | 0.7273 | 0.7129 | 0.4906 |

**Interpretation:** More recall and more exact counts, *at the cost of markedly more false positives*. Neither backend satisfies production-grade unknown source-count detection. These parameters were not tuned on a separate train split; don't treat results as unbiased generalization estimates.

## Important limitations

1. All recordings are synthetic broadband noises. No speech, tones, room-scale diversity, microphone distortions, or real robot noise.
2. Echo stress uses a single delayed scaled mixture copy; it is **not physically measured, spatially correct multichannel RIR**. Calibrated reverberation must use the `convolve_rir` interface with independently measured RIRs or a qualified room model.
3. Rotated arrays are fixed at different orientations **within** each 200ms window. This measures orientation handling, **not continuous dynamic motion**, source tracking through rotation, or Doppler.
4. Each source is at 2.5m in horizontal plane. The algorithms only estimate 2D azimuth and use planar microphone geometry.
5. An extra spatial peak is not proof of an independent physical source. Track quality, identity continuity, separation, event semantics, and hardware calibration are untested.
6. The benchmark deliberately compares different algorithms on the same files, but is not independently real-world validated. Use disjoint held-out recordings, scene variability, and stronger statistical analysis before publishing accuracy claims.

## Commands

```bash
python -m pytest -q
python benchmarks/spatial_challenge_v06.py
```

Results in `benchmark_results/v06_spatial_challenge/{cases.json,results.csv,summary.json,mixtures/*.npz}`.

## Next gate

Replace toy echo tests with calibrated RIRs, introduce time-varying array trajectories and pose history, test 1–3 sources across multiple microphone geometries, and compare false-track lifecycles. Separately evaluate waveform separation with ground-truth clean stems and objective metrics. Do not optimize only for source recall.
