# v0.5 Experimental Multisource Spatial Interfaces

## Scope and implementation status

This release adds **four independent interfaces**: `DirectionEstimator`, `MultiSourceLocalizer`, `SourceTracker`, and `SourceSeparator`. The concrete `SRPCandidateLocalizer` performs planar far-field SRP-PHAT scanning with peak prominence, pair-support, suppression radius, and silence rejection. It returns **spatial hypotheses**, not certified independent sources. `CandidateTracker` provides provisional fixed-frame bearing tracks with confirmation, coasting, and deletion. `UnsupportedSeparator` deliberately raises `SeparationUnavailableError`: **no real waveform separator is shipped**.

All prior v0.4 APIs remain available, including `onda_hearing.localization` and `onda_hearing.tracking`. The v0.5 implementation is intentionally opt-in under `onda_hearing.spatial`.

## Physical and algorithmic assumptions

- Planar XY-array, at least three non-collinear positions. Any microphone count satisfying this condition is accepted. No array pattern is hardcoded.
- Far-field plane-wave steering and scalar sound speed. No near-field distance estimation or elevation.
- Cross-spectrum PHAT normalization; the score is **not a calibrated probability**. Peak and pair-support thresholds are heuristic and require real acoustic calibration.
- FFT-based SRP uses whole analysis windows; this reference is **offline**, not a certified real-time streaming backend.
- Repeated spatial hypotheses do not prove physical source identity, even if tracks become confirmed.
- Bearings must be expressed in the **same fixed frame** before tracking. For moving arrays, use time-aligned calibrated transforms. v0.5 does not implement this transformation.
- Source separation is **not implemented** and is not silently replaced with beamforming.

## Deterministic benchmark

Run `PYTHONPATH=src python benchmarks/multisource_matrix.py`. It produces `benchmark_results/v05_multisource/results.csv` and `summary.json`. The benchmark uses independently seeded broadband sources and the v0.4 free-field renderer, never injecting source positions into inference. Cases: one source, two well-separated equal-power sources, two sources 30° apart with 10 dB imbalance, two sources with 20 dB imbalance, and three sources. **This first matrix does not yet exercise real room RIRs or moving arrays; those remain explicit acceptance gates.**

Benchmarks are exploratory, not pass/fail claims for all conditions. The output `unmatched_candidates_proxy` is not a formal optimal-assignment false-positive metric and should not be presented as such. For a public qualification suite, add one-to-one assignment, false-positive rates, recall, track identity switches, confidence calibration, room RIR distributions, measured ego noise, moving microphone poses, and held-out physical recordings.

## API

```python
from onda_hearing.spatial import SRPCandidateLocalizer, SRPConfig, CandidateTracker

localizer = SRPCandidateLocalizer(SRPConfig(max_candidates=3))
frame = localizer.estimate(audio_packet, microphone_array)
tracker = CandidateTracker(confirmation_hits=3)
tracks = tracker.update(frame, timestamp_s=0.25)
```

The `LocalizationFrame.certified_source_count` property is deliberately false for the bundled implementation. External backends should use capability negotiation and supply an independent qualification report before advertising multisource support.

## Release gates before production use

1. Correctly reject spurious peaks on one-source reverberant signals.
2. Recover one, two, and three simultaneously active sources across separations, levels, and broadband/tonal signals; evaluate both positive and negative examples.
3. Compare beamforming/separation backend output with known clean signals using objective separation metrics.
4. Validate tracking while microphones rotate, with matched timestamped robot poses and identity-switch accounting.
5. Bound processing latency, memory, drop behavior, resets, and long-running stability.
6. Test on real arrays with calibrated channel geometry, gains, timing, robot ego-noise, and independent physical ground truth.

Until these gates pass, v0.5 is a research and engineering baseline and is **not** a deployable source-separation or robust multi-target-tracking solution.
