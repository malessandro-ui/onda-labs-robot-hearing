# Onda Robot Hearing v0.2.0-alpha — Engineering release notes

## Changes
- Replaces naive global top-N peak picking with local-maximum selection on a circular azimuth grid.
- Suppresses candidates with insufficient relative strength, prominence, or positive pair support.
- Validates GCC inputs and refuses nonfinite audio and invalid speed-of-sound parameters.
- Rejects overlapping or out-of-order streaming audio blocks.
- Applies tracker confidence gate, consecutive-hit confirmation, and idle-time eviction.
- Retains a strictly fixed frame assumption; upstream world-frame pose compensation is required for moving arrays.

## What is *not* solved
- Joint multisource DOA with unknown source count under overlap.
- Waveform separation or independent semantic classification of concurrent sounds.
- Global optimal assignment, source-crossing recovery, persistent physical identity.
- Real microphone calibration, room acoustics, ROS 2 or Isaac Sim integration.

## Acceptance evidence
Run `python -m pytest -q` and `python examples/overlapping_sources.py`.
Single-source tests reject the former mirrored phantom peak. The overlapping
example primarily reports the dominant direction, not both simultaneously.

## Next gate
Integrate a tested ODAS/other multisource provider behind a direction-measurement
contract; build controlled two/three-source truth benchmarks, and validate world-frame
tracking after synchronized pose transforms. Require actual separated waveforms
before claiming source separation.
