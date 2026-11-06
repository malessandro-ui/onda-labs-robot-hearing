# Implementation gates and next code batches

## Batch 1 (included): independent numerical core

Run tests and example. Verify correct azimuth sign and mixture waveform. Currently the tracker operates in the array frame and does not compensate rotation.

## Batch 2: multi-source DSP and source-to-track identity

Implement STFT pipeline, source-count hypothesis, SRP candidate generation with stronger peak validation, constrained data association, separation backend interface, track-owned separated waveform stream, and explicit ambiguous/unresolved outputs. Add tests for known two/three source overlaps, co-located sources, unequal levels, reverb, and speech. Do not guarantee separation from microphone count alone. Integrate ODAS as a process or native library backend only after licensing/process audits and timestamped output parity.

## Batch 3: robust temporal geometry and Motion v7

Implement interpolated pose buffer, SE(3) transforms at capture timestamps, world-frame bearing measurements and covariance, ID management, clock synchronization checks, discontinuity handling, robot IMU/joint state bridge. Validate 90-degree rotations and persistent IDs from physical ReSpeaker capture.

## Batch 4: RIR, domain randomization, noise

Add pyroomacoustics adapter for static shoebox; measured RIR convolution; bounded parameter distributions conditioned on sensor/environment, per-channel gain/time response, actuator-state noise sample library. Quantify real/sim mismatch, avoid randomization outside plausible ranges.

## Batch 5: Isaac Sim, MuJoCo, Isaac Lab

Separate adapter packages, scene binding and source schedules; Isaac 5.1 runtime startup after SimulationApp; MuJoCo site/body binding and stepping; Isaac Lab tensor observations, resets and no truth leakage. Include compatibility smoke tests in real installed runtimes.

## Batch 6: semantic BEATs, inference runtime, audio-visual grounding

Expose semantic streaming/event boundary backend, model checkpoints and vocabulary pinned by hash; paid Onda runtime BEATs/ODAS/Motion v7; camera detection and calibrated ray association; optional customer-defined text descriptions and safe policy handoff.

## Batch 7: real-world qualification

Record spatial multichannel ground truth across microphone geometries, rooms, ego-noise, motion, and overlapping sources. Publish error quantiles, ID-switch counts, latency and CPU/GPU/RAM footprints; perform uptime and resilience tests before claiming production reliability.
