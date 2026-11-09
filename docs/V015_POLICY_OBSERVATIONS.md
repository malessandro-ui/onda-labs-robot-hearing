Onda v0.15: Simulator-independent policy observations
======================================================

Scope
-----

This additive patch introduces a five-float acoustic observation ABI with one
implementation shared by MuJoCo, Isaac Lab and future adapters (Gazebo/Genesis).
It does not replace the v0.14 hearing runtime, estimator, or control logic.

Feature order (float32, shape (5,) per environment):

1. bearing_sin: sine of bearing in the configured reference frame
2. bearing_cos: cosine of bearing in the configured reference frame
3. evidence_scaled: min(raw spatial score / evidence_scale, 1.0)
4. valid: 1 for accepted and non-stale evidence, 0 otherwise
5. age_s: elapsed seconds since most recent acoustic decision, capped at max_age_s

Important: evidence_scaled is a numerical score, NOT a calibrated probability.
It must not be presented as uncertainty or a localization confidence probability.
An invalid observation has zeros in the first four fields and may retain its
bounded age field. A never-observed stream returns five zeros.

Input contracts
---------------

PolicyHearingEncoder.ingest_hearing accepts the existing v0.14
HearingObservation and reads *only* its capture time and spatial directions.
It rejects ambiguous near-equal candidates and enforces coordinate frames.

PolicyHearingEncoder.ingest_loop accepts the v0.14 LoopObservation but updates
state only for new_acoustic_decision=True. This prevents retained telemetry
rows from resetting the observation age.

AcousticEvidence is a minimal explicit input type for other perception backends.
No field carries source coordinates or simulator ground truth.

Clock and reset requirements
----------------------------

All times are integer nanoseconds in the same clock domain, per environment.
Call reset() after every episode restart or simulated time rewind. Repeated,
backward or non-finite events are rejected. Silence/rejected decisions clear old
bearing state. Pausing without an acoustic decision does not refresh evidence.

MuJoCo
------

Use MuJoCoPolicyBridge(runtime, pose_adapter) with the existing HearingRuntime
and MuJoCoSceneAdapter. The host owns mj_step; call bridge.observe(time_ns)
after each physics update. This advances acoustic sampling and returns the
policy vector. A perception backend must be installed on HearingRuntime to
produce directions. Without one, the policy features correctly stay invalid.

Isaac Lab
---------

Attach a HearingObservationBank(num_envs) as env.onda_hearing_bank; update it
from the actual audio pipeline after physics updates and publish integer
nanosecond timestamps as env.onda_time_ns. Reference the callable
onda_hearing.simulators.isaac_lab_policy.acoustic_policy_observation from an
ObservationTermCfg(func=...). It returns (num_envs, 5) float32 Torch data on
env.device. Reset bank state for environment IDs whenever Isaac Lab resets them.

This adapter intentionally does not generate audio inside the observation term.
Native Isaac Lab environment integration, GPU batching, latency measurements,
realistic reverberation and policy training remain separate acceptance gates.

Validation
----------

Run `python -m pytest -q tests/test_v015_policy.py` in the core SDK environment.
Run the inherited v0.14 suite before merging. Execute native MuJoCo and Isaac
Sim v0.14 acceptance suites unchanged. Native Isaac Lab checks require an
installed compatible Isaac Lab release and are not claimed by these unit tests.
