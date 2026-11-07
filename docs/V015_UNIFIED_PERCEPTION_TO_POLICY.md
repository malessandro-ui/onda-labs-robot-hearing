# Onda v0.15 — Unified Perception-to-Policy Foundation

## Scope

An additive release-development patch for the existing v0.15 policy layer.
It does NOT redesign the v0.14 hearing runtime or replace the validated
MuJoCo and Isaac Sim hear-and-turn experiments. The goal is a shared,
simulator-neutral interface from **genuine multi-channel microphone PCM** to
**policy-visible auditory features** and structured hearing observations.

The project remains on package version `0.14.0a0` until v0.15 acceptance is
complete; this is a development checkpoint.

## Architecture

1. Simulator adapter returns a `SceneSnapshot` with robot/microphone pose,
   source positions, and simulation nanosecond timestamp.
2. `HearingRuntime` uses geometry and source waveforms to render multi-channel
   PCM. Source positions are **privileged simulation inputs for rendering**.
3. `HearingWindowBuffer` assembles a bounded contiguous sample window and a
   synchronized array-pose trajectory. It does not pass source positions to
   the localizer.
4. `AcousticOnlySelector` applies motion-aware SRP and conservative acceptance
   rules to audio-derived directional hypotheses.
5. `WindowedHearingPipeline` constructs an `AcousticEvidence` event, retains
   or invalidates current evidence, and exports `PolicyFrame.features`:
   `('bearing_sin','bearing_cos','evidence_scaled','valid','age_s')`.
6. `PolicyFrame.observation` is a `HearingObservation` only on new acoustic
   decision steps. Accepted directions are model inferences, **not scene
   source coordinates**. Rejected decisions have no directions.
7. The user-supplied robot policy chooses actions, reward, safety behavior,
   and task objectives. The SDK does not prescribe them.

## Entry points

- `onda_hearing.policy.WindowedHearingPipeline`: reusable core independent of
  MuJoCo / Isaac Sim / Gazebo Python imports.
- `onda_hearing.simulators.mujoco_policy.WindowedMuJoCoPolicyBridge`: owns the
  pipeline and extracts scene snapshots after a caller performs `mj_step`.
- `onda_hearing.policy.AuditoryObservation`: typed expandable representation,
  separate from the existing `HearingObservation` v1.0 contract.
- `onda_hearing.policy.SemanticEvent`, `SemanticResult`, `SemanticPolicyEncoder`:
  typed semantic outputs and explicit `UNSUPPORTED`, `NO_DETECTION`,
  `DETECTED` states; these **do not implement sound classification**.
- `onda_hearing.policy.SourcePolicyEncoder`: deterministic padded `(K,7)`
  track tensor, with masks and expiration. **No actual tracker is added**.

## Guarantees and caveats

- All clocks for the windowed pipeline use one configured simulation domain;
  rewinds and duplicate sensor timestamps fail closed until reset.
- A negative acoustic decision clears older directional evidence; a missing
  decision ages its previous detection until stale.
- The first scene snapshot establishes the renderer epoch, and does not
  generate samples.
- The original five-element spatial policy features remain unchanged.
- Scores are **not calibrated probabilities**. Features are normalized by the
  caller-configurable scale and clipped, which does not imply calibration.
- A semantic class label must come from **inference**, not scheduled source
  identity or the scenario ground truth.
- `SemanticResult(UNSUPPORTED)` is distinct from a running backend's
  `SemanticResult(NO_DETECTION, backend_id='...')`.
- `SemanticEvent.last_observed_ns` is the last supporting measurement.
  `offset_ns` is optional and only populated when an inference backend
  supports an offset estimate; it must not be fabricated from source schedules.
- The source policy tensor is an optional snapshot, **not a stable identity
  slot assignment**. Obtain persistent track IDs from structured observations.
- Current physics rendering uses direct-path acoustics; this work does not
  validate realistic reverberation, multi-source classification, hardware,
  Isaac Lab vectorized execution, or physical robot behavior.

## Install without modifying v0.14 release

On the existing `develop/v0.15-unified-hearing` Git branch, preview then
rsync the patch from its staging folder. Verify `git diff` and `git status`
prior to running tests. Do not use on the v0.14 `main` branch.

## Test commands

```bash
python -m pytest -q tests/test_v015_policy.py tests/test_v015_unified_perception.py
python -m pytest -q
python examples/v015_mujoco_policy_observations.py --source-bearing 38 --strict --output-dir benchmark_results/v015_policy_plus38
python examples/v015_mujoco_policy_observations.py --source-bearing -38 --strict --output-dir benchmark_results/v015_policy_minus38
python examples/v015_mujoco_policy_observations.py --mode silence --strict --output-dir benchmark_results/v015_policy_silence
```

The native MuJoCo acceptance runs must be executed on a machine with MuJoCo
installed; simulator-free tests do not substitute for them. A native Isaac Lab
policy integration, multiple simultaneous sources, semantic classifier backend,
real robot sensor transport, and a Gazebo bridge are future acceptance gates.
