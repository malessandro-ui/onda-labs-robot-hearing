# Onda v0.15 — Isaac Lab policy integration (CPU reference backend)

This is an **additive** adapter for the validated Onda v0.15 `WindowedHearingPipeline`. It does not change the acoustic core, existing MuJoCo examples, scene contracts, rewards, training algorithms, or the `main` v0.14 release.

## What is implemented

- Independent `WindowedHearingPipeline` per environment, with audio generation, motion-aware SRP, acoustic-only candidate selection, timestamps, policy encoder, and reset state.
- Batched conversion of robot and sound-source **world** positions from Isaac Lab asset tensors, converting Isaac scalar-first **WXYZ** quaternions to Onda **XYZW**.
- Per-environment **episode-local** nanosecond clocks and partial resets; reset environments never inherit old acoustic observations.
- Fixed-shape read-only 5-float features `(N, 5)` in the exact order `('bearing_sin','bearing_cos','evidence_scaled','valid','age_s')`.
- Optional read-only `ObservationTermCfg` callable, and reset `EventTermCfg` callback for applications with explicit post-physics integration.
- A simple `OndaIsaacLabEnvWrapper` for existing vectorized tasks that adds a separate `onda_hearing` observation group after `env.step`, without controlling the robot, changing the task reward, or changing the original policy tensor.
- CPU reference backend cap (default 16 environments). **This is not a GPU-parallel acoustic engine**: PCM rendering, SRP, and staging batched poses from GPU to CPU introduce host overhead. Benchmark before scaling; GPU acoustic pipeline is future work.

**Not yet validated:** native Isaac Lab / Isaac Sim execution, GPU-throughput scaling, ROS 2, arbitrary body geometry alignment, multi-source tracking, semantic inference, or robot learning effectiveness. The tests bundled with this patch use real Onda PCM and localization in simulator-independent tests plus a mocked Isaac Lab environment lifecycle; they do not import or run Kit.

## Installation

Apply on `develop/v0.15-unified-hearing`, **after commit `2941deb`**. The patch adds two source files, updates one existing adapter module, adds a test and a reference example. It does not update `pyproject.toml` version until native acceptance is achieved.

```bash
cd ~/Developer/onda-labs-robot-hearing
source .venv-v015/bin/activate
python -m pytest -q tests/test_v015_isaac_lab.py \
  tests/test_v015_policy.py tests/test_v015_unified_perception.py
python -m pytest -q
```

For native validation on your NVIDIA Isaac Sim / Isaac Lab machine, use the Python environment bundled/configured for **that installed Isaac Lab release**; do not install Isaac Sim into the Mac venv. Check that `isaaclab` and `torch` import inside the launched Isaac Sim app. This code intentionally avoids importing Isaac Lab at package import time, so existing MuJoCo users need no NVIDIA dependencies.

## Path A: wrap an existing RL task (simplest initial smoke test)

Use the provided `examples/v015_isaac_lab_task_integration.py` **from your existing Isaac Lab application after the SimulationApp is launched and your task is constructed**. Your task's scene must have a rigid/articulated array-bearing asset and a real source object asset, and expose `root_pos_w` / `root_quat_w` or, for articulated head, `body_pos_w` / `body_quat_w`.

```python
from v015_isaac_lab_task_integration import attach_onda

# Example uses an already constructed Isaac Lab task, NOT a task Onda creates.
# env.scene['robot'] and env.scene['phone'] must actually exist.
wrapped = attach_onda(
    env,
    array_asset='robot',
    source_asset='phone',
    source_audio=source_pcm_mono_16khz,
    array_body_index=None,  # use actual body index when array is on head
)
obs, info = wrapped.reset()
obs, reward, terminated, truncated, extras = wrapped.step(actions)
audio_features = obs['onda_hearing']  # torch.float32 [num_envs,5]
# Your policy decides how to combine this with obs['policy'].
```

The example assumes the sound source is continuously scheduled from episode t=0. Your task should add other scenario design and source-event scheduling. **No source ID, class label or location is passed to the policy**. The source world positions are used solely inside the renderer. The policy gets only inferred acoustic features.

**Resets:** Isaac Lab vectorized RL tasks ordinarily auto-reset done environments inside `env.step`. The wrapper resets their acoustic pipelines and seeds the **post-reset** world poses; non-reset environments continue their own local clocks. If your task does **not** auto-reset on done, modify the wrapper integration to reset/seed at the point where physics actually resets.

**Timing:** wrap after RL task construction. `step_dt` is the *policy-step* duration. If you want microphone poses sampled at physics cadence, use Path B and call `advance` from a dedicated post-physics callback with a matching acoustic time increment. Do not call both paths together.

## Path B: native ObservationManager term with explicit lifecycle hook

Use this when integrating into a subclass or custom task that has a reliable *post-physics, pre-observation* hook. The audio step must be driven by that hook, not by `ObservationTermCfg` (observation terms can be re-evaluated for logging, resets and multiple groups).

```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from onda_hearing.simulators.isaac_lab_runtime import (
    onda_acoustic_policy_observation, onda_reset_hearing,
)

# Add to your task's observation config:
hearing = ObsTerm(func=onda_acoustic_policy_observation)
# Term returns torch.float32 [num_envs,5] and *only reads cached features*.
# In your task's appropriate post-physics hook BEFORE the manager computes obs:
positions, quaternions, sources = pose_reader.read(env)
manager.advance(positions, quaternions, sources, step_token=current_step_id)
# In your custom task reset hook AFTER physics/scene reset:
manager.reset(reset_env_ids)
manager.seed(*pose_reader.read(env), env_ids=reset_env_ids)
```

You can register `onda_reset_hearing` as an Isaac Lab `EventTermCfg(func=onda_reset_hearing, mode='reset')` only if `env.onda_hearing_manager` is already attached before the event fires. Be careful with order: reset notifications may precede physical reset writes. If they do, call `seed` only after the new robot/source poses exist. The wrapper (Path A) avoids this issue for typical auto-reset RL tasks.

**Do not** use the read-only observation term as an automatic audio updater. Doing so produces duplicate PCM and clock drift when Isaac Lab queries observations multiple times.

## Frames and units

- Positions: world XYZ, **meters**. Both microphone mount and source coordinates must be in the SAME world frame (Isaac Lab `root_pos_w`, not environment-relative local coordinates).
- Source labels: IDs for *renderer bookkeeping*, never policy semantic labels.
- Rotation: Isaac Lab asset poses use quaternion **WXYZ**; Onda `SceneSnapshot` expects **XYZW**.
- Microphone geometry: in array-local meters. In `attach_onda` the supplied four-microphone geometry is an example. Replace with the physical microphone layout and calibrated array frame before claiming physical fidelity.
- For an array mounted on an articulated head, use the head `body_index`; root robot pose is not the same as head pose.
- `array_body_index` points to an index in Isaac Lab `body_pos_w`, NOT a USD prim index.
- Any rotation/translation between a robot body's pose and actual microphone-array frame must be applied before passing poses to Onda. The initial helper assumes coincidence, by design rather than silently inventing calibration.

## Performance and validation gates

1. Local Python: `pytest -q tests/test_v015_isaac_lab.py` (quaternions, time, partial resets, CPU audio inference, wrapper behavior).
2. Native Isaac Lab: import, task creation and wrapped `.reset()` produce a separate `[num_envs,5]` tensor on the task device.
3. Native sound: with a stationary broadband emitter at +38° and -38° relative to a stationary array, validate recognized directions within 10°. Use **acoustic** detections only; ground truth belongs to the evaluation path.
4. Native silence: zero audio must produce an invalid observation (`valid==0`) and no inferred direction.
5. Native async resets: when one of two envs resets, its cache must immediately invalidate, other environment's clock/history must continue.
6. Performance profiling: measure wall time per environment and per observation after warmup; CPU pipeline and CPU/GPU transfers are expected bottlenecks. Avoid claims about RL throughput until profiling proves feasibility.
7. No action coupling: Onda's wrapper does not issue actuator commands or alter actions/rewards.

## Supported APIs and limitations

This patch targets the documented Isaac Lab ManagerBased/DirectRLEnv interfaces (`num_envs`, `device`, `step_dt`, `scene`, `step`, `reset`). Isaac Lab versions differ in lifecycle ordering, task registration, and optional manager APIs. Test the installed version before declaring native support. The adapter accepts GPU tensors but stages poses on CPU. The `OndaIsaacLabEnvWrapper` is intentionally a plain Python wrapper, not a Gymnasium subclass; if downstream wrappers demand gym's exact types, integrate the `IsaacLabHearingManager` directly instead.

**Do not merge to main or bump the version yet.** First pass native Isaac Lab 5.1-compatible acceptance tests and preserve the results in `docs/validation/`.
