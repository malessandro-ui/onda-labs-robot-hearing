# Onda v0.15 — Native Isaac Lab motion validation

**Development branch only:** `develop/v0.15-unified-hearing` (tested baseline: `7e42d78`). Do not replace `main` or claim release support without independent review.

## Purpose

Checks whether Onda's four-channel synthetic acoustic inference follows a **rotating virtual microphone array in a real Isaac Lab / PhysX scene** while a sound-source rigid body remains stationary in *world* coordinates. Unlike the stationary Cartpole test, this exercise changes the array object's real simulated WXYZ orientation and samples the resulting scene data using Onda's existing `IsaacLabSceneReader`. There are no privileged bearings in Onda's policy features.

This is **not yet an articulated robotic head, actuator-driven yaw, or physical microphone calibration test**. The array is a dedicated non-colliding kinematic rigid object, and poses are commanded through Isaac Lab's `write_root_pose_to_sim`. It is a software-level native motion integration gate.

## What is installed

- `tools/native_isaaclab_motion/native_motion_acceptance.py`: the finite-run native test
- `tools/native_isaaclab_motion/rotation_math.py`: independent evaluation-only WXYZ math
- `tools/native_isaaclab_motion/run_motion_acceptance.sh`: bounded, reproducible launcher
- `tests/test_native_motion_math.py`: CPU-only quaternion regression

No PyTorch, NVIDIA, Isaac Sim, Isaac Lab or Onda SDK dependency upgrades are performed. No existing source files are overwritten.

## Test method

Two independent native `Isaac-Cartpole-v0` environments are constructed with a 4 cm virtual microphone mount and a distinct fixed acoustic-emitter rigid body. Broadband 300–3800 Hz mono PCM is rendered by Onda's existing CPU acoustic system into its configured four-channel microphone geometry. The same microphone array and SRP setup from `examples/v015_isaac_lab_task_integration.py` is used, with its own history and clocks per environment.

The camera/policy is not given the emitter coordinate. Ground truth is computed **only inside the assertion code** using world-space positions and the measured native mic-rig quaternion.

Per-environment world-source angles start at +38° and −38°. Both microphone rigs go through four phases: yaw `0°, +30°, −30°, 0°`. Every change after the initial phase is gradually applied over 25 native policy steps, preserving the acoustic pipeline state. Each phase has 90 further policy steps; only the final 35 steady-state steps are scored to allow the estimator to settle. Source drift <= 1 mm and native rig rotation error <= 1.5° are checked.

Acceptance assertions:

- Native CUDA environment with 2 environments and unchanged policy observation shape.
- `onda_hearing` tensor `[2,5]`, `torch.float32`, finite and CUDA-backed throughout.
- At least 12 valid acoustic detections per environment per phase within the tail window.
- P75 circular angular error <= 10° per environment per phase.
- Inferred bearing must change by >=15° from initial 0° yaw to +30° yaw, then return within 10° of baseline on returning to 0°.
- The emitter stays fixed in native world coordinates; the rig **actually rotates** according to native quaternion data.
- No unexpected Cartpole auto reset; a reset invalidates continuity and fails this test.
- GPU task/SDK action and reward code remains unmodified.
- Separate timing of entire policy step (PhysX + Onda + Python) and `IsaacLabHearingManager.advance` (CPU render + SRP + GPU->CPU staging). Performance is descriptive, not a pass/fail gate.

## Run (Shadeform)

From your Mac, copy the ZIP to the VM using `scp -i ~/Downloads/YOUR_SSH_KEY.pem ... shadeform@YOUR_SHADEFORM_IP:/ephemeral/onda-nvidia/`; the public IP may change.

On the VM, extract it with Python:

```bash
cd /ephemeral/onda-nvidia
python -m zipfile -e Onda_v015_Native_Motion_DropIn.zip .
cd /ephemeral/onda-nvidia/onda-v015
cp -v /ephemeral/onda-nvidia/onda_v015_motion_dropin/tools/native_isaaclab_motion/* tools/native_isaaclab_motion/
cp -v /ephemeral/onda-nvidia/onda_v015_motion_dropin/tests/test_native_motion_math.py tests/
cp -v /ephemeral/onda-nvidia/onda_v015_motion_dropin/docs/V015_NATIVE_MOTION_VALIDATION.md docs/
bash tools/native_isaaclab_motion/run_motion_acceptance.sh
```

**Before copying** ensure the target `tools/native_isaaclab_motion` directory exists (see the commands in the chat). No existing SDK code is replaced.

## Outputs

Generated in the development checkout: `docs/validation/native_isaaclab_motion_v015/results.json` and `motion.log`. The JSON report is atomically written and includes the pass/fail conclusion, measured world poses, all phase errors and timings, and exception traceback on failure. The shell script removes the old JSON before running to prevent stale PASS results.

### Shutdown safety

On this RTX 4090 headless machine, `SimulationApp.close()` has blocked even after physics test success. The runner uses `os._exit()` only **after fsync + atomic results save**, inside a bounded one-off process, and the shell imposes a 240-second limit. **Never use hard-exit in a persistent robot SDK, RL trainer or user process.** Clean Kit shutdown remains a separate issue.

### If the run fails

Inspect the traceback or phase error in `docs/validation/native_isaaclab_motion_v015/motion.log`. Do not weaken the acceptance tolerance or fabricate measurements. Check source and rig native poses, reset behavior, and microphone geometry. If it times out, inspect `ps` for orphaned processes before retrying.

## Git workflow

Keep on `develop/v0.15-unified-hearing` until the native test passes. Commit the runner, math tests, documentation, and **results.json** with a meaningful message. Check the `git diff --cached` to avoid accidentally committing large generated logs. `main` remains the public v0.14 version, and a PR to promote v0.15 should follow independent testing and a version/docs review.
