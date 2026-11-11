# Onda v0.20 — Unified Geometry-Aware Acoustic Propagation

**Version:** Experimental standalone additive module, 2026-11-10. Requires existing Onda v0.17 renderer and installed v0.19 `physical.py`/`physical_adapters.py`. Does not replace v0.19, its tests, or v0.18 benchmarks. Designed against user-supplied exact v0.17/v0.19 ZIP source files; full remote Shadeform tree was not supplied to this audit. Native MuJoCo script has to be executed on the target server; not run in the local validation container.

## 1. Fundamental objective and architecture

For each source `s` and microphone `m` synthesize one coherent pressure waveform:

```text
source PCM / schedule
    |
    +-- v0.19 DIRECT: retarded delay, inverse-distance gain, ray-scene
    |   intersection, per-panel frequency-dependent thin-screen TL
    |              |
    |              +-----> sum once --> receiver-only deterministic noise --> PCM [frames, 4]
    |              |
    +-- v0.20 REFLECTION: finite image-source bounces (orders 1-2),
        segment visibility/TL, material absorption/scattering, fractional
        arrival sample, path-length gain, continuous delay/gain ramps
```

**Avoiding direct path double-counting:** The combined renderer uses the v0.19 direct renderer with `noise_std=0`; its own reflection synthesizer emits **only orders 1 and 2**, never a direct acoustic path. Receiver noise is added exactly once using the unchanged v0.17 deterministic sample-index noise. Onda's optional legacy full-room `rir_taps` is rejected rather than blindly added to a direct waveform.

For geometrically stationary specular surfaces the reflection path is calculated by the classic *image-source method*. Reflection order 1: mirror the real source across a planar surface; the intersection of source-image-to-mic line with the finite polygon is the bounce. Reflection order 2: construct two successive image sources and trace backward through the reflecting surfaces in reverse order. The renderer verifies all bounce points are inside authored triangles; invalid/outside or wrong-sided bounces contribute zero. The reflection delay is the total path length divided by sound speed; its spreading gain is `reference_distance / path_length` (with the renderer's near-field clamp). No artificial echo time is selected.

Visibility is tested for **each path leg**, including any intermediate wall or door barrier; its explicit transmission loss is part of that reflection path's filter. The reflector's specular intensity coefficient is estimated as `(1-energy_absorption)*(1-scattering)` and its *pressure amplitude* is the square root. The resulting band magnitude plus TL for any penetrated panels is converted to a causal approximate minimum-phase FIR via the existing v0.19 function. Coefficients are **never inferred from generic material names**.

The `ReflectionMaterial` API requires band centers and both absorption/scattering **energy fractions**, provenance, and optional reflection-pressure polarity; model parameters are supplied, not learned or validated. This matches the broad physical organization of the supplied Pyroomacoustics source, but does not reuse or vendor its C++ implementation.

## 2. What has been validated here

The supplied source archives were inspected: Onda v0.17 renderer, v0.18 acoustic semantic tests, v0.19 geometry renderer, user-provided Pyroomacoustics 0.10.1 source and Isaac Audio Sensors geometry, propagation and material modules.

- **18 passing focused module tests (plus one skipped optional native MuJoCo test), with 48 passing compatibility tests** run in a reconstructed local SDK, including analytic single- and double-bounce geometry, finite surface validation, reflection arrival after direct arrival, material-loss behavior, occluded reflections, chunk independence, receiver noise not multiplied, source schedule controls, and reset.
- Geometry cross-check against an actual installed **Pyroomacoustics 0.10.1** `ShoeBox` first-order image-source model: six wall image positions match within `2e-7 m`; analytic path distances match to machine precision. **This is not a full RIR-amplitude, late-field, or measurement validation.**
- Static four-mic demonstration successfully synthesizes direct + reflected PCM (2 s, 16 kHz), with expected ~2.915 ms direct and ~12.021 ms first-order reflection path times.
- Original v0.19 / v0.17 acoustic / v0.18 acoustic tests passed alongside the new module in the reconstructed environment (**66 passed total, 1 optional native test skipped**). A full current-branch regression has NOT been executed here.
- Native MuJoCo **motor-driven hinged-door integration is provided but not executed in this environment**, which lacks MuJoCo. Target machine must run the included native acceptance script.

## 3. How to run on Shadeform

```bash
cd /ephemeral/onda-nvidia/onda-v015
source /ephemeral/onda-nvidia/env_isaaclab/bin/activate
python ../onda_v020_unified_physics/INSTALL.py --dest . --check
python ../onda_v020_unified_physics/INSTALL.py --dest . --install
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_unified_physics_v020.py
PYTHONPATH="$PWD/src" python -m pytest -q
PYTHONPATH="$PWD/src" python tools/demo_v020_unified_pcm.py --out runs/v020_unified_static
python - <<'PY'
import importlib.util
print('MuJoCo installed:', importlib.util.find_spec('mujoco') is not None)
PY
PYTHONPATH="$PWD/src" python tools/validate_v020_mujoco_moving_door.py \
  --output runs/v020_mujoco_moving_door --order 1
```

Do not install any new dependency unless the optional MuJoCo native harness reports `No module named mujoco`. The Pyroomacoustics comparison script is also optional and can be run on the Mac environment where `pyroomacoustics==0.10.1` is already installed.

## 4. Native moving-door scenario details

`tools/validate_v020_mujoco_moving_door.py` constructs a native MuJoCo MJCF, containing:

- Two rooms separated by explicit native **MuJoCo box geometry** wall segments and lintel.
- One hinged door with a native hinge joint, mass, damping, actuator and time-varying actuator target. MuJoCo advances with 0.002 s physics steps.
- A fixed audio source in one room and native-geometry-based four-mic array in the other room. WAV source can be user-supplied `--wav telephone_16k_mono.wav`, or an explicitly synthetic 440/3000 Hz test tone is generated.
- Native door box world position and orientation are extracted from `data.geom_xpos`, `data.geom_xmat`, and `model.geom_size`, not from a hand-built door-angle attenuation switch.
- The acoustic scene is updated at every 256-sample (16 ms) audio boundary before rendering. Source and microphone world positions are obtained from native MuJoCo body poses.
- The unified/direct-only difference is saved as a **reflection-only waveform**. Receiver PCM is 16 kHz 4-channel IEEE float WAV, with NO clipping/normalization.
- Machine-readable metrics include hinge angle progression, frequency-band open-vs-closed level differences, wet RMS, path counts, scene revisions and simulated-seconds/wall-second rate.

The native demo's pass criteria check hinge movement and nonzero reflected energy. They do **not** claim acoustic ground-truth fidelity or safe real-time operation. Inspect WAVs and metrics before drawing acoustic conclusions. `--order 2` adds up to second-order reflected paths; it can cost substantially more CPU time.

## 5. Known limitations and scientific boundaries

1. **Not a wave-equation solver:** Image-source paths assume geometrical specular reflections. Low-frequency diffraction and bending around a corner/doorframe are not implemented. Multi-room NLOS sound will be underpredicted when these paths dominate.
2. **Material impedance and phase:** Loss curves are user supplied, frequency based and minimum phase approximated, but reflection phase, angle dependence, multilayer wall resonances and flexural transmission are not accurately modeled. 100% absorption / 0% transmission are illustrative idealizations. Differentiating acoustic reflection and transmission is crucial.
3. **Missing paths:** Only orders 1-2 of specular reflections are included. No stochastic late reverberation, proper room decay, volumetric propagation, door-seal leakage, mechanical structure-borne vibration, or around-door acoustic aperture scattering. This does not equal Isaac Audio Sensors' optional Steam Audio NLOS + diffuse system.
4. **Scene updates:** Door and reflector geometry are quasi-static at audio-block boundaries (typically 16 ms). Smooth delay/gain ramps reduce discontinuities but do not solve time-dependent moving-boundary wave scattering or exact Doppler effects. Updating geometry outside the audio timeline is undefined; scene and audio must share a simulation clock.
5. **Geometry quality:** Only **planar** reflection facets; mesh intersections use basic O(N) ray tests with AABB rejection. Real-world dynamic USD triangles, arbitrary holes, nonmanifold meshes and thousands of reflectors require a robust preprocessing and BVH acceleration stage.
6. **No calibrated physical accuracy percentage:** Example parameters explicitly say `SYNTHETIC_UNCALIBRATED`. True acoustic accuracy needs measured door open/closed broadband impulse responses, octave-band transfer functions, measured 4-mic geometry/phase, first reflection arrival times and RT60 across many source/mic positions.
7. **Runtime:** Current pure Python ray/path updates are intended for research/small-room testbeds; hard real-time or multi-env Isaac Lab requires profiling and likely native/C++ acceleration.

## 6. Validation roadmap toward high physical fidelity

- Capture measured room impulse responses at source/mic locations, first with no barrier, then with multiple known door angles, open/closed gaskets and materials. Register measurements with a precision excitation signal and sample clock.
- Fit **frequency-dependent transmission loss**, complex reflection/impedance and opening/leakage parameters against calibration recordings; evaluate on held-out positions and room arrangements.
- For each microphone verify: direct arrival delay (<1 sample in analytic free field), reflected-path timing (<1 ms against large planar surfaces), octave-band transfer curves (target 3 dB mean error on held-out measured conditions), and energy decay/RT60 (when diffuse model introduced). All are PROPOSED engineering targets, not achieved results.
- Add finite edge diffraction or validated approximation; connected-room aperture routing and probe-based NLOS. Implement diffuse late reverb (decoupled from specular direct path) with explicit energy accounting.
- Develop native intersection/BVH and geometry update cache; measure real-time factor on the exact 4090 host. Add moving-source and rotating-mic tests under synchronized robot pose data.
- Only after physics acceptance, feed the **same generated PCM** to `hearing.push_audio()` in the existing v0.17 native MuJoCo / Isaac Lab perception harness. Ground-truth material/path metadata must remain evaluator-only.

## 7. Dependency and provenance

This add-on is original glue + geometry implementation and does not copy either Pyroomacoustics C++ or Isaac Audio Sensors source. Optional comparison uses `pyroomacoustics==0.10.1` (MIT); simulator harness uses installed MuJoCo Python runtime. The original references were consulted for physics algorithm structure, not duplicated. No Steam Audio SDK is bundled. Keep license and native terms checks separate before packaging commercial releases.
