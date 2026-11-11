# Onda Robot Hearing — Acoustic Physics Audit and v0.19 Upgrade

**Scope/date:** 2026-11-10. Source-backed assessment of the files furnished to this review: Isaac Audio Sensors source bundle (420 text files), Pyroomacoustics v0.10.1 source bundle (323 text files), Onda v0.17 acoustic-semantic drop-in, v0.17 native integration drop-in, and v0.18 paired benchmark drop-in. The **full live Shadeform working tree was not uploaded**. Actual v0.18 headroom hotfix and any later edits must be inspected and regression-tested on Shadeform before merging. Therefore this is a tested additive implementation against a reconstructed v0.17 source base, not a certification of the remote checkout.

## 1. Finding: current Onda v0.17 direct physics

Source: `onda_v017_dropin/src/onda_hearing/acoustic_scene/{renderer.py,types.py,adapters.py,assets.py}`.

**Genuinely implemented:**

- Per-source PCM scheduling/triggers, loop and stop with retarded acoustic arrival.
- 4+ channel array world placement, 3D orientation (`XY­ZW`) and pose-timeline interpolation.
- Sound-speed delay and inverse distance pressure attenuation, configurable source cardioid.
- Approximate moving-source retarded emission time through three iterations.
- Deterministic sample-index noise, optional stationary causal FIR per microphone.
- Sequential chunk checks, overrange and clip telemetry.

**Not currently scene aware:**

- No ray/triangle intersection. Audio cannot infer that a closed door or wall blocks source–receiver line of sight.
- No **material transmission loss** or independent reflectivity/scattering information.
- `rir_taps` are one post-mixture [taps, channels] FIR: they are not derived from scene geometry or evaluated separately for each emitter–microphone pair.
- No reflection-path generation, edge diffraction, gaps, doors, USD mesh extraction, late diffuse field, or measured microphone frequency-response calibration.
- Scalar white noise does not reproduce motor or fan ego noise spectrum.

Existing v0.18 benchmark (`tools/benchmark_acoustic_semantics_v018.py`) correctly offers paired clean vs one-meter rendered conditions and measured synthetic SNR. It evaluates semantic robustness; it **does not validate physical scene sound transmission**.

## 2. What Isaac Audio Sensors actually adds

Analyzed source paths relative to `isaac_audio_sensors/`:

| Module | Concrete role |
|---|---|
| `src/isaac_audio_sensors/isaac/acoustic_scene/session.py` | USD scene preparation, acoustic representation selection, material association, geometry/pose revisions |
| `.../geometry.py` | Polygon triangulation and scene mesh geometry |
| `.../materials.py` + `core/acoustics/materials.py` | Authored frequency-band absorption / scattering / transmission loss with provenance |
| `.../_steam_audio.py` / `steam.py` | Native Steam Audio scene and direct/transmission integration |
| `.../_specular.py` | Native Pyroomacoustics-derived image-source/reflection interface |
| `.../_nlos.py` | Optional Steam probe routing for selected non-line-of-sight paths; not full wave diffraction |
| `.../_diffuse.py` / `_pra.py` | Experimental diffuse transport and pressure synthesis |
| `.../_convolution.py` | Bounded streaming convolution with impulse response crossfades |
| `.../propagation.py` | Backend composition, per-receiver source-to-mic processing, bounded runtime configuration |

The sophistication is structural: separate direct path, scene geometry, transmission bands, specular paths and diffuse options. Some branches require native Steam Audio and native Pyroomacoustics libraries. Copying the pure-Python classes alone will not yield equal physics. Their validation documentation recognizes uncalibrated material uncertainty and unresolved moving-room equivalence.

## 3. What Pyroomacoustics 0.10.1 actually solves

Analyzed source paths relative to `pyroomacoustics/`:

- `pyroomacoustics/room.py`: `Room`, `ShoeBox`, image-source model and optional ray-tracing simulation, microphone RIRs. Shoebox is a static rectangular room abstraction, not a dynamic two-room door solver.
- `pyroomacoustics/libroom_src/room.cpp`: native 2D/3D image-source recursion / shoebox image sources and ray-tracing internals.
- `pyroomacoustics/libroom_src/rir_builder.cpp`: impulse builder for discrete-time fractional arrival times.
- `pyroomacoustics/libroom_src/wall.cpp`: wall-level absorption and scattering propagation.
- `pyroomacoustics/simulation/ism.py`: image-source impulse response synthesis and convolution utilities.

Image-source methods are physically motivated geometric acoustics, especially useful for specular early reflections from large, rigid surfaces. They are not exact diffraction solvers; room and material representation remain critical. Generic `Material(energy_absorption=...)` controls reflection energy; **absorption cannot substitute for door transmission loss**.

## 4. v0.19 drop-in design

### A. `GeometryAwareAcousticRenderer` (tested)

- Uses the **unchanged Onda v0.17** renderer once per emitter, retaining its source scheduling, distance spreading, sample timing, movement and quaternion geometry.
- Computes source-to-receiver segment intersection with explicit world-space triangles and an AABB broad phase, one hit per physical panel.
- Sums explicit, frequency-band transmission-loss (TL) curves in decibels for intersected panels; no visual-name material hallucination.
- Designs causal minimum-phase approximate FIRs from magnitude-only TL; uses one filter and history per emitter–microphone path, plus fixed-size crossfades when an object moves/changes occlusion state.
- Rechecks physics geometry on an absolute sample grid or when scene revision increments; accepts door hinge updates between render calls.
- Adds receiver noise only once after all sources. Maintains `render`, `reset`, `trigger`, `stop`, `set_gain_db`, and `telemetry` methods.
- Preserves v0.17 default `AcousticSceneRenderer` completely untouched.

**Important approximation:** transmission through an idealized *thin screen*, rather than combined door-panel structure, seam leakage, beam tracing, edge diffraction or impedance matched physical panel. Two separate objects may compound TL; two geometric faces representing the same physical barrier should be one `AcousticSurface` to avoid counting the same wall twice.

### B. `physical_adapters.py` (source-backed contracts)

- Reads explicit triangulated/convex planar USD meshes via USD `XformCache`, resolves stage scale into meters.
- Converts MuJoCo **box geoms** into world-space meshes using `geom_xpos`, `geom_xmat`, `geom_size`.
- Nonplanar, concave, deformed or hole-containing mesh faces need a separate robust geometric preprocessing/triangulation stage and are rejected instead of silently corrupted.
- The simulator must call `geometry.put(updated_surface)` after its pose/state update and before the next audio render call; concurrent mutation during audio render is unsupported.

### C. `room_physics.py` (tested separate static-room backend)

- Uses actual Pyroomacoustics 0.10.1 APIs to generate shoebox room impulse responses and convolve source PCM; retains direct + reflection tails without clipping or normalization.
- Requires explicitly declared absorption, room dimensions, source and microphone positions. Does **not** assume wall names imply calibrated coefficients.
- This is **not combined** with A by default. Direct-path Onda and Pyroomacoustics full RIR each contain direct sound, so naive summation would double count it. A unified hybrid path needs explicit direct/reflection decomposition and consistent amplitude/time calibration.

### D. License and provenance

No Isaac Audio Sensors / Pyroomacoustics code was copied or vendored. Original implementation is separately distributed and calls Pyroomacoustics as an optional installed dependency. Confirm dependency licenses, Steam Audio native SDK terms, and deployment targets before adding native integration.

## 5. Door acceptance demonstration

The included demo produces a 3-second, four-microphone render using an explicitly authored test wall with a swinging door. At 0 degrees the door intersects the direct path. At 90 degrees it swings out of that ray. The test signal contains 440 Hz and 3 kHz components.

Synthetic reference material (not measured):

| Center Hz | Door TL dB |
|---:|---:|
| 125 | 6 |
| 500 | 10 |
| 1000 | 16 |
| 4000 | 29 |
| 8000 | 35 |

**Measured from the generated demo**: open-vs-closed differential approximately 9.35 dB at 440 Hz and 24.67 dB at 3 kHz on microphone 0. The exact values depend on causal FIR approximation and finite-length analysis; they do not represent a known physical door.

Three generated files: `door_open_4ch.wav`, `door_closed_4ch.wav`, `door_open_closed_open_4ch.wav`. Closing/opening fades smoothly rather than instantly switching coefficients.

## 6. Verification run

Reconstructed local sources from the user-provided v0.17 acoustic semantic drop-in. Ran `PYTHONPATH=... python -m pytest` with new physics tests and previously provided v0.17 scene and v0.18 paired-acoustics tests:

**48/48 passed**: 16 v0.19 physics tests and 32 existing v0.17/v0.18 tests. Also ran the actual demonstrator to produce the WAVs and differential spectral metrics above. Both parts are reproducible from the delivered files. These do not equal a full remote branch regression or Isaac Sim native test.

## 7. Remaining work to approach Isaac Audio Sensors' geometry backend

**Stage 1 — this drop-in:** source-specific direct-path occlusion, explicit TL, open/closed moving door, causal transitions, static Pyroomacoustics shoebox RIR in separate mode. Good for observable *illustrative* doorway effects, not full fidelity.

**Stage 2 — validated geometry:** CAD/USD mesh triangulation with holes and concavities; BVH acceleration for thousands of meshes; derived world-mesh updates for articulated robots and active USD scene graphs; accurate unit and material mapping; full robot microphone frequency-response calibration.

**Stage 3 — unified reflection model:** consistent geometry-native image-source reflection paths and material reflection coefficients, full per-source/per-mic transfer functions, use Pyroomacoustics RIRs for supported simple rooms, and obtain/qualify native Steam Audio when needed. A merged direct and reflected response must avoid double counting.

**Stage 4 — non-line-of-sight:** valid indirect paths through open doors and corridors, frequency-dependent edge diffraction (or declared approximation), leaks around seals, probe sensitivity and multi-room acoustic connectivity.

**Stage 5 — physical validation:** record precisely aligned source/array impulses in a measured room, with open/closed door at several angles, measure octave-band ΔTL, impulse delay, direct/reverb ratio, RT60 and spatial errors. Compare sim to physical measurements and publish error bars. Without these data, no percent-accuracy or 'physically exact' claim is justified.

**Stage 6 — scaled realtime:** sample-accurate geometry changes, bounded native CPU/GPU resources, chunk invariance under pose updates, multi-source per-robot 4ch throughput, field perturbations (ego noise, fan/motor spatial sources), and Isaac Lab multi-env stress tests.

## 8. Suggested physical test pass criteria (engineering targets, not achieved)

- Direct-path onset error < 1 sample for static free-field geometry against analytical prediction, with known resampling and calibration tolerances.
- Band-averaged closed/open transmission error target < 3 dB in octave bands **on held-out measured door recordings** after fitting once to a separate calibration dataset. This is an illustrative goal, not a validated guarantee.
- Early reflection arrival-time error target < 1 ms for large planar surfaces.
- No audible crackles when an articulated door moves, with time-aligned waveforms and bounded click energy.
- Zero unapproved clipping and consistent frequency-dependent material metadata provenance.
- All existing v0.17, v0.18, native Isaac, MuJoCo and motion regression suites green on the target SHA.

## 9. Known scientific limits

The inverse-distance amplitude model assumes approximately point-source free-field pressure propagation. The new frequency-dependent FIR realizes a selected attenuation magnitude but does not model the actual door's complex transmission phase. When direct paths are blocked, this backend has no edge-diffracted energy, leak geometry, multiple-bounce paths or whole-room reverberation. Pyroomacoustics RIRs provide useful static room echoes but do not alone replace Steam Audio/NLOS or general dynamic USD propagation. The resulting synthetic waveforms are physically **motivated**, not proven physically **accurate**.
