# Onda Robot Hearing v0.4: Acoustic physics reference

## Status and scope
This is a **research validation engine**, not a calibrated physical room simulator or a certified real-time renderer. It is a backwards-compatible package addition. The v0.3 public data contracts and legacy renderer remain available unchanged; no claims are made about the legacy simulator's acoustic fidelity.

## Public API
`onda_hearing.physics`:
- `AcousticConfig`: sound speed (m/s), sample rate (Hz), minimum distance clamp (m), interpolation support, size limits.
- `PoseTrajectory`: time-indexed translations in meters and proper SO(3) rotations. Positions interpolate linearly and rotations interpolate on SO(3); outside recorded range, end poses are held. This **is not a dynamics engine**.
- `PhysicalSource`: source ID, mono audio sampled at the renderer's sample rate, scheduled start time, world pose trajectory, amplitude.
- `DirectPathRenderer.render`: returns frame-major float32 multichannel PCM `[frames,mics]`; sources overlap by numerical summation with no clipping. Receiver position for each sample is taken at arrival time; source trajectory at iteratively solved emission time.
- `convolve_rir`: RIR convolution of dry source signals `[samples,sources]` with measured/synthesized impulse responses `[taps,sources,mics]` into `[samples+taps-1,mics]`. Includes room and direct paths; do **not** cascade it after direct-path simulation unless deliberately constructing a separately modeled contribution.
- `sample_acoustic_profile`: reproducible draws of RT60, noise standard deviation, and static per-channel gains within specified bounds. **Sampling parameters does not synthesize an RIR, and RT60 from this profile is not currently used by DirectPathRenderer.**

## Physics
Free field: `x_m(t) = Σ_s gain_s * (r_ref / max(r(t), r_min)) * source_s(t_emit)` where `t_emit = t_arrival - ||receiver_m(t_arrival)-source_s(t_emit)||/c`.

- `r_min` clamps singular amplitude inside an unphysical near-field proximity. Validate distances >= `r_min` for quantitative amplitude tests.
- Windowed-sinc fractional interpolation assumes reasonably bandlimited source audio. Tests cover a known sinusoid and TDOA, not every spectrum or signal length.
- Source motion changes arrival-time mapping and thus includes simplified geometric Doppler due to relative travel delay. No head-related transfer function, scattering, obstruction, multi-path, absorption, directional microphone responses, sensor clocks, or structural noise.
- Trajectories must be sufficiently smooth/subsonic for the retarded-time iteration to converge. Nonconvergence raises an error. Instant jumps and near-field singular geometries are not physically accurate.
- For sources and receivers in motion, validation presently tests continuity and finite output, not comparison with motion-acoustics instrumentation.
- Acoustic profile fields are sampled once per profile, representing episode-level parameters. No dynamically resampled independent gain per sample.

## Validation suite
10 additional tests in `tests/test_physics_v04.py` cover:
1. TDOA from independently computed source/microphone geometry.
2. 1/r energy amplitude using a steady sine at two distances.
3. Linear addition of simultaneously scheduled sources.
4. Windowed-sinc delay phase against analytic sinusoid.
5. Array motion temporal continuity.
6. Moving source delay convergence.
7. Known finite impulse-response convolution and multi-source summation.
8. Fixed-seed profile reproducibility and bounds.
9. Invalid inputs fail explicitly.
10. v0.3 legacy acoustic API remains operational.

Current tests are numerical component tests, **not** laboratory validation of real-world acoustic physics. Before making real-world claims, compare to measured multichannel recordings/RIRs, test audio hardware calibration, and benchmark processing latency.

## Next gates
1. Integrate `contracts.AudioPacket` boundary (requires explicit full-scale gain handling and clipping policy). Float PCM from the physics renderer is not automatically a normalized ADC capture.
2. Add chunked / real-time streaming and stateful RIR overlap-save with continuity and latency tests. Current reference rendering and RIR convolution are offline.
3. Validate generated signals against independent pyroomacoustics reference and real measured RIR datasets.
4. Add physically constrained materials and sensor-response randomization calibrated on real measured distributions.
5. Introduce simulator adapters only after time/pose data interfaces are validated.
