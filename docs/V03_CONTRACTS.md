# v0.3.0-alpha — Public contracts

This release adds the **new** import namespace `onda_hearing.contracts` and keeps
`onda_hearing.AudioBlock`, `onda_hearing.MicrophoneArray`, `HearingPipeline` and the
old examples working as before. It does **not** replace the older pipeline yet.

## Contract guarantees

- Audio PCM: normalized float32 `[frames, channels]`, copied and read-only.
- Sensor geometry: arbitrary 1–256 channels; explicit channel indices, IDs,
  geometry version, frame identifiers and SI units (meters).
- Capture: source-clock IDs and nanosecond capture intervals, validated against
  sample counts. Different clocks are never implicitly synchronized.
- Directions: radians, +X to +Y in the array XY plane; raw spatial `score`
  is **not** represented as calibrated probability.
- Tracks: distinct from directional detections and from separated waveforms.
- Backend capabilities: declared channel/rate/rank/source-count limitations;
  `run_checked` validates them before calling the backend.
- Observation: strictly sensor-derived fields; privileged simulator truth has
  no field in `HearingObservation`.

## Compatibility and migration

Use `from_legacy_block(old_block, new_array, clock_id='simulation')` to convert
legacy renderer output to a modern `AudioPacket`. The caller must supply the
correct geometry identity and capture clock. This adapter does not reinterpret
coordinate frames or perform microphone calibration.

## Known boundaries

The contract types are stable **candidates**, not a frozen 1.0 public ABI.
`SpatialBackend` is only a protocol; existing legacy SRP-PHAT code is **not**
claimed to implement it. No ODAS, BEATs, source separation, physical capture,
real-time scheduling, Isaac or MuJoCo runtime qualification is added here.
Single-source baseline output remains conservative, not a proven multisource
solution. Channel and sample-rate restrictions belong to each backend.

Future work: per-channel calibration, typed semantic events, covariance-rich 3D
tracking, clock-offset estimation, serialization schema, versioned compatibility
and cross-device timestamp tests.
