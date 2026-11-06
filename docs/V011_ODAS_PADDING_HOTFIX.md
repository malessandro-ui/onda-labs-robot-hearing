# ODAS offline hop-alignment hotfix

The Docker ODAS adapter now pads non-hop-aligned signed 16-bit interleaved PCM input with zero-valued complete microphone samples. It preserves the original audio prefix exactly and reports `original_samples`, `padded_samples`, and `padding_samples` in `DockerODASResult`. The `audio_sha256` hashes the **padded bytes actually supplied to ODAS**.

Frames whose nominal `sample_offset` is **greater than or equal to the original sample count** are excluded. ODAS's `timeStamp` is treated as a hop counter, not a verified acquisition-time clock. No correction for window latency or time offset is claimed. Frames whose analysis windows overlap the original tail may still contain padding; authoritative per-frame scoring requires calibrated ODAS frame centering and warm-up rules.

The existing zero-padding correction is benchmark-input framing only. It does not change the SDK's physical microphone simulation, ODAS geometry, or source localization. Running the same recording with ODAS remains an external acceptance gate.

## Tests

`python -m pytest -q`

Test-only fake subprocess runs verify exact byte preservation, multichannel padding, output filtering and a failure when only padding-time frames exist. Docker image execution was not available in this build environment.
