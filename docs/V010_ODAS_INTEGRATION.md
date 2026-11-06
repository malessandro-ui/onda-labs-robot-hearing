# Onda Robot Hearing v0.10 — optional ODAS integration

## Design and evidence scope

This release provides a **real, optional executable adapter**, not a reimplementation of ODAS or a mock. No ODAS process is available in CI. Without a separately installed `odascore`, a calibrated `.cfg`, and a binding manifest, **no ODAS benchmark scores are reported**. The Onda SRP and experimental covariance MUSIC baselines remain independently usable.

ODAS is maintained by [IntRoLab](https://github.com/introlab/odas) under the MIT license. ODAS's C source and executable are **not bundled** in the SDK. The Python SDK's license has not been selected by Onda Labs; adoption of ODAS does not determine that license.

## ODAS offline contract (strict)

The backend uses ODAS's documented `raw.interface.type="file"` and `ssl.potential` file sink with `format="json"`. The preexisting ODAS configuration must be **manually audited** and provide:

- `raw.fS` matching `AudioPacket.sample_rate_hz`; `raw.nChannels` matching array channels; `raw.nBits=16`; `raw.hopSize` matching the binding.
- `raw.interface.type="file"`, `raw.interface.path` equal to the binding's `raw_input_path`.
- `mapping.map` ODAS **1-based**, exact channel order corresponding to `MicrophoneArray.microphones` sorted by zero-based `channel`.
- `general.mics` physical positions in **metres**, and microphone orientation/directivity matching the hardware.
- `ssl.potential.format="json"`, `ssl.potential.interface.type="file"`, and path equal to binding `potential_output_path`.
- Appropriate sinks for other ODAS stages. Some ODAS configurations require full `sst`, `sss`, `classify` blocks even if they are not scored.
- The config must be tested with its installed ODAS version. The SDK does **not** assert compatibility with every ODAS release.

The configuration SHA256 and SDK geometry SHA256 must agree with the separately approved binding. These are change-detection and **operator-asserted bindings**: the code cannot prove the contents of arbitrary ODAS libconfig text are correct. A camera/robot array calibration remains the operator's responsibility. Run ODAS only on trusted local configurations.

Create a binding manifest **in the working environment where ODAS is installed**, for example:

```python
from pathlib import Path
import hashlib, json
from onda_hearing.spatial.odas_adapter import geometry_digest
from onda_hearing.contracts import MicrophoneArray
array = MicrophoneArray.from_positions(
    [[-.045,-.035,0],[.045,-.035,0],[.045,.035,0],[-.045,.035,0]],
    array_id='benchmark_array',frame_id='body',geometry_id='v09')
config = Path('/absolute/path/to/audited-odas.cfg').resolve()
work = Path('/absolute/path/to/empty-isolated-odas-workspace').resolve()
manifest = dict(config_path=str(config),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    geometry_sha256=geometry_digest(array),sample_rate_hz=16000,channels=4,
    raw_input_path=str(work/'audio.raw'),potential_output_path=str(work/'potential.json'),
    raw_hop_size=128,max_candidates=4)
Path('odas_binding.json').write_text(json.dumps(manifest,indent=2))
```

The paths inside the .cfg must match the manifest. Workspaces must be isolated, writable, and contain no stale raw/potential output. Do not configure ODAS to connect to arbitrary network endpoints from untrusted configs. The adapter uses subprocess with a fixed argument list (no shell), timeout, bounded JSON parsing and cleanup. It never fetches, installs, or downloads ODAS automatically.

## Run paired benchmark

```bash
python -m pytest -q
python benchmarks/odas_comparison_v010.py --limit 3
python benchmarks/odas_comparison_v010.py --odas-binding ./odas_binding.json
```

`results.csv` preserves explicit `odas_unavailable_no_binding`, `unsupported_motion_no_world_frame_steering`, or error states. `summary.json` reports `null` metrics when no cases were actually evaluated. Comparing methods on equal saved PCM does not imply comparable **temporal integration**: the current ODAS parser uses the **last** emitted SSL potential frame, while Onda's SRP aggregates across audio frames. Future work must add synchronized per-frame evaluation before interpreting any ODAS ranking.

The experimental covariance MUSIC baseline performs model-order selection using MDL per frequency band; it does not use truth source counts but is not motion compensated and is not a validated general multisource solution. Do not claim ODAS has been benchmarked unless a real ODAS binary ran and produced valid JSON.

## Next engineering gate

1. Test the adapter on a pinned, built ODAS revision with an independently audited config, using native ODAS CLI and fixture JSON.
2. Add per-frame timestamps, alignment and candidate matching for ODAS rather than last-frame summaries.
3. Validate channel map, microphone orientation, sample rate and audio scaling on actual ReSpeaker/Infineon arrays.
4. Compare static-source counts and bearings at equal temporal support; then evaluate synchronized robot movement and world frame tracking as separate qualification gates.
