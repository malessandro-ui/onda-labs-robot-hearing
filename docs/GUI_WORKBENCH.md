# Onda Hearing Workbench

The workbench is a robot acoustic intelligence observatory over the real v0.17
acoustic-scene renderer. It does not use a second scene schema or fabricate
perception results. Its information hierarchy answers four questions:

1. What does the robot hear?
2. Where does the sound come from?
3. What does Onda Sound Brain / BEATs think produced it?
4. How confident and current is the robot's interpretation?

## Start

```bash
python -m pip install -e '.[dev]'
onda-hearing-gui
```

The server binds to `127.0.0.1:8765` and opens the browser. To run without
opening a browser, or to choose another port:

```bash
onda-hearing-gui --no-browser --port 9000
```

The convenience script performs the editable install and starts the same CLI:

```bash
./tools/run_gui.sh
```

## Implemented workflow

1. Select a microphone geometry and reference acoustic environment.
2. Add, remove and configure up to eight emitters.
3. Configure sound, bearing, distance, gain, onset and directivity.
4. Run deterministic PCM rendering through `AcousticSceneRenderer`.
5. Inspect PCM-derived GCC-PHAT/SRP localization and policy features.
6. Listen to the rendered mono monitor mix and inspect all microphone channels.
7. Export a JSON report without embedding the audio payload.

The GUI uses simulator truth only to draw the evaluation overlay and calculate
single-source angular error. Localization and policy observations receive PCM.
When multiple emitters are active, the GUI explicitly marks source association
as unresolved.

## Optional Onda Sound Brain / BEATs

The observatory detects BEATs configuration but never silently downloads or
loads a checkpoint. Configure a trusted local model explicitly:

```bash
export ONDA_BEATS_CHECKPOINT=/absolute/path/to/checkpoint.pt
export ONDA_BEATS_LABELS=/absolute/path/to/ordered_labels.json
export ONDA_BEATS_MODULE_DIR=/absolute/path/to/microsoft/unilm/beats
export ONDA_GUI_ALLOW_TRUSTED_BEATS=1
onda-hearing-gui
```

When these requirements are absent, the semantic panel says **Not configured**.
When available, classifications describe the complete audio window. The GUI
does not attach a label to a direction because v0.17 does not implement a
validated semantic-spatial association layer.

## Current limits

- Fixed-world scene authoring is implemented. Object-bound emitters remain
  available through the Python API and native MuJoCo/Isaac adapters.
- Room presets are stationary FIR approximations, not geometric ray tracing.
- The Waveformer option is visible but disabled. v0.17 contains no qualified
  Waveformer runtime adapter, so the GUI fails closed instead of simulating one.
- The Infineon board requires its measured microphone coordinates, channel map,
  timing and gain calibration before it should be added as a named preset.
- This is a localhost development tool, not a remotely exposed production
  service. It has no authentication and defaults to loopback binding.

## Test

```bash
python -m pytest -q tests/test_gui_workbench.py
python -m pytest -q
```
