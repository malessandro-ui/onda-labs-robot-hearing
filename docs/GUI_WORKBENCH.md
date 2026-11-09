# Onda Hearing Instrument

The GUI is a compact acoustic-intelligence instrument over the real v0.17
acoustic-scene renderer. The scene stays visually primary; a small lower-right
popup shows the robot's current bearing and confidence. Click it to reveal the
waveform, audio playback, sound meaning, scene setup and export controls.

It does not use a second scene schema or fabricate perception results. The
instrument answers four questions:

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

## Add objects and listen live

1. Open the lower-right instrument and choose **Add object**. Name it and use
   **Bearing** and **Distance** to place it in the scene. You can add up to eight objects.
2. Pick a built-in sound or attach a mono, uncompressed 16 kHz PCM WAV (up to
   10 seconds and 330 KB). Leave **Loop sound** enabled for continuous emission.
3. Choose **Start live**. The compass and waveform update about eight times per
   second. Choose **Monitor audio** to hear the simulated microphone mix.
4. Change the scene while listening. The GUI restarts the stream after a short
   debounce and recomputes the estimate from the edited scene. One-shot sounds
   play again when the stream restarts.

Every scene object uses an SDK `Binding` and `Emitter` with its own pose and
audio asset. The reference renderer feeds sequential PCM into
`StreamingHearing` for GCC-PHAT/SRP localization. If configured, BEATs classifies
recent audio windows asynchronously. Live mode is wall-clock paced and runs
against the browser's reference simulation.

**Render clip** runs the offline render with a configurable duration and enables
**Export** for its JSON diagnostics. The report includes custom WAV names and
omits audio bytes. Attached files are held in the current browser page.

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

When these requirements are absent, the instrument says **BEATs not configured**.
When available, classifications describe the complete audio window. The GUI
does not attach a label to a direction because v0.17 does not implement a
validated semantic-spatial association layer.

## Current limits

- The GUI owns its scene object poses. Native MuJoCo/Isaac poses and physical
  microphone capture use the SDK's separate runtime adapters.
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
