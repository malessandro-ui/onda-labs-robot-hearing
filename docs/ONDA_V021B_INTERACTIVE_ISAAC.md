# Onda v0.21B — Interactive Isaac Sim Scene Instrument

## Capability contract

This is an integration on top of **working v0.21A + USD pose fix**, using
v0.20.2's existing unified direct + finite specular reflection engine.

**Interactive:** select USD objects in Isaac Sim or by browser search; inspect
prim type, USD/physics APIs, world position and bounds; attach/edit/remove WAV
emitters; upload stereo/mono WAV with explicit conversion to 16 kHz mono;
configure **3–16 microphone coordinates**; optionally create a viewport rig
marker in the USD session layer; save per-scene sidecar configs; request static
acoustic scene builds; render multi-channel PCM; inspect each mic's amplitude,
world positions, computed specular bounce records, and runtime telemetry.

**Not implemented:** physics-step streaming audio during a continuously moving
scene, Fabric/PhysX poses synchronized every step, acoustic material calibration,
diffraction, or late diffuse reverberation. Changing warehouse geometry after a
build requires rebuilding the snapshot; moving emitters/rigs are captured at
render-start. GUI updates are live; audio is **on-demand** and can be slower
than real time on complex scenes. Complex visual meshes beyond loader budgets
are deliberately skipped instead of converted to solid boxes.

## Architecture

- `src/onda_hearing/simulators/isaac_scene_loader.py`: bounded USD-to-acoustic
  converter (v0.21A; v0.21B prevents marker geometry being treated as acoustic).
- `isaac_scene_inspector_v021b.py`: cached object catalog and prim inspection.
- `isaac_emitter_registry.py`: persistent, transactional, validated emitter
  settings, now supporting edits.
- `isaac_microphone_array_v021b.py`: microphone rigs, world/prim anchored poses,
  3–16 channels and scene-keyed atomic JSON persistence.
- `isaac_wav_import_v021b.py`: bounded explicit WAV channel conversion and
  polyphase sample-rate conversion.
- `isaac_audio_snapshot.py`: multichannel static-Pose3 renderer; per-mic stats,
  wave-envelope, and actual `UnifiedGeometryAcousticRenderer.last_reflections`.
- `isaac_gui_bridge.py`: localhost HTTP and Kit-thread command queue. The API
  worker *never touches USD*; render workers only touch captured acoustic models.
- `exts/omni.onda.hearing/omni/onda/hearing/extension.py`: Kit extension,
  selection updates, optional session-layer rig and asynchronous render jobs.
- `web/index.html`: browser Hearing Studio, usable via SSH port forwarding even
  when the RTX workstation has no graphical display.

## Installation

```bash
cd /ephemeral/onda-nvidia
unzip -q Onda_v021B_Interactive_Isaac_DropIn.zip
cd /ephemeral/onda-nvidia/onda-v015
source /ephemeral/onda-nvidia/env_isaaclab/bin/activate
python ../Onda_v021B_Interactive_Isaac_DropIn/APPLY.py --sdk-root "$PWD" --dry-run
python ../Onda_v021B_Interactive_Isaac_DropIn/APPLY.py --sdk-root "$PWD" --install
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_isaac_*_v021b.py
PYTHONPATH="$PWD/src" python -m pytest -q
```

The installer verifies exact v0.21A source hashes before replacing the six
existing files plus extension manifest. It creates `*.pre_v021b.bak` backups,
installs new files additively, and supports `--rollback`. The v0.21A patched
`acoustic_scene/adapters.py`, all v0.20 physics and all existing project files
are untouched. Never run `git add .` on a repo containing untracked `runs/`.

## Native headless acceptance on Shadeform

Run a real six-channel warehouse render and verify generated PCM:

```bash
cd /ephemeral/onda-nvidia/onda-v015
source /ephemeral/onda-nvidia/env_isaaclab/bin/activate
export PYTHONPATH="$PWD/src:$PWD/exts/omni.onda.hearing${PYTHONPATH:+:$PYTHONPATH}"
python -u -X faulthandler tools/validate_v021b_native.py \
  > /tmp/onda_v021b_native.log 2>&1
code=$?
echo "Exit code: $code"
grep -E '\[ONDA B\]|Traceback|Error|PASS|FAIL' /tmp/onda_v021b_native.log | tail -90
cat runs/v021b_native_smoke/report.json
```

The native test can run *headlessly*. For the actual native GUI widget you must
use an Isaac Sim session with a visible 3D viewport (on the remote machine or
through a supported streaming desktop). Headless browser access still supports
object search/selection and snapshot rendering but does not display a 3D
viewport. A successful `pytest` run **is not** proof that native Kit UI works;
run this acceptance and then test viewport interaction separately.

## Launch interactive

```bash
cd /ephemeral/onda-nvidia/onda-v015
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
isaacsim isaacsim.exp.full --ext-folder "$PWD/exts" --enable omni.onda.hearing
```

In a separate Mac terminal, for the remote localhost API:

```bash
ssh -i ~/Downloads/YOUR_SSH_KEY.pem -L 8765:127.0.0.1:8765 \
  shadeform@YOUR_SHADEFORM_IP
```

Open `http://127.0.0.1:8765` on the Mac. If the host does not have a desktop
session, you may run the extension with a headless Isaac Kit experience, but
must use the browser's object catalog rather than the native 3D viewport.

## Typical workflow

1. In the browser select **Open NVIDIA Full Warehouse** or use Isaac's File >
   Open Scene. The asynchronous loader only permits that vetted asset URL.
2. Search `forklift` and choose `/Root/forklift/S_ForkliftBody`, or select it in
   the native viewport. The selected object detail panel includes world pose,
   dimensions and available USD API schemas.
3. Upload WAV or use a preinstalled 16 kHz mono PCM WAV under
   `$ONDA_ISAAC_AUDIO_ROOT` (default `assets/onda_audio`); choose gain, loop,
   onset, optional duration and directivity; click **Attach**. Later choose
   **Edit** or **Remove**. The source follows the prim's USD pose at render time.
4. Select a triangle, square, hexagon or circle microphone preset, then edit
   every XYZ coordinate manually. Save the rig or choose **Create movable
   viewport rig**. In Isaac Sim select `/OndaMicrophoneRig` and use the Move
   gizmo; position changes are reflected in the dashboard. Viewport markers
   are authored only in the session layer, never back to NVIDIA's USD asset.
5. Explicitly acknowledge synthetic/uncalibrated acoustic coefficients and
   build bounded geometry around source and rig. (Rebuild after USD edits.)
6. Set reflection order 0, 1 or 2 and render the current static pose. The WAV
   has the **same number of channels as the microphone rig**, not a forced 4.
7. Inspect waveform envelopes, per-channel RMS/peak, world coordinates,
   actual computed bounce coordinates and delays, active path counts, and
   renderer time. Download the multichannel WAV for Onda localization/BEATs.

## HTTP endpoints

Existing v0.21A API remains supported:

```
GET     /api/isaac/scene
GET     /api/isaac/selection
GET     /api/isaac/emitters
POST    /api/isaac/emitters
DELETE  /api/isaac/emitters/{id}
GET     /api/isaac/status
GET     /api/isaac/results
GET     /api/isaac/events?since=...
POST    /api/isaac/scene/build
POST    /api/isaac/render
GET     /api/isaac/renders/{uuid}.wav
```

New v0.21B API:

```
GET     /api/isaac/objects?q=forklift&offset=0&limit=32
POST    /api/isaac/selection              {"prim_path":"/Root/forklift"}
PATCH   /api/isaac/emitters/{id}          {"gain_db":-3,"loop":false}
GET     /api/isaac/microphones
PUT     /api/isaac/microphones            {"local_positions_m":[[...],[...],[...]], ...}
POST    /api/isaac/microphones/marker     {"world_position_m":[-1,3.6,1.5]}
GET     /api/isaac/audio-assets
POST    /api/isaac/audio-assets           application/octet-stream WAV (<=64 MiB)
POST    /api/isaac/open-warehouse        {}
```

Mutating requests require `X-Onda-Client: OndaGUI`. This is a localhost-only,
**developer integration, not a production authentication solution**. Never
publish port 8765 directly to an internet-facing interface. Browser same-origin
preflight behavior and loopback binding reduce accidental cross-origin use,
but anyone with localhost access to the port can issue requests. For remote
access use a trusted SSH tunnel and restrict access to the host.

## Performance and fidelity

Static geometry is capped by selected area/triangle/surface budgets. The native
scene can have ~26k prims and ~677k estimated triangles, but Onda operates on
selected bounded acoustic surfaces. Static wave rendering occurs off the Kit
thread; the USD geometry build currently runs on the Kit thread and can cause
a temporary viewport pause. `render_realtime_factor` measures **numerical audio
rendering only**, not Isaac stage loading or USD scene building.

Reflection diagnostics come from the renderer's real `last_reflections` data
and include bounce points, length, delay, penetration and relevant surface
IDs. They represent **geometric model predictions**, not independently
measured ground truth. No diffraction, late diffuse reverb, true door seals,
or calibrated warehouse materials. No continuous real-time PCM streaming yet.
