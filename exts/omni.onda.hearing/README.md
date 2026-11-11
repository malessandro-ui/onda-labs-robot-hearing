# Onda Hearing native Isaac Sim extension — v0.21

Enable using `isaacsim isaacsim.exp.full --ext-folder "$PWD/exts" --enable omni.onda.hearing` from SDK root with `PYTHONPATH="$PWD/src"`.

Select a USD prim and attach a mono 16 kHz PCM WAV from `assets/onda_audio`.
The extension starts a localhost-only HTTP bridge (default `127.0.0.1:8765`), unless `ONDA_ISAAC_DISABLE_HTTP=1`.

This release supports selection, persisted emitter bindings, bounded static acoustic USD extraction, and background static four-channel render jobs. It does **not** yet provide continuous Isaac Lab/Fabric pose sync, a robot control loop or calibrated acoustic materials.
