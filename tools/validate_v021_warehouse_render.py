#!/usr/bin/env python3
"""End-to-end Isaac Sim 5.1 USD -> Onda v0.20 renderer -> 4ch WAV smoke test.

Launches SimulationApp first. Uses the real Full Warehouse USD geometry and
binds a mono WAV emitter to the specified existing USD prim. Static snapshot
only (no continuous physics/Fabric motion). Default materials are explicitly
synthetic/uncalibrated. Keep duration short until geometry budget is profiled.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import wave
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", default="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
    p.add_argument("--source-prim", default="/Root/forklift/S_ForkliftBody")
    p.add_argument("--source-offset", type=float, nargs=3, default=(0., 0., 1.3))
    p.add_argument("--mic-center", type=float, nargs=3, default=(-1., 3.6, 1.5))
    p.add_argument("--duration", type=float, default=0.6)
    p.add_argument("--order", type=int, choices=(0, 1, 2), default=1)
    p.add_argument("--max-reflectors", type=int, default=6)
    p.add_argument("--max-surfaces", type=int, default=64)
    p.add_argument("--radius", type=float, default=15.)
    p.add_argument("--wav", default=None, help="Optional external mono 16k WAV; copied into demo audio root")
    p.add_argument("--out-dir", default="runs/v021_warehouse_render")
    return p.parse_args()


def write_demo_audio(path: Path, rate=16000):
    import numpy as np
    n = int(1.2*rate)
    t = np.arange(n, dtype=np.float64) / rate
    signal = 0.20 * (np.sin(2*np.pi*440*t) + 0.4*np.sin(2*np.pi*1200*t))
    envelope = np.minimum(1., np.minimum(t/.03, (1.2-t)/.03))
    pcm = (np.clip(signal*envelope, -1., 1.)*32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main():
    args = parse_args()
    if not 0.05 <= args.duration <= 5.:
        raise SystemExit("--duration must be 0.05..5 seconds")
    if not 1 <= args.max_reflectors <= 12:
        raise SystemExit("--max-reflectors must be 1..12 for this acceptance demo")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from onda_hearing.simulators.isaac_scene_loader import IsaacSceneLoader, SceneLoaderConfig, scene_identifier
        from onda_hearing.simulators.isaac_emitter_registry import IsaacEmitterRegistry
        from onda_hearing.acoustic_scene.adapters import StaticPoseResolver, IsaacUSDSceneSnapshot
        from onda_hearing.acoustic_scene.renderer import RendererConfig
        from onda_hearing.acoustic_scene.types import Binding, MicArray, Pose3
        from onda_hearing.acoustic_scene.unified_physics import UnifiedGeometryAcousticRenderer, UnifiedPhysicsConfig

        asset = args.asset
        if asset.startswith('/Isaac/'):
            root = get_assets_root_path()
            if not root: raise RuntimeError("Isaac asset root unavailable")
            asset = root.rstrip('/') + asset
        print("[ONDA] opening USD:", asset, flush=True)
        opened = open_stage(asset)
        success = opened[0] if isinstance(opened, tuple) else opened
        if not success: raise RuntimeError("Isaac Sim could not open the requested USD stage")
        for _ in range(4): app.update()
        stage = get_current_stage()
        if stage is None: raise RuntimeError("no stage after open_stage")
        prim = stage.GetPrimAtPath(args.source_prim)
        if not prim or not prim.IsValid():
            raise KeyError(f"source prim not found in USD: {args.source_prim}")
        source = IsaacUSDSceneSnapshot(stage).capture(args.source_prim)
        receiver = Pose3(tuple(args.mic_center))
        source_location = np.asarray(source.xyz) + np.asarray(args.source_offset)
        # A scene region containing source + microphone, not just the mic.
        mid = 0.5*(source_location+np.asarray(receiver.xyz))
        radius = max(args.radius, float(np.linalg.norm(source_location-np.asarray(receiver.xyz)))/2+5)
        config = SceneLoaderConfig(
            region_center_m=tuple(float(x) for x in mid),
            region_radius_m=radius,
            max_surfaces=args.max_surfaces,
            max_reflectors=args.max_reflectors,
            allow_uncalibrated_demo_materials=True,
        )
        start = time.perf_counter()
        scene = IsaacSceneLoader(config).build(stage)
        print("[ONDA] scene:", json.dumps(scene.public_summary(), indent=2), flush=True)
        audio_root = out / 'audio'
        audio_root.mkdir(parents=True, exist_ok=True)
        wav = audio_root/'onda_demo_beep.wav'
        if args.wav is not None:
            import shutil
            shutil.copyfile(Path(args.wav).resolve(strict=True), wav)
        else:
            write_demo_audio(wav)
        reg = IsaacEmitterRegistry(stage=stage, audio_root=audio_root,
                                   manifest=out/'emitters_demo.json',
                                   scene_id=scene_identifier(stage), sample_rate=16000)
        rec = reg.add({"emitter_id":"onda_warehouse_demo", "prim_path":args.source_prim,
                       "audio_asset":wav.name, "loop":True,
                       "local_offset_m":tuple(args.source_offset)})
        emitters, assets = reg.for_renderer()
        mics = MicArray(binding=Binding(world=receiver), local_positions_m=(
            (0.08,0.,0.), (0.,-0.08,0.), (-0.08,0.,0.), (0.,0.08,0.)))
        renderer = UnifiedGeometryAcousticRenderer(
            config=RendererConfig(sample_rate=16000, clip_output=False),
            microphones=mics, emitters=emitters, assets=assets,
            geometry=scene.geometry, reflectors=scene.reflectors,
            resolver=StaticPoseResolver({args.source_prim: source}),
            unified_config=UnifiedPhysicsConfig(
                max_specular_order=args.order,
                max_reflectors=max(1, len(scene.reflectors)),
                geometry_step_samples=256,
                max_path_states=768,
            ),
        )
        frames = int(round(args.duration*16000))
        pcm = renderer.render(0, frames)
        if pcm.shape != (frames, 4) or not np.isfinite(pcm).all():
            raise RuntimeError(f"invalid four-channel PCM: {pcm.shape}")
        over = int(np.count_nonzero(np.abs(pcm)>1))
        file_path = out/'warehouse_unified_4ch.wav'
        with wave.open(str(file_path), 'wb') as w:
            w.setnchannels(4); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((np.clip(pcm, -1, 1)*32767).astype('<i2').tobytes())
        elapsed = time.perf_counter()-start
        report = {
            'result':'PASS', 'mode':'isaac_usd_static_snapshot_v021',
            'asset':args.asset, 'source_prim':rec.prim_path,
            'source_world_position_m':list(map(float, source_location)),
            'microphone_center_m':list(receiver.xyz),
            'audio_file':str(file_path),
            'samples_per_channel':frames, 'channels':4,
            'rms':float(np.sqrt(np.mean(pcm.astype(float)**2))),
            'peak_abs':float(np.max(np.abs(pcm))),
            'pcm_overrange_samples':over,
            'wall_seconds_total':elapsed,
            'acoustic_scene':scene.public_summary(),
            'renderer':renderer.telemetry(),
            'limitations':['DEMO_SYNTHETIC_UNCALIBRATED_materials',
                           'static_USD_source_poses', 'no_fabric_motion_sync',
                           'no_diffraction_or_late_diffuse_reverb'],
        }
        (out/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print("[ONDA] result:", json.dumps({k:report[k] for k in (
            'result','channels','rms','peak_abs','pcm_overrange_samples',
            'wall_seconds_total','audio_file')}, indent=2), flush=True)
    finally:
        app.close()


if __name__=='__main__': main()
