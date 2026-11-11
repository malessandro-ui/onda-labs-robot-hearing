"""Onda v0.21B multichannel static Isaac USD snapshot renderer.

Only consumes Onda pose/geometry objects captured on Kit's application thread.
The worker performs numerical synthesis (no USD or Kit access). Supports 3..16
custom microphone coordinates and orientations, and exposes bounded, genuine
v0.20 reflector diagnostics. No fabricated acoustics or simulator-motion claims.
"""
from __future__ import annotations

from pathlib import Path
import math
import os
import tempfile
import time
import wave
import numpy as np

from onda_hearing.acoustic_scene.adapters import StaticPoseResolver
from onda_hearing.acoustic_scene.renderer import RendererConfig
from onda_hearing.acoustic_scene.types import Binding, Emitter, MicArray, Pose3, rotate
from onda_hearing.acoustic_scene.unified_physics import UnifiedGeometryAcousticRenderer, UnifiedPhysicsConfig

DEFAULT_MIC_OFFSETS_M = (
    (0.08, 0., 0.), (0., -0.08, 0.), (-0.08, 0., 0.), (0., 0.08, 0.),
)


def _waveform_envelope(samples: np.ndarray, bins:int=128) -> list[dict]:
    # Memory and HTTP payload bounded even for 16 microphones.
    n, ch = samples.shape
    intervals=np.linspace(0,n,bins+1,dtype=int)
    out=[]
    for c in range(ch):
        hi=[]; lo=[]
        for i in range(bins):
            chunk=samples[intervals[i]:max(intervals[i]+1,intervals[i+1]),c]
            hi.append(round(float(np.max(chunk)),6)); lo.append(round(float(np.min(chunk)),6))
        out.append({"mic":c,"min":lo,"max":hi})
    return out


def _reflection_summary(renderer, limit=100):
    """Summarize real computed paths; keys are emitter, mic index, reflector IDs."""
    result=[]
    for (eid,channel,surface_ids), data in sorted(renderer.last_reflections.items(),
                  key=lambda row:(row[0][0],row[0][1],row[0][2]))[:limit]:
        result.append({"emitter_id":eid,"microphone_index":int(channel),
                       "surface_ids":list(surface_ids),"length_m":float(data["path_length_m"]),
                       "arrival_delay_ms":float(data["delay_s"])*1000.,
                       "bounce_points_m":[list(map(float,p)) for p in data["bounce_points_m"]],
                       "transmitted_surfaces":list(data["penetrated"])})
    return result


def render_static_snapshot(*, geometry, reflectors, emitters:list[Emitter],assets,
                           source_poses:dict[str,Pose3],microphone_center_m,
                           output_file:str|Path,duration_s:float=0.6,
                           sample_rate:int=16000,specular_order:int=1,
                           reflector_limit:int=6,microphone_array:MicArray|None=None) -> dict:
    if not math.isfinite(float(duration_s)) or not .05 <= duration_s <= 3.:
        raise ValueError("duration_s must be 0.05..3 seconds")
    if sample_rate != 16000:
        raise ValueError("v0.21B snapshot mode requires 16000 Hz PCM")
    if specular_order not in (0,1,2) or not isinstance(reflector_limit,int) or not 0<=reflector_limit<=12:
        raise ValueError("invalid reflection order/reflector limit")
    if not 1<=len(emitters)<=8: raise ValueError("static snapshot supports 1..8 emitters")
    if microphone_array is None:
        microphone_array=MicArray(Binding(world=Pose3(tuple(microphone_center_m))),DEFAULT_MIC_OFFSETS_M)
    if not isinstance(microphone_array,MicArray) or microphone_array.binding.world is None:
        raise ValueError("snapshot microphone_array requires a captured world Pose3")
    channel_count=len(microphone_array.local_positions_m)
    if not 3<=channel_count<=16:
        raise ValueError("v0.21B snapshot supports 3..16 channels")
    for e in emitters:
        if e.binding.object_path and e.binding.object_path not in source_poses:
            raise ValueError(f"missing source USD pose: {e.binding.object_path}")
    refs={key:reflectors[key] for key in list(reflectors)[:reflector_limit]}
    r=len(refs)
    sequences=r if specular_order>=1 else 0
    if specular_order>=2: sequences+=r*(r-1)
    path_count=sequences*len(emitters)*channel_count
    if path_count>10000:
        raise ValueError("reflection path-state budget exceeded; reduce reflectors/order/emitters")
    config=UnifiedPhysicsConfig(max_specular_order=specular_order,
           max_reflectors=max(1,r),max_path_states=max(768,path_count),
           reflection_taps=129,transition_samples=320,geometry_step_samples=256)
    renderer=UnifiedGeometryAcousticRenderer(
        RendererConfig(sample_rate=sample_rate,clip_output=False,noise_std=0.),
        microphone_array,emitters,assets,geometry,refs,
        resolver=StaticPoseResolver(source_poses),unified_config=config)
    start=time.perf_counter()
    frames=int(round(duration_s*sample_rate))
    chunks=[renderer.render(offset,min(4096,frames-offset)) for offset in range(0,frames,4096)]
    pcm=np.concatenate(chunks,axis=0)
    if pcm.shape!=(frames,channel_count) or not np.isfinite(pcm).all():
        raise RuntimeError("Onda generated invalid microphone PCM")
    render_seconds=time.perf_counter()-start
    wav=Path(output_file).expanduser().resolve(strict=False)
    wav.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".onda_pcm_",suffix=".wav",dir=str(wav.parent))
    os.close(fd)
    try:
        with wave.open(tmp,"wb") as file:
            file.setnchannels(channel_count);file.setsampwidth(2);file.setframerate(sample_rate)
            file.writeframes((np.clip(pcm,-1.,1.)*32767).astype("<i2").tobytes())
        os.replace(tmp,wav)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    pose=microphone_array.binding.world
    world_positions=[list(map(float,np.asarray(pose.xyz)+rotate(pose.quat_xyzw,p)))
                     for p in microphone_array.local_positions_m]
    rms=[float(np.sqrt(np.mean(np.square(pcm[:,c],dtype=np.float64)))) for c in range(channel_count)]
    peak=[float(np.max(np.abs(pcm[:,c]))) for c in range(channel_count)]
    return {
        "result":"PASS","mode":"isaac_usd_static_snapshot_v021b",
        "sample_rate":sample_rate,"channels":channel_count,"frames_per_channel":frames,
        "duration_s":duration_s,"rms":float(np.sqrt(np.mean(np.square(pcm,dtype=np.float64)))),
        "max_abs":float(np.max(np.abs(pcm))),
        "overrange_samples":int(np.count_nonzero(np.abs(pcm)>1.)),
        "render_wall_seconds":render_seconds,
        "render_realtime_factor":float(duration_s/render_seconds) if render_seconds>0 else None,
        "reflectors_used":r,"specular_order":specular_order,"emitter_count":len(emitters),
        "source_world_positions_m":{
            path:list(pose.xyz) for path,pose in source_poses.items()},
        "microphone_world_positions_m":world_positions,
        "microphone_center_m":list(pose.xyz),"microphone_orientation_xyzw":list(pose.quat_xyzw),
        "channel_rms":rms,"channel_peak":peak,
        "waveform_envelope":_waveform_envelope(pcm),
        "active_reflection_paths":_reflection_summary(renderer),
        "active_reflection_paths_total":len(renderer.last_reflections),
        "renderer_telemetry":renderer.telemetry(),
        "limitations":["STATIC_USD_POSE_SNAPSHOT","NO_LIVE_PHYSX_FABRIC_MOTION_SYNC",
                       "GEOMETRIC_SPECULAR_NO_DIFFRACTION","MATERIAL_CURVES_MAY_BE_UNCALIBRATED"],
    }
