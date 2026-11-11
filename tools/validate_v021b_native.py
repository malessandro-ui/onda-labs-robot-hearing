"""Native Isaac Sim 5.1 smoke: imported warehouse, object catalog, emitter,
6-microphone array and real v0.20 specular geometry rendering.

Run inside env_isaaclab from SDK root. The original NVIDIA USD is not edited.
The script authors only a test WAV and per-scene Onda sidecar configuration.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
import uuid
import wave

import numpy as np

# Initialize Kit before importing Omniverse USD modules.
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':640,'height':480})
try:
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.utils.stage import open_stage

    sdk=Path.cwd()
    source=sdk/'exts'/'omni.onda.hearing'/'omni'/'onda'/'hearing'/'extension.py'
    spec=importlib.util.spec_from_file_location('onda_v021b_native_smoke_extension',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OndaHearingExtension=module.OndaHearingExtension

    asset='/Isaac/Environments/Simple_Warehouse/full_warehouse.usd'
    url=get_assets_root_path().rstrip('/')+asset
    print('[ONDA B] opening',url,flush=True)
    opened=open_stage(url)
    ok=opened[0] if isinstance(opened,tuple) else opened
    if not ok:raise RuntimeError('USD warehouse failed to open')
    for _ in range(6):app.update()
    stage=omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath('/Root/forklift/S_ForkliftBody').IsValid():
        raise RuntimeError('Full Warehouse forklift USD prim unavailable')
    # Isolate smoke assets and manifests so repeat runs never change users'
    # existing scene registrations or microphone arrangements.
    smoke_root=sdk/'runs'/'v021b_native_smoke'
    smoke_root.mkdir(parents=True,exist_ok=True)
    session=smoke_root/('session_'+uuid.uuid4().hex[:12])
    session.mkdir(parents=True,exist_ok=False)
    os.environ['ONDA_ISAAC_DISABLE_HTTP']='1'
    os.environ['ONDA_ISAAC_AUDIO_ROOT']=str(session/'audio')
    os.environ['ONDA_ISAAC_EMITTER_MANIFEST']=str(session/'emitters.json')
    os.environ['ONDA_ISAAC_MICROPHONE_MANIFEST']=str(session/'microphones.json')
    ext=OndaHearingExtension()
    ext.on_startup('omni.onda.hearing')
    assert ext._state.snapshot('scene')['data']['loaded']
    objects=ext._handle_command('list_objects',{'q':'forklift','offset':0,'limit':40,'category':''})
    if objects['total']<1:raise RuntimeError('object browser failed to index forklift')
    selected=ext._handle_command('select_prim',{'prim_path':'/Root/forklift/S_ForkliftBody'})
    details=ext._state.snapshot('selection')['data']
    if not details.get('position_m'):raise RuntimeError('selected forklift has no valid world pose')
    name='onda_v021b_smoke_'+uuid.uuid4().hex[:8]
    audio=ext._audio_root/(name+'.wav')
    t=np.arange(16000,dtype=np.float64)/16000
    waveforms=(.12*np.sin(2*np.pi*440*t)*32767).astype('<i2')
    with wave.open(str(audio),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
        wav.writeframes(waveforms.tobytes())
    print('[ONDA B] attaching WAV to selected forklift',flush=True)
    ext._handle_command('add_emitter',{'emitter_id':name,
         'prim_path':'/Root/forklift/S_ForkliftBody','audio_asset':audio.name,
         'gain_db':-3.,'loop':True})
    ext._handle_command('set_microphones',{'center_m':(-1.,3.6,1.5),
         'preset':'hexagon6','anchor_prim_path':None})
    print('[ONDA B] building USD acoustic geometry',flush=True)
    built=ext._handle_command('build_scene',{'region_center_m':(1.09,7.2,1.4),
         'region_radius_m':15.,'max_reflectors':6,
         'allow_uncalibrated_demo_materials':True})
    if built['surfaces']<1:raise RuntimeError('warehouse loader built no acoustic surfaces')
    print('[ONDA B] scene:',built['surfaces'],'surfaces',built['reflectors'],'reflectors',flush=True)
    job=ext._handle_command('render_snapshot',{'duration_s':.2,
         'specular_order':1,'reflector_limit':6})
    deadline=time.monotonic()+120
    report=None
    while time.monotonic()<deadline:
        app.update()
        report=ext._state.snapshot('results')['data'].get('last_render')
        if report and report.get('job_id')==job['job_id'] and report.get('state') in ('complete','failed'):
            break
        time.sleep(.015)
    if not report or report.get('state')!='complete':
        raise RuntimeError(f'Native renderer did not succeed: {report}')
    if report['channels']!=6 or report['rms']<=0:
        raise RuntimeError('Expected non-silent six-channel PCM')
    output=ext._render_dir/(job['job_id']+'.wav')
    with wave.open(str(output),'rb') as wav:
        assert wav.getnchannels()==6 and wav.getframerate()==16000 and wav.getnframes()==3200
    result={'result':'PASS','asset':asset,
            'source_prim':'/Root/forklift/S_ForkliftBody',
            'channels':report['channels'],'rms':report['rms'],
            'active_reflection_paths':report['active_reflection_paths_total'],
            'output_wav':str(output),
            'render_seconds':report['render_wall_seconds'],
            'geometry_summary':built,'renderer_telemetry':report['renderer_telemetry'],
            'limitations':report['limitations']}
    (smoke_root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('[ONDA B] PASS',json.dumps(result,indent=2),flush=True)
finally:
    try:
        if 'ext' in locals():ext.on_shutdown()
    except Exception as exc:print('[ONDA B] shutdown warning:',exc,flush=True)
    app.close()
