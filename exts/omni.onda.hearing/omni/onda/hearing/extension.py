"""Onda v0.21B — interactive Isaac Sim USD hearing extension.

Stage ownership: ALL USD/Kit APIs are called on the Kit application thread.
The localhost FastAPI worker receives bounded commands and immutable snapshots.
The numeric rendering worker sees only captured Onda model objects, never USD.

Scope: interactive selection and static-pose snapshots, not continuous PhysX
or Fabric audio synchronization. USD source files are never written; the optional
microphone marker is authored into the USD *session* layer only.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from threading import Thread

import omni.ext
import omni.kit.app
import omni.ui as ui
import omni.usd


def _sdk_root():
    root=Path(__file__).resolve().parents[5]
    sdk=root/'src'
    if sdk.is_dir() and str(sdk) not in sys.path:
        sys.path.insert(0,str(sdk))
    return root


class OndaHearingExtension(omni.ext.IExt):
    def on_startup(self,ext_id):
        root=_sdk_root()
        from onda_hearing.simulators.isaac_scene_loader import (IsaacSceneLoader,SceneLoaderConfig,scene_identifier)
        from onda_hearing.simulators.isaac_emitter_registry import IsaacEmitterRegistry
        from onda_hearing.simulators.isaac_microphone_array_v021b import (MicrophoneRigStore,MARKER_PATH)
        from onda_hearing.simulators.isaac_gui_bridge import IsaacBridgeState,IsaacBridgeServer
        self._root=root; self._loader_type=IsaacSceneLoader
        self._loader_config=SceneLoaderConfig; self._scene_identifier=scene_identifier
        self._registry_type=IsaacEmitterRegistry; self._mic_store_type=MicrophoneRigStore
        self._marker_path=MARKER_PATH
        self._usd=omni.usd.get_context()
        self._state=IsaacBridgeState()
        self._registry=None;self._microphones=None;self._acoustic=None
        self._catalog=None;self._stage_key=None;self._stage_epoch=0
        self._selected_path=None
        self._window=None;self._stage_sub=None;self._update_sub=None;self._server=None
        self._render_thread=None
        from onda_hearing.simulators.isaac_stage_open_tracker_v021b import StageOpenTracker
        self._stage_opens=StageOpenTracker(timeout_seconds=float(
            os.environ.get('ONDA_ISAAC_STAGE_OPEN_TIMEOUT_S','180')))
        self._update_number=0;self._last_marker_check=0.;self._last_marker_save=0.
        self._last_open_heartbeat=0.
        self._render_dir=root/'runs'/'v021_isaac'/'renders'
        self._audio_root=Path(os.environ.get('ONDA_ISAAC_AUDIO_ROOT',str(root/'assets'/'onda_audio'))).expanduser().resolve()
        self._audio_root.mkdir(parents=True,exist_ok=True)
        self._explicit_emitter_manifest=os.environ.get('ONDA_ISAAC_EMITTER_MANIFEST')
        self._explicit_microphone_manifest=os.environ.get('ONDA_ISAAC_MICROPHONE_MANIFEST')
        self._sync_stage(force=True)
        self._stage_sub=self._usd.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event,name='Onda v021B stage')
        self._update_sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,name='Onda v021B Kit commands')
        self._state.set_status(connected_to_isaac=True,scene_loaded=self._usd.get_stage() is not None,
            version='v0.21B',static_acoustic_snapshot=True,live_pcm_streaming=False,
            audio_root=str(self._audio_root),rendering=False,
            disclaimer='Experimental geometry/specular acoustic model with uncalibrated demo materials')
        if os.environ.get('ONDA_ISAAC_DISABLE_HTTP','0')!='1':
            try:
                self._server=IsaacBridgeServer(self._state,
                    port=int(os.environ.get('ONDA_ISAAC_HTTP_PORT','8765')),
                    web_root=root/'web',render_root=self._render_dir,audio_root=self._audio_root)
                self._server.start()
                print(f'[ONDA] v0.21B Dashboard http://127.0.0.1:{self._server.port}/')
            except Exception as exc:
                self._state.set_status(last_error=f'HTTP bridge startup: {exc}')
                print(f'[ONDA] HTTP bridge unavailable: {exc}')
        # A GUI-capable Isaac session provides native controls. Headless mode
        # still exposes the browser API and all Kit-owned scene operations.
        try: self._build_ui()
        except Exception as exc:
            self._window=None
            print(f'[ONDA] Native panel unavailable (headless?): {exc}')
        print('[ONDA] v0.21B ready: interactive selections, 3..16 channel snapshots')

    def _sync_stage(self,force=False):
        from pxr import UsdGeom
        stage=self._usd.get_stage()
        key=self._scene_identifier(stage) if stage is not None else None
        if not force and stage is not None and key==self._stage_key:return
        self._stage_key=key;self._stage_epoch+=1
        self._catalog=None;self._registry=None;self._microphones=None;self._acoustic=None
        self._selected_path=None
        self._state.set_emitters([])
        self._state.set_selection({'prim_path':None,'type':None,'position_m':None})
        if stage is None:
            self._state.set_scene({'loaded':False,'source':None,'acoustics':None})
            self._state.set_microphones({'pose_available':False,'count':0})
            self._state.set_results({'last_render':None,'last_scene_build':None})
            self._state.set_status(scene_loaded=False,acoustic_scene_built=False,rendering=False)
            return
        # Each USD stage has a separate sidecar manifest; switching scenes
        # never overwrites another environment's emitter or mic setup.
        scene_dir=self._root/'runs'/'v021_isaac'/'scenes'/key
        emitter_manifest=Path(self._explicit_emitter_manifest) if self._explicit_emitter_manifest else scene_dir/'emitters.json'
        microphone_manifest=Path(self._explicit_microphone_manifest) if self._explicit_microphone_manifest else scene_dir/'microphones.json'
        self._registry=self._registry_type(stage=stage,audio_root=self._audio_root,
                                manifest=emitter_manifest,scene_id=key,sample_rate=16000)
        try: emit_report=self._registry.load()
        except Exception as exc:
            emit_report={'loaded':0,'error':str(exc)}
            self._state.set_status(last_error=f'emitter restore: {exc}')
        self._microphones=self._mic_store_type(microphone_manifest,key)
        try: mic_report=self._microphones.load(validate_anchor=self._assert_xformable)
        except Exception as exc:
            mic_report={'loaded':False,'error':str(exc)}
            self._state.set_status(last_error=f'microphone restore: {exc}')
        if self._microphones.rig.anchor_prim_path==self._marker_path:
            try: self._create_or_update_marker(initial=True)
            except Exception as exc: self._state.set_status(last_error=f'marker restore: {exc}')
        self._state.set_scene({'loaded':True,'source':str(stage.GetRootLayer().identifier),
            'scene_id':key,
            'default_prim':str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
            'meters_per_unit':float(UsdGeom.GetStageMetersPerUnit(stage)),
            'up_axis':str(UsdGeom.GetStageUpAxis(stage)),
            'acoustics':None,'manifest_load':emit_report,'microphone_manifest_load':mic_report})
        self._state.set_emitters(self._registry.list_records())
        self._publish_mics()
        self._state.set_results({'last_render':None,'last_scene_build':None})
        self._state.set_status(scene_loaded=True,acoustic_scene_built=False,
                               rendering=False,scene_epoch=self._stage_epoch,
                               emitter_manifest=str(emitter_manifest),microphone_manifest=str(microphone_manifest),
                               live_pcm_streaming=False)

    def _pose_at(self,path):
        from onda_hearing.acoustic_scene.adapters import IsaacUSDSceneSnapshot
        stage=self._usd.get_stage()
        if stage is None: raise RuntimeError('no stage')
        return IsaacUSDSceneSnapshot(stage).capture(path)

    def _assert_xformable(self,path):
        from pxr import UsdGeom
        from onda_hearing.simulators.isaac_emitter_registry import validate_prim_path
        validate_prim_path(path)
        stage=self._usd.get_stage()
        if stage is None: raise ValueError('no loaded scene')
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Xformable):
            raise ValueError(f'prim must be a valid Xformable object: {path}')
        return prim

    def _publish_mics(self):
        if self._microphones is None:
            self._state.set_microphones({'count':0,'pose_available':False})
            return
        self._state.set_microphones(self._microphones.rig.public(self._pose_at))

    def _selection_snapshot(self):
        from onda_hearing.simulators.isaac_scene_inspector_v021b import prim_details
        stage=self._usd.get_stage()
        if stage is None: return {'prim_path':None,'type':None,'position_m':None}
        selection=self._usd.get_selection().get_selected_prim_paths()
        if not selection:return {'prim_path':None,'type':None,'position_m':None}
        path=str(selection[-1])
        try:return prim_details(stage,path)
        except Exception as exc:
            return {'prim_path':path,'error':f'USD selection unavailable: {str(exc)[:200]}',
                    'emitter_eligible':False}

    def _on_stage_event(self,event):
        # This event fires for native 3D viewport AND Stage tree selections.
        from omni.usd import StageEventType
        if int(event.type) in (int(StageEventType.OPENED),int(StageEventType.CLOSED),
                                int(StageEventType.OPEN_FAILED)):
            # OPENED/CLOSED may precede the asynchronous open completion.
            # Do not initialize manifests against a half-composed USD stage.
            if self._stage_opens.current is None:
                self._sync_stage(force=True)
            else:
                self._state.set_status(load_phase='usd_stage_event',
                                       last_stage_event=int(event.type))
        elif int(event.type)==int(StageEventType.SELECTION_CHANGED):
            self._capture_selection()

    def _capture_selection(self):
        selected=self._usd.get_selection().get_selected_prim_paths()
        path=str(selected[-1]) if selected else None
        if path==self._selected_path:return
        self._selected_path=path
        self._state.set_selection(self._selection_snapshot())
        self._refresh_ui()

    def _on_update(self,event):
        # MAIN THREAD: do not call .drain from a worker or FastAPI handler.
        try:
            self._update_number+=1
            self._poll_stage_open()
            stage=self._usd.get_stage()
            if (self._stage_opens.current is None and stage is not None
                    and self._scene_identifier(stage)!=self._stage_key):
                self._sync_stage()
            self._state.drain(self._handle_command,max_count=12)
            # Selection event is primary; this is a fallback for Kit builds
            # where stage selection notifications are coalesced.
            if self._update_number%8==0:self._capture_selection()
            if self._microphones is not None and self._microphones.rig.anchor_prim_path==self._marker_path:
                # Track users dragging the marker through Isaac's transform gizmo,
                # but avoid filesystem writes on every frame.
                now=time.monotonic()
                if now-self._last_marker_check>.25:
                    self._last_marker_check=now
                    self._sync_marker_from_viewport(now)
        except Exception as exc:
            self._state.set_status(last_error=f'Kit update: {exc}')

    def _sync_marker_from_viewport(self,now):
        pose=self._pose_at(self._marker_path)
        record=self._microphones.rig
        if max(abs(a-b) for a,b in zip(pose.xyz,record.marker_world_position_m))<1e-5 and \
            max(abs(a-b) for a,b in zip(pose.quat_xyzw,record.marker_world_orientation_xyzw))<1e-5:
            return
        # Keep all per-mic offsets while reflecting marker drag and rotate.
        new=replace(record,marker_world_position_m=pose.xyz,
                    marker_world_orientation_xyzw=pose.quat_xyzw)
        self._microphones.rig=new
        self._publish_mics()
        if now-self._last_marker_save>.75:
            self._microphones.save();self._last_marker_save=now

    def _handle_command(self,name,payload):
        if name=='open_warehouse':return self._open_warehouse(payload)
        stage=self._usd.get_stage()
        if stage is None or self._registry is None or self._microphones is None:
            raise RuntimeError('no active USD stage; open Full Warehouse first')
        if name=='list_objects':
            from onda_hearing.simulators.isaac_scene_inspector_v021b import (catalog_from_stage,page_catalog)
            if self._catalog is None:
                self._catalog=catalog_from_stage(stage)
            return page_catalog(self._catalog,**payload)
        if name=='select_prim':
            if set(payload)!={'prim_path'}:raise ValueError('prim_path required')
            path=payload['prim_path']
            self._assert_xformable(path)
            ok=self._usd.get_selection().set_selected_prim_paths([path],False)
            if ok is False:raise RuntimeError('Isaac rejected selection')
            self._capture_selection()
            return {'selected':path}
        if name=='add_emitter':
            request=dict(payload)
            if not request.get('prim_path'):request['prim_path']=self._selected_path
            rec=self._registry.add(request)
            self._state.set_emitters(self._registry.list_records());self._refresh_ui()
            return {'emitter':vars(rec),'registry_revision':self._registry.revision}
        if name=='edit_emitter':
            if set(payload)!={'emitter_id','changes'}:raise ValueError('invalid emitter edit')
            rec=self._registry.update(payload['emitter_id'],payload['changes'])
            self._state.set_emitters(self._registry.list_records())
            return {'emitter':vars(rec),'registry_revision':self._registry.revision}
        if name=='remove_emitter':
            out=self._registry.remove(payload['emitter_id'])
            self._state.set_emitters(self._registry.list_records());return out
        if name=='set_microphones':
            request=dict(payload)
            record=self._microphones.set_rig(request,validate_anchor=self._assert_xformable)
            if record.anchor_prim_path==self._marker_path:
                self._create_or_update_marker(initial=('marker_world_position_m' in request))
            self._publish_mics()
            return {'microphones':self._state.snapshot('microphones')['data'],
                    'revision':self._microphones.revision}
        if name=='ensure_microphone_marker':
            if payload and set(payload)!={'world_position_m'}:
                raise ValueError('only world_position_m is allowed')
            old=self._microphones.rig
            world=tuple(payload.get('world_position_m',
                        old.marker_world_position_m if old.anchor_prim_path==self._marker_path else old.center_m))
            new=self._microphones.set_rig({
                'anchor_prim_path':self._marker_path,'center_m':(0.,0.,0.),
                'orientation_xyzw':(0.,0.,0.,1.),
                'marker_world_position_m':world},validate_anchor=None)
            self._create_or_update_marker(initial=True)
            self._publish_mics()
            return {'prim_path':self._marker_path,'microphones':new.public(self._pose_at)}
        if name=='build_scene':
            allowed={'region_center_m','region_radius_m','max_surfaces',
                    'max_triangles_total','max_triangles_per_mesh','max_reflectors',
                    'categories','include_prim_paths','exclude_path_substrings',
                    'allow_uncalibrated_demo_materials'}
            if set(payload)-allowed:raise ValueError(f'unknown scene options: {sorted(set(payload)-allowed)}')
            options=dict(payload)
            if 'region_center_m' not in options:
                mic_center=self._microphones.rig.capture(self._pose_at)[2].xyz
                sources=[self._pose_at(r['prim_path']).xyz for r in self._registry.list_records()]
                if sources:
                    options['region_center_m']=tuple((a+b)/2 for a,b in zip(mic_center,sources[0]))
                else: options['region_center_m']=mic_center
            config=self._loader_config(**options)
            self._state.set_status(building_geometry=True)
            try:built=self._loader_type(config).build(stage)
            finally:self._state.set_status(building_geometry=False)
            self._acoustic=built
            scene=self._state.snapshot('scene')['data'];scene['acoustics']=built.public_summary()
            self._state.set_scene(scene)
            self._state.set_results({'last_scene_build':built.public_summary()})
            self._state.set_status(acoustic_scene_built=True,last_error=None)
            self._refresh_ui()
            return built.public_summary()
        if name=='render_snapshot':return self._start_render_snapshot(payload)
        raise ValueError(f'unknown Isaac command: {name}')

    def _poll_stage_open(self):
        """Kit-thread completion of a callback-based USD stage open."""
        result=self._stage_opens.poll()
        if result is None:
            request=self._stage_opens.current
            if request is not None:
                now=time.monotonic()
                if now-self._last_open_heartbeat>=3.:
                    self._last_open_heartbeat=now
                    self._state.set_status(
                        loading_stage=True,
                        load_elapsed_s=round(now-request.started_at,1),
                        load_phase='opening')
            return
        if not result.success:
            self._state.set_status(
                loading_stage=False,
                load_phase='timeout' if result.timed_out else 'failed',
                last_error=f'open stage: {result.reason[:400]}',
                stage_load_token=result.token)
            print(f'[ONDA] warehouse load failed: {result.reason[:400]}')
            return
        stage=self._usd.get_stage()
        if stage is None:
            self._state.set_status(loading_stage=False,load_phase='failed',
                last_error='open stage reported success but USD context has no stage')
            return
        identifier=str(stage.GetRootLayer().identifier)
        if identifier!=result.url:
            self._state.set_status(loading_stage=False,load_phase='failed',
                last_error=f'USD loaded unexpected stage: {identifier[:300]}')
            return
        try:
            self._sync_stage(force=True)
        except Exception as exc:
            self._state.set_status(loading_stage=False,load_phase='failed',
                last_error=f'USD stage opened, Onda setup failed: {type(exc).__name__}: {str(exc)[:300]}')
            print(f'[ONDA] stage setup failed: {exc}')
            return
        self._state.set_status(loading_stage=False,load_phase='loaded',
            last_message='Full Warehouse loaded',last_error=None,
            stage_load_token=result.token)
        print(f'[ONDA] Full Warehouse loaded: {identifier}')

    def _open_warehouse(self,payload):
        if payload and set(payload)!={'asset'}:
            raise ValueError('only asset is supported')
        asset=payload.get('asset','/Isaac/Environments/Simple_Warehouse/full_warehouse.usd')
        if asset!='/Isaac/Environments/Simple_Warehouse/full_warehouse.usd':
            raise ValueError('only vetted Full Warehouse asset is enabled in v0.21B')
        from isaacsim.storage.native import get_assets_root_path
        root=get_assets_root_path()
        if not root:raise RuntimeError('Isaac asset root not available')
        url=root.rstrip('/')+asset
        pending=self._stage_opens.current
        if pending is not None:
            if pending.url!=url:
                raise ValueError('another USD stage is loading')
            # Repeated dashboard clicks are idempotent and DO NOT mask errors.
            return {'state':'loading','asset':asset,'request_id':pending.token,
                    'note':'already in progress; poll /api/isaac/status'}
        current=self._usd.get_stage()
        if current is not None and str(current.GetRootLayer().identifier)==url:
            self._sync_stage(force=False)
            self._state.set_status(loading_stage=False,load_phase='loaded',last_error=None)
            return {'state':'loaded','asset':asset,'note':'USD stage already active'}
        # If Kit is still busy after a watchdog timeout, do not queue another
        # potentially conflicting open. The original USD request may survive.
        if hasattr(self._usd,'can_open_stage') and not self._usd.can_open_stage():
            raise RuntimeError('USD context is busy opening another stage; inspect Kit logs')
        request,started=self._stage_opens.start(url)
        assert started
        self._last_open_heartbeat=time.monotonic()
        self._state.set_status(loading_stage=True,load_phase='opening',last_error=None,
                               load_elapsed_s=0.,stage_load_token=request.token)
        try:
            accepted=self._usd.open_stage_with_callback(
                url,self._stage_opens.callback_for(request.token))
        except Exception as exc:
            self._stage_opens.abandon(request.token)
            self._state.set_status(loading_stage=False,load_phase='failed',
                last_error=f'open_stage_with_callback: {type(exc).__name__}: {exc}')
            raise
        if not accepted:
            self._stage_opens.abandon(request.token)
            self._state.set_status(loading_stage=False,load_phase='failed',
                last_error='Isaac Sim declined the stage-open request')
            raise RuntimeError('Isaac Sim declined the stage-open request')
        print(f'[ONDA] opening Full Warehouse request={request.token} url={url}')
        return {'state':'loading','asset':asset,'request_id':request.token,
                'note':'poll /api/isaac/status for loaded, failed, or timeout'}

    def _start_render_snapshot(self,payload):
        from onda_hearing.simulators.isaac_audio_snapshot import render_static_snapshot
        if self._acoustic is None:raise ValueError('build acoustic geometry before rendering')
        if self._render_thread is not None and self._render_thread.is_alive():
            raise ValueError('another render is still running')
        allowed={'duration_s','specular_order','reflector_limit'}
        if set(payload)-allowed:raise ValueError('unknown rendering options; configure microphones separately')
        duration=float(payload.get('duration_s',0.6))
        order=int(payload.get('specular_order',1));limit=int(payload.get('reflector_limit',6))
        if not .05<=duration<=3. or order not in (0,1,2) or not 0<=limit<=12:
            raise ValueError('invalid duration, order or reflector limit')
        # Validate and snapshot paths on Kit; read potentially large audio files
        # later on the numerical worker so the viewport does not block on I/O.
        from onda_hearing.simulators.isaac_emitter_registry import EmitterRecord
        from onda_hearing.acoustic_scene.assets import load_wav
        enabled_records=[EmitterRecord(**record) for record in self._registry.list_records()
                         if record['enabled']]
        emitters=[record.to_onda() for record in enabled_records]
        audio_files={record.emitter_id:self._audio_root/record.audio_asset
                     for record in enabled_records}
        if not 1<=len(emitters)<=8:
            raise ValueError('rendering requires 1..8 enabled emitters; disable others')
        poses={e.binding.object_path:self._pose_at(e.binding.object_path)
               for e in emitters if e.binding.object_path}
        array,world_positions,center_pose=self._microphones.rig.capture(self._pose_at)
        scene=self._acoustic
        epoch=self._stage_epoch
        job=uuid.uuid4().hex
        output=self._render_dir/(job+'.wav')
        self._state.set_results({'last_render':{'job_id':job,'state':'running','channels':len(array.local_positions_m)}})
        self._state.set_status(rendering=True,last_error=None)
        def render_worker():
            try:
                assets={eid:load_wav(path,eid) for eid,path in audio_files.items()}
                report=render_static_snapshot(
                    geometry=scene.geometry,reflectors=scene.reflectors,
                    emitters=emitters,assets=assets,source_poses=poses,
                    microphone_center_m=center_pose.xyz,microphone_array=array,
                    output_file=output,duration_s=duration,sample_rate=16000,
                    specular_order=order,reflector_limit=limit)
                report.update({'job_id':job,'state':'complete','scene_id':scene.scene_id,
                               'audio_url':'/api/isaac/renders/'+job+'.wav'})
                if self._stage_epoch==epoch:
                    self._state.set_results({'last_render':report})
                    self._state.set_status(rendering=False,last_error=None)
            except Exception as exc:
                if self._stage_epoch==epoch:
                    self._state.set_results({'last_render':{'job_id':job,'state':'failed',
                                      'error':str(exc)[:500]}})
                    self._state.set_status(rendering=False,last_error=f'Render: {exc}')
        self._render_thread=Thread(target=render_worker,
                                   name='onda-v021b-static-multichannel-render',daemon=True)
        self._render_thread.start()
        return {'job_id':job,'state':'running','channels':len(array.local_positions_m),
                'note':'static USD snapshot; poll /api/isaac/results'}

    def _create_or_update_marker(self,initial=False):
        """Create selectable mic gizmo ONLY in session layer, never NVIDIA USD."""
        from pxr import Usd,UsdGeom,Gf
        stage=self._usd.get_stage()
        if stage is None:raise RuntimeError('no stage')
        rig=self._microphones.rig
        if rig.anchor_prim_path!=self._marker_path:raise ValueError('rig not marker anchored')
        scale=float(UsdGeom.GetStageMetersPerUnit(stage))
        if scale<=0:raise ValueError('invalid USD unit scale')
        existing=stage.GetPrimAtPath(self._marker_path)
        if existing and existing.IsValid() and not existing.GetCustomDataByKey('onda_microphone_marker'):
            raise ValueError('USD stage already contains /OndaMicrophoneRig; refusing to overwrite it')
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            xf=UsdGeom.Xform.Define(stage,self._marker_path)
            xf.GetPrim().SetCustomDataByKey('onda_microphone_marker',True)
            xformable=UsdGeom.Xformable(xf.GetPrim())
            ops=xformable.GetOrderedXformOps()
            trans=next((op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeTranslate),None)
            if trans is None:trans=xformable.AddTranslateOp()
            if initial or not ops:
                trans.Set(Gf.Vec3d(*(v/scale for v in rig.marker_world_position_m)))
                # Preserve marker orientation across stage reopen. Child mic
                # spheres inherit this quaternion automatically.
                orient=next((op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeOrient),None)
                if orient is None:orient=xformable.AddOrientOp()
                x,y,z,w=rig.marker_world_orientation_xyzw
                orient.Set(Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z))))
            for index in range(16):
                path=f'{self._marker_path}/Mic_{index+1:02d}'
                sphere=UsdGeom.Sphere.Define(stage,path)
                sphere.GetRadiusAttr().Set(float(.025/scale))
                image=UsdGeom.Imageable(sphere.GetPrim())
                image.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                image.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if index<len(rig.local_positions_m)
                                                  else UsdGeom.Tokens.invisible)
                ops=UsdGeom.Xformable(sphere.GetPrim()).GetOrderedXformOps()
                t=next((o for o in ops if o.GetOpType()==UsdGeom.XformOp.TypeTranslate),None)
                if t is None:t=UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp()
                if index<len(rig.local_positions_m):
                    t.Set(Gf.Vec3d(*(v/scale for v in rig.local_positions_m[index])))
        self._catalog=None

    def _build_ui(self):
        self._window=ui.Window('Onda Hearing — Scene Instrument',width=460,height=620)
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=7,height=0):
                    ui.Label('ONDA LABS  ·  Isaac Sim v0.21B',height=27)
                    ui.Label('Select a prim in viewport or Stage tree',height=20)
                    self._selected_label=ui.Label('Selected: (none)',word_wrap=True,height=58)
                    ui.Label('Audio file (inside Onda audio root)',height=18)
                    self._asset_field=ui.StringField(height=25)
                    self._asset_field.model.set_value('forklift_alarm.wav')
                    with ui.HStack(height=30):
                        ui.Label('Gain dB',width=100)
                        self._gain_field=ui.FloatField(width=100)
                        self._gain_field.model.set_value(0.)
                        ui.Label('Loop',width=40)
                        self._loop_field=ui.CheckBox(width=35)
                        self._loop_field.model.set_value(True)
                    ui.Button('Attach WAV to selected object',height=30,clicked_fn=self._attach_ui)
                    ui.Separator(height=12)
                    ui.Label('Microphone rig (3–16 channels)',height=23)
                    with ui.HStack(height=26):
                        ui.Label('Channels',width=95)
                        self._mic_count_field=ui.IntField(width=70)
                        self._mic_count_field.model.set_value(4)
                        ui.Label('Rig X / Y / Z in world meters',width=230)
                    with ui.HStack(height=29):
                        self._mx_field=ui.FloatField();self._my_field=ui.FloatField();self._mz_field=ui.FloatField()
                        for field,val in zip((self._mx_field,self._my_field,self._mz_field),(-1.,3.6,1.5)):
                            field.model.set_value(val)
                    ui.Button('Apply circular array',height=29,clicked_fn=self._apply_mics_ui)
                    ui.Button('Create / Select movable microphone rig',height=30,
                              clicked_fn=self._marker_ui)
                    ui.Label('Edit individual mic XYZ positions in browser dashboard',height=30)
                    ui.Separator(height=12)
                    ui.Label('Experimental warehouse acoustics',height=25)
                    ui.Button('Build bounded acoustic geometry',height=30,clicked_fn=self._build_scene_ui)
                    ui.Button('Render static microphone WAV',height=30,clicked_fn=self._render_ui)
                    self._status_label=ui.Label('Acoustic geometry not built',height=76,word_wrap=True)
                    ui.Label('Browser dashboard: http://127.0.0.1:8765/',height=26)
        self._refresh_ui()

    def _refresh_ui(self):
        if self._window is None:return
        if getattr(self,'_selected_label',None) is not None:
            details=self._state.snapshot('selection')['data']
            label=self._selected_path or '(none)'
            self._selected_label.text='Selected: '+label+'\n'+str(details.get('type',''))
        if getattr(self,'_status_label',None) is not None:
            if self._acoustic:
                s=self._acoustic.public_summary()
                self._status_label.text=f"Acoustics: {s['surfaces']} surfaces / {s['reflectors']} reflectors\nStatic USD snapshot; demo material curves"
            else:self._status_label.text='Acoustics not built. Materials are not calibrated.'

    def _ui_call(self,command,payload):
        try:
            out=self._handle_command(command,payload)
            self._state.set_status(last_message=command+' accepted',last_error=None)
            return out
        except Exception as exc:
            self._state.set_status(last_error=f'{command}: {exc}')
            print(f'[ONDA] {command} failed: {exc}')

    def _attach_ui(self):
        self._ui_call('add_emitter',{'prim_path':self._selected_path,
            'audio_asset':self._asset_field.model.get_value_as_string(),
            'gain_db':self._gain_field.model.get_value_as_float(),
            'loop':self._loop_field.model.get_value_as_bool()})

    def _apply_mics_ui(self):
        import math
        n=self._mic_count_field.model.get_value_as_int()
        if not 3<=n<=16:
            self._state.set_status(last_error='microphone count must be 3..16');return
        coords=[[.08*math.cos(2*math.pi*i/n),.08*math.sin(2*math.pi*i/n),0.] for i in range(n)]
        self._ui_call('set_microphones',{
            'anchor_prim_path':None,
            'center_m':(self._mx_field.model.get_value_as_float(),
                        self._my_field.model.get_value_as_float(),self._mz_field.model.get_value_as_float()),
            'orientation_xyzw':(0.,0.,0.,1.),'local_positions_m':coords})

    def _marker_ui(self):
        x=self._mx_field.model.get_value_as_float();y=self._my_field.model.get_value_as_float()
        z=self._mz_field.model.get_value_as_float()
        out=self._ui_call('ensure_microphone_marker',{'world_position_m':[x,y,z]})
        if out:
            self._usd.get_selection().set_selected_prim_paths([self._marker_path],False)
            self._capture_selection()

    def _build_scene_ui(self):
        self._ui_call('build_scene',{'allow_uncalibrated_demo_materials':True})

    def _render_ui(self):
        self._ui_call('render_snapshot',{'duration_s':.6,'specular_order':1,'reflector_limit':6})

    def on_shutdown(self):
        if self._microphones and self._microphones.rig.anchor_prim_path==self._marker_path:
            try:self._sync_marker_from_viewport(time.monotonic());self._microphones.save()
            except Exception:pass
        if self._state:self._state.close()
        if self._server:self._server.stop()
        if self._stage_opens:self._stage_opens.reset()
        self._update_sub=None;self._stage_sub=None
        if self._window:
            self._window.destroy();self._window=None
        print('[ONDA] v0.21B extension stopped')
