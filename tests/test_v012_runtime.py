"""Numerical invariants and simulator-adapter tests; not physical-simulator smoke tests."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from onda_hearing.contracts import MicrophoneArray,SpatialResult
from onda_hearing.physics import AcousticConfig
from onda_hearing.runtime import (HearingRuntime,SceneSnapshot,ScheduledSource,SampleScheduler,
    rotation_from_xyzw,xyzw_from_rotation)
from onda_hearing.simulators import MuJoCoSceneAdapter,IsaacSimSceneAdapter

ARRAY=MicrophoneArray.from_positions([[.04,.04,0],[-.04,.04,0],[-.04,-.04,0],[.04,-.04,0]],
                                     array_id='robot_array',frame_id='robot/head',geometry_id='v1')

def snap(t,angle=0,source=(1.,0.,0.)):
    return SceneSnapshot(t,(0,0,0),(0,0,math.sin(angle/2),math.cos(angle/2)),{'phone':source})

def new_runtime(perception=None):
    rng=np.random.default_rng(3)
    signal=(.02*rng.normal(size=16000)).astype(np.float32)
    return HearingRuntime(ARRAY,(ScheduledSource('phone',signal,0),),perception=perception)

def test_scheduler_exact_count_120hz_one_second():
    clock=SampleScheduler(48000,0)
    for k in range(1,121):
        ts=round(k*1e9/120)
        portion=clock.preview(ts)
        clock.commit(ts,portion)
    assert clock.emitted==48000

def test_scheduler_zero_sample_steps_and_fractional_rate():
    clock=SampleScheduler(16000,0)
    first=clock.preview(20_000);assert first.count==0
    clock.commit(20_000,first)
    second=clock.preview(125_000);assert second.count==2

def test_scheduler_rejects_rewind_and_duplicate_time():
    clock=SampleScheduler(16000,0)
    with pytest.raises(ValueError):clock.preview(0)
    p=clock.preview(20_000);clock.commit(20_000,p)
    with pytest.raises(ValueError):clock.preview(10_000)

def test_same_pcm_from_two_chunkings_stationary():
    r1=new_runtime();r1.advance(snap(0))
    one=r1.advance(snap(200_000_000)).audio.pcm
    r2=new_runtime();r2.advance(snap(0))
    a=r2.advance(snap(100_000_000)).audio.pcm
    b=r2.advance(snap(200_000_000)).audio.pcm
    assert np.array_equal(one,np.vstack([a,b]))

def test_output_contract_and_timestamp():
    rt=new_runtime()
    assert rt.advance(snap(0)) is None
    output=rt.advance(snap(100_000_000))
    assert output.audio.pcm.shape==(1600,4)
    assert output.audio.sample_rate_hz==16000
    assert output.audio.capture.start_ns==0
    assert output.audio.capture.end_ns==100_000_000
    assert output.observation.spatial.directions==()
    assert output.observation.spatial.backend_id=='no_perception'

def test_reset_replays_identically():
    rt=new_runtime();rt.advance(snap(0));a=rt.advance(snap(100_000_000)).audio.pcm.copy()
    rt.reset();rt.advance(snap(0));b=rt.advance(snap(100_000_000)).audio.pcm.copy()
    assert np.array_equal(a,b)

def test_source_mismatch_and_rollback():
    rt=new_runtime();rt.advance(snap(0))
    with pytest.raises(ValueError):rt.advance(SceneSnapshot(100_000_000,(0,0,0),(0,0,0,1),{}))
    assert rt.advance(snap(100_000_000)).sample_stop==1600

def test_invalid_pcm_does_not_advance_clock():
    rt=HearingRuntime(ARRAY,(ScheduledSource('phone',np.ones(16000),0,5.),))
    rt.advance(snap(0))
    with pytest.raises(ValueError,match='outside'):
        rt.advance(snap(100_000_000,source=(.5,0,0)))
    assert rt._scheduler.emitted==0

def test_perception_sees_no_source_ground_truth():
    class Spy:
        def __init__(self):self.called=False
        def reset(self):pass
        def estimate(self,audio,array):
            self.called=True
            assert hasattr(audio,'pcm') and array==ARRAY
            assert not hasattr(audio,'source_positions_m')
            return SpatialResult(audio.capture,audio.frame_id,(),'test_spy')
    spy=Spy();rt=new_runtime(spy);rt.advance(snap(0))
    assert rt.advance(snap(100_000_000)).observation.spatial.backend_id=='test_spy'
    assert spy.called

def test_motion_affects_pcm():
    moving=new_runtime();moving.advance(snap(0));x=moving.advance(snap(200_000_000,angle=math.pi/2)).audio.pcm
    fixed=new_runtime();fixed.advance(snap(0));y=fixed.advance(snap(200_000_000)).audio.pcm
    assert np.max(np.abs(x-y))>1e-5

def test_rotation_roundtrip_180_and_90():
    for angle in (0,math.pi/2,math.pi):
        q=(0,0,math.sin(angle/2),math.cos(angle/2))
        assert np.allclose(rotation_from_xyzw(xyzw_from_rotation(rotation_from_xyzw(q))),rotation_from_xyzw(q))

def test_nonmatching_source_clock_rejected():
    with pytest.raises(ValueError):SceneSnapshot(-1,(0,0,0),(0,0,0,1),{'phone':(1,0,0)})
    with pytest.raises(ValueError):SceneSnapshot(0,(0,0,0),(0,0,0,2),{'phone':(1,0,0)})

def test_mujoco_reads_and_copies_mutable_memory():
    class Model:
        def site(self,name):return SimpleNamespace(id={'head':0,'phone':1}[name])
    class Data:
        site_xpos=np.array([[1.,2.,3.],[4.,5.,6.]])
        site_xmat=np.array([np.eye(3).reshape(-1),np.eye(3).reshape(-1)])
    data=Data()
    adapter=MuJoCoSceneAdapter(Model(),data,array_site='head',source_sites={'phone':'phone'})
    shot=adapter.snapshot(200)
    data.site_xpos[:]=100
    assert shot.array_position_m==(1.,2.,3.)
    assert shot.source_positions_m['phone']==(4.,5.,6.)

def test_mujoco_rejects_invalid_rotation():
    class Model:
        def site(self,name):return SimpleNamespace(id=0)
    data=SimpleNamespace(site_xpos=np.zeros((1,3)),site_xmat=np.zeros((1,9)))
    adapter=MuJoCoSceneAdapter(Model(),data,array_site='head',source_sites={})
    with pytest.raises(ValueError):adapter.snapshot(0)

def test_isaac_adapter_is_optional_import():
    # In this environment pxr is not part of the core SDK.
    import importlib.util
    if importlib.util.find_spec('pxr') is None:
        with pytest.raises(RuntimeError,match='requires pxr'):
            IsaacSimSceneAdapter(None,array_prim='/World/head',source_prims={})

def test_reject_huge_advance():
    cfg=AcousticConfig(sample_rate_hz=16000,max_samples=500)
    rt=HearingRuntime(ARRAY,(ScheduledSource('phone',np.zeros(16000),0),),config=cfg)
    rt.advance(snap(0))
    with pytest.raises(ValueError,match='maximum'):rt.advance(snap(100_000_000))

def test_isaac_adapter_reads_stage_prims_via_usd_cache(monkeypatch):
    import sys,types
    class Vec:
        def __init__(self,values):self.values=values
        def __getitem__(self,idx):return self.values[idx]
    class Quat:
        def GetImaginary(self):return Vec([0.,0.,0.])
        def GetReal(self):return 1.
    class Mat:
        def __init__(self,xyz):self.xyz=xyz
        def ExtractTranslation(self):return Vec(self.xyz)
        def ExtractRotationQuat(self):return Quat()
    class Prim:
        def __init__(self,path):self.path=path
        def IsValid(self):return True
    class Stage:
        def GetPrimAtPath(self,path):return Prim(path)
    class XformCache:
        def __init__(self,timecode):pass
        def GetLocalToWorldTransform(self,prim):
            return Mat((1,2,3) if prim.path.endswith('head') else (4,5,6))
    fake_pxr=types.ModuleType('pxr')
    fake_pxr.Usd=types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda:None))
    fake_pxr.UsdGeom=types.SimpleNamespace(XformCache=XformCache)
    monkeypatch.setitem(sys.modules,'pxr',fake_pxr)
    adapter=IsaacSimSceneAdapter(Stage(),array_prim='/World/head',source_prims={'phone':'/World/phone'})
    shot=adapter.snapshot(1_000_000)
    assert shot.array_position_m==(1.,2.,3.)
    assert shot.source_positions_m['phone']==(4.,5.,6.)
    assert shot.array_orientation_xyzw==(0.,0.,0.,1.)


def test_live_runtime_packet_times_are_contiguous_at_44100():
    from onda_hearing.runtime import ScheduledSource,HearingRuntime
    fs=44100
    r=HearingRuntime(ARRAY,(ScheduledSource('phone',np.zeros(10000),0),),sample_rate_hz=fs)
    r.advance(snap(0))
    packets=[]
    for i in range(1,6):
        result=r.advance(snap(i*1_000_000))
        if result:packets.append(result.audio)
    assert sum(len(p.pcm) for p in packets)==220
    for before,after in zip(packets,packets[1:]):
        assert before.capture.end_ns==after.capture.start_ns
    assert packets[-1].capture.end_ns==round(220*1e9/fs)


def test_pcm_propagation_not_privileged_source_position():
    r=new_runtime();r.advance(snap(0));output=r.advance(snap(100_000_000))
    assert not hasattr(output.observation,'source_positions_m')
    assert np.any(output.audio.pcm!=0)
