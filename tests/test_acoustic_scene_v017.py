import math
import numpy as np
import pytest
from onda_hearing.acoustic_scene import (
    Pose3,Binding,Schedule,Emitter,MicArray,AudioAsset,RendererConfig,
    AcousticSceneRenderer,StaticPoseResolver,PoseTimeline)
from onda_hearing.acoustic_scene.types import compose_pose
from onda_hearing.acoustic_scene.adapters import resolve

FS=16000
MIC=MicArray(Binding(world=Pose3((0,0,0))),((0,0,0),(.06,0,0)))
SINE=np.sin(2*np.pi*823*np.arange(FS*2)/FS).astype(np.float32)*.2

def scene(source=(2,0,0), **kw):
    return AcousticSceneRenderer(RendererConfig(sample_rate=FS,**kw),MIC,
       [Emitter('phone','ring',Binding(world=Pose3(source)),Schedule(onset_s=0,loop=True))],
       {'ring':AudioAsset('ring',SINE,FS)})

def test_validation_binding():
    with pytest.raises(ValueError):Binding()
    with pytest.raises(ValueError):Binding(Pose3((0,0,0)),'robot')
    with pytest.raises(ValueError):Pose3((0,0,0),(0,0,0,0))

def test_validation_asset():
    with pytest.raises(ValueError):AudioAsset('a',[float('nan')],FS)
    with pytest.raises(ValueError):AudioAsset('a',[1.5],FS)

def test_missing_audio():
    with pytest.raises(KeyError):AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('x','missing',Binding(world=Pose3((1,0,0))))],{})

def test_rate_mismatch():
    with pytest.raises(ValueError):AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('x','a',Binding(world=Pose3((1,0,0))))],{'a':AudioAsset('a',[0.5],8000)})

def test_chunk_equivalence():
    a=scene().render(0,1024)
    b=scene();c=np.concatenate([b.render(0,137),b.render(137,887)])
    np.testing.assert_allclose(a,c,atol=1e-6)

def test_nonsequential_rejected():
    s=scene();s.render(0,32)
    with pytest.raises(ValueError):s.render(0,32)

def test_reset_reproduces():
    s=scene(noise_std=.001,seed=44)
    a=s.render(0,100);s.reset();b=s.render(0,100)
    np.testing.assert_array_equal(a,b)

def test_noise_chunk_invariance():
    s=scene(noise_std=.001,seed=4);a=s.render(0,311)
    s.reset();b=np.concatenate([s.render(0,101),s.render(101,210)])
    np.testing.assert_allclose(a,b,atol=1e-7)

def test_delay_relative_geometry():
    s=scene(source=(3,0,0))
    x=s.render(0,1024)
    assert np.max(np.abs(x))>.01
    assert not np.array_equal(x[:,0],x[:,1])

def test_inverse_distance():
    near=scene(source=(1,0,0)).render(0,2048)
    far=scene(source=(3,0,0)).render(0,2048)
    assert np.sqrt(np.mean(near**2)) > 2*np.sqrt(np.mean(far**2))

def test_fixed_object_pose_matches_world():
    p=StaticPoseResolver({'phone':Pose3((2.,0.,0.))})
    fixed=scene().render(0,512)
    attached=AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('p','ring',Binding(object_path='phone'),Schedule(loop=True))],{'ring':AudioAsset('ring',SINE,FS)},p).render(0,512)
    np.testing.assert_array_equal(fixed,attached)

def test_offset_rotates():
    q=(0.,0.,math.sin(math.pi/4),math.cos(math.pi/4))
    p=compose_pose(Pose3((1.,2.,0.),q),(1.,0.,0.),(0.,0.,0.,1.))
    np.testing.assert_allclose(p.xyz,(1,3,0),atol=1e-8)

def test_onset_and_duration():
    s=AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('p','ring',Binding(world=Pose3((1,0,0))),Schedule(onset_s=.05,duration_s=.05,loop=True))],{'ring':AudioAsset('ring',SINE,FS)})
    a=s.render(0,1900)
    assert np.count_nonzero(a[:700])==0
    assert np.count_nonzero(a[900:1450])>0
    assert np.count_nonzero(a[1750:])==0

def test_silence_disabled():
    s=AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('p','ring',Binding(world=Pose3((1,0,0))),enabled=False)],{'ring':AudioAsset('ring',SINE,FS)})
    assert np.all(s.render(0,100)==0)

def test_multiple_emitters_are_additive():
    e=Emitter('one','ring',Binding(world=Pose3((2,0,0))),Schedule(loop=True))
    a=AcousticSceneRenderer(RendererConfig(),MIC,[e],{'ring':AudioAsset('ring',SINE,FS)}).render(0,300)
    e2=Emitter('two','ring',Binding(world=Pose3((2,0,0))),Schedule(loop=True))
    b=AcousticSceneRenderer(RendererConfig(),MIC,[e,e2],{'ring':AudioAsset('ring',SINE,FS)}).render(0,300)
    np.testing.assert_allclose(2*a,b,atol=1e-6)

def test_directivity_rear_attenuated():
    def power(point):
        ss=AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('p','ring',Binding(world=Pose3(point)),Schedule(loop=True),directivity='cardioid')],{'ring':AudioAsset('ring',SINE,FS)})
        return np.sqrt(np.mean(ss.render(0,1000)**2))
    assert power((-2,0,0))>power((2,0,0))*10

def test_timeline_interpolates_and_refuses_far_future():
    timeline=PoseTimeline()
    timeline.append('a',0.,Pose3((0,0,0)))
    timeline.append('a',1.,Pose3((2,0,0)))
    assert timeline.pose_at('a',.5).xyz[0]==pytest.approx(1.)
    with pytest.raises(ValueError):timeline.pose_at('a',1.2)

def test_moving_source():
    timeline=PoseTimeline()
    timeline.append('phone',0.,Pose3((2,0,0)))
    timeline.append('phone',.1,Pose3((2,1,0)))
    s=AcousticSceneRenderer(RendererConfig(),MIC,[Emitter('e','ring',Binding(object_path='phone'),Schedule(loop=True))],{'ring':AudioAsset('ring',SINE,FS)},timeline)
    a=s.render(0,1000)
    assert np.isfinite(a).all() and np.sqrt(np.mean(a**2))>.001

def test_rir_chunk_equivalence():
    ir=np.zeros((3,2));ir[0]=1;ir[2]=.2
    s=scene(rir_taps=ir);a=s.render(0,500)
    s.reset();b=np.concatenate([s.render(0,99),s.render(99,401)])
    np.testing.assert_allclose(a,b,atol=1e-6)

def test_shape_dtype():
    a=scene().render(0,100)
    assert a.shape==(100,2) and a.dtype==np.float32 and a.flags['C_CONTIGUOUS']

def test_event_trigger_and_stop_propagation():
    s=AcousticSceneRenderer(RendererConfig(),MIC,
        [Emitter('alarm','ring',Binding(world=Pose3((1,0,0))),Schedule(onset_s=None,loop=True))],
        {'ring':AudioAsset('ring',SINE,FS)})
    a=s.render(0,100)
    assert not np.any(a)
    s.trigger('alarm',150)
    b=s.render(100,200)
    assert np.all(b[:50]==0)
    assert np.count_nonzero(b[120:])>0
    s.stop('alarm',330)
    c=s.render(300,180)
    assert np.count_nonzero(c[:40])>0
    assert np.all(c[100:]==0)


def test_runtime_gain_control():
    s=scene()
    a=s.render(0,300)
    s.reset();s.set_gain_db('phone',-6.020599913279624)
    b=s.render(0,300)
    np.testing.assert_allclose(a*.5,b,atol=1e-6)


def test_trigger_past_rejected():
    s=scene();s.render(0,200)
    with pytest.raises(ValueError):s.trigger('phone',100)
    with pytest.raises(ValueError):s.stop('phone',150)

def test_explicit_pcm_overrange_telemetry():
    asset=AudioAsset('loud',np.ones(2000,dtype=np.float32)*.9,FS)
    e=Emitter('e','loud',Binding(world=Pose3((.2,0,0))),Schedule(loop=True))
    s=AcousticSceneRenderer(RendererConfig(clip_output=True),MIC,[e],{'loud':asset})
    out=s.render(0,100)
    assert np.max(out)<=1 and s.telemetry()['clipped_samples']>0
    s.reset()
    assert s.telemetry()['clipped_samples']==0

def test_direct_path_sample_delay_matches_geometry():
    """Exactly 100 samples of propagation at 343 m/s and 16 kHz."""
    distance=343.*100/FS
    single=MicArray(Binding(world=Pose3((0,0,0))),((0,0,0),))
    impulse=np.zeros(300,dtype=np.float32);impulse[0]=1.
    s=AcousticSceneRenderer(RendererConfig(),single,[Emitter('s','pulse',Binding(world=Pose3((distance,0,0))))],{'pulse':AudioAsset('pulse',impulse,FS)})
    y=s.render(0,250)[:,0]
    assert int(np.argmax(y))==100
    assert y[100]==pytest.approx(1./distance,rel=1e-5)
    assert np.count_nonzero(y[:100])==0


def test_pose_timeline_vector_matches_scalar_pose():
    timeline=PoseTimeline()
    timeline.append('robot',0.,Pose3((0,0,0),(0,0,0,1)))
    timeline.append('robot',.5,Pose3((1,1,0),(0,0,1,0)))
    timeline.append('robot',1.,Pose3((2,2,0),(0,0,0,1)))
    query=np.linspace(0,1,51)
    pos,q=timeline.sample_arrays('robot',query)
    for idx,t in enumerate(query):
        p=timeline.pose_at('robot',float(t))
        np.testing.assert_allclose(pos[idx],p.xyz,atol=1e-7)
        assert abs(float(np.dot(q[idx],p.quat_xyzw)))==pytest.approx(1.,abs=1e-7)


def test_pose_timeline_expiry_detected():
    timeline=PoseTimeline(maxlen=3)
    for n in range(5):timeline.append('head',n*.1,Pose3((n,0,0)))
    with pytest.raises(ValueError):timeline.sample_arrays('head',np.array([.01]))


def test_renderer_moving_offset_applied():
    timeline=PoseTimeline()
    timeline.append('phone',0.,Pose3((2,0,0)))
    timeline.append('phone',.1,Pose3((2,1,0)))
    s=AcousticSceneRenderer(RendererConfig(),MIC,
        [Emitter('e','ring',Binding(object_path='phone',local_offset_m=(.2,0,0)),Schedule(loop=True))],
        {'ring':AudioAsset('ring',SINE,FS)},timeline)
    assert np.isfinite(s.render(0,1024)).all()
