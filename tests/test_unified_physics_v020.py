"""Physics unit and regression-contract checks for v0.20.

These establish geometry, sample streaming, and energy-model consistency;
not real-world material calibration.
"""
import math
import numpy as np
import pytest

from onda_hearing.acoustic_scene import (Pose3, Binding, Schedule,
    Emitter, MicArray, AudioAsset, RendererConfig)
from onda_hearing.acoustic_scene.physical import (
    TransmissionMaterial, AcousticSurface, AcousticGeometry,
    GeometryAwareAcousticRenderer)
from onda_hearing.acoustic_scene.unified_physics import (
    ReflectionMaterial, UnifiedPhysicsConfig, UnifiedGeometryAcousticRenderer,
    find_specular_path)

SR = 16000
CLEAR = TransmissionMaterial((125.,8000.),(60.,60.),'synthetic_opaque_wall')
REFL = ReflectionMaterial((125.,8000.),(.2,.2),(0.,0.),'synthetic_test')


def wall_y(y, sid='side', xmin=-4, xmax=4):
    return AcousticSurface.quad(sid, ((xmin,y,0),(xmax,y,0),
                                      (xmax,y,3),(xmin,y,3)),CLEAR)


def wall_x(x, sid='left', ymin=-4, ymax=4):
    return AcousticSurface.quad(sid, ((x,ymin,0),(x,ymax,0),
                                      (x,ymax,3),(x,ymin,3)),CLEAR)


def make_scene(signal=None, refl=True, noise=0., refmat=REFL, wall=None,
               cfg=None, order=1):
    if signal is None:
        signal = np.zeros(SR*2,dtype=np.float32)
        signal[0] = .2
    mic = MicArray(Binding(world=Pose3((2.,0.,1.))),
                   ((0.,0.,0.),(0.,.04,0.),(0.,-.04,0.),(.04,0.,0.)))
    e = Emitter('one','src',Binding(world=Pose3((1.,0.,1.))),Schedule(onset_s=0))
    w = wall or wall_y(2.)
    geom = AcousticGeometry((w,))
    conf = cfg or UnifiedPhysicsConfig(max_specular_order=order,
                                       transition_samples=320)
    r = UnifiedGeometryAcousticRenderer(
        RendererConfig(sample_rate=SR,noise_std=noise,seed=91),mic,[e],
        {'src':AudioAsset('src',np.asarray(signal,dtype=np.float32),SR)},geom,
        {w.surface_id:refmat} if refl else {},unified_config=conf)
    return r,geom


def render_all(r,n,chunk):
    blocks=[]
    while r.next_sample<n:
        at=r.next_sample
        blocks.append(r.render(at,min(chunk,n-at)))
    return np.concatenate(blocks)


def test_reflection_coefficients_and_material_provenance():
    with pytest.raises(ValueError):
        ReflectionMaterial((400,100),(0.,.3),(0.,0.),'test')
    with pytest.raises(ValueError):
        ReflectionMaterial((100,400),(-.2,.3),(0.,0.),'test')
    with pytest.raises(ValueError):
        ReflectionMaterial((100,400),(.2,.3),(0.,0.),'')
    loss=REFL.equivalent_loss().transmission_loss_db
    assert all(abs(x-(-10*math.log10(.8)))<1e-10 for x in loss)


def test_single_bounce_point_and_geometric_length():
    w=wall_y(2.)
    path=find_specular_path((1.,0.,1.),(2.,0.,1.),(w,),AcousticGeometry((w,)))
    assert path is not None
    expected=np.linalg.norm(np.array([2,0,1])-np.array([1,4,1]))
    assert abs(path.length_m-expected)<1e-8
    np.testing.assert_allclose(path.points_m[0],(1.5,2.,1.),atol=1e-8)
    assert path.penetrated == ()


def test_finite_surface_excludes_outside_bounce():
    w=wall_y(2.,xmin=-1,xmax=1)
    assert find_specular_path((1,0,1),(2,0,1),(w,),AcousticGeometry((w,))) is None


def test_receiver_other_side_cannot_have_specular_bounce():
    w=wall_y(2.)
    assert find_specular_path((1,1,1),(2,3,1),(w,),AcousticGeometry((w,))) is None


def test_two_bounces_have_consistent_reverse_tracing():
    bot=wall_y(0.,'bottom',xmin=0.,xmax=5.)
    left=wall_x(0.,'left',ymin=0.,ymax=5.)
    geom=AcousticGeometry((bot,left))
    path=find_specular_path((1.,1.,1.),(2.,3.,1.),(bot,left),geom)
    assert path is not None
    assert abs(path.length_m-math.sqrt(3**2+4**2))<1e-8
    assert len(path.points_m)==2
    assert path.points_m[0][1] == pytest.approx(0.,abs=1e-8)
    assert path.points_m[1][0] == pytest.approx(0.,abs=1e-8)


def test_reflection_later_than_direct_for_impulse():
    r,_=make_scene()
    y=render_all(r,500,128)[:,0]
    first=int(np.argmax(abs(y[:100])))
    reflection=int(np.argmax(abs(y[120:260]))+120)
    assert first in range(42,51)  # 1m / 343 m/s at 16kHz
    expect=int(round(math.hypot(1,4)*SR/343.))
    assert abs(reflection-expect)<5,(reflection,expect)
    assert np.max(abs(y[120:260]))>1e-3


def test_no_reflections_identical_to_v019_direct():
    r,geo=make_scene(refl=False,noise=.01)
    source=r.direct
    r2,_=make_scene(refl=False,noise=.01)
    # v0.20 combined dry direct + only one global noise stream.
    np.testing.assert_allclose(render_all(r,4096,313),render_all(r2,4096,4096),atol=1e-7)
    assert not r.last_reflections
    assert source.telemetry()['overrange_samples']==0


def test_reflection_streaming_chunk_invariant():
    x=(.06*np.sin(2*np.pi*523.2*np.arange(SR)/SR)).astype(np.float32)
    a,_=make_scene(signal=x)
    b,_=make_scene(signal=x)
    np.testing.assert_allclose(render_all(a,SR,113),render_all(b,SR,4096),atol=1e-6)


def test_reflection_absorption_at_unity_suppresses_reflected_energy():
    signal=np.zeros(SR,dtype=np.float32);signal[0]=.1
    absorbed=ReflectionMaterial((125.,8000.),(1.,1.),(0.,0.),'test_100pct_absorption')
    r,_=make_scene(signal=signal,refmat=absorbed)
    d,_=make_scene(signal=signal,refl=False)
    a=render_all(r,2048,512)
    b=render_all(d,2048,512)
    assert np.max(abs(a-b))<1e-6


def test_occluded_reflection_uses_explicit_transmission():
    r,g=make_scene()
    reflected=r._sequences[0]
    with_open=render_all(r,3000,512)
    # A vertical slab crosses first leg of wall reflection path at y=1m;
    # but leaves the source->microphone direct path at y=0 unobstructed.
    slab=AcousticSurface.quad('screen',((0.,1.,0.),(3.,1.,0.),
                                               (3.,1.,3.),(0.,1.,3.)),CLEAR)
    w=wall_y(2.)
    geo=AcousticGeometry((w,slab))
    r2, _=make_scene()
    r2.geometry.put(slab)
    with_screen=render_all(r2,3000,512)
    delayed=int(round(math.hypot(1,4)*SR/343.))
    open_peak = np.max(abs(with_open[delayed-3:delayed+5,0]))
    closed_peak = np.max(abs(with_screen[delayed-3:delayed+5,0]))
    assert open_peak > 1e-3
    assert closed_peak < .3*open_peak
    assert r2.last_reflections[('one',0,('side',))]['penetrated']


def test_moving_surface_geometry_revision_and_no_nan():
    tone=np.full(SR*2,.04,dtype=np.float32)
    r,g=make_scene(signal=tone)
    first=render_all(r,SR//2,257)
    # Dynamically replace specular wall with a wall far outside the ray footprint.
    g.put(wall_y(2.,xmin=20,xmax=30))
    second=render_all(r,SR,257)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert r.telemetry()['geometry_revision']==g.revision
    assert r.active_specular_paths==0


def test_trigger_stop_gain_propagation_and_reset():
    r,_=make_scene()
    r.set_gain_db('one',-12.)
    r.trigger('one',0)
    r.stop('one',600)
    assert r.render(0,256).shape==(256,4)
    with pytest.raises(ValueError):r.render(0,256)
    r.reset()
    assert r.render(0,256).shape==(256,4)


def test_reject_full_legacy_rir_and_path_explosion():
    from onda_hearing.acoustic_scene.renderer import RendererConfig
    cfg=RendererConfig(sample_rate=SR,rir_taps=np.ones((10,4),np.float32))
    x=np.zeros(4096,dtype=np.float32)
    with pytest.raises(ValueError,match='double-count'):
        make_scene(signal=x,cfg=None)[0].__class__(cfg,None,[],{},AcousticGeometry(),{})
    with pytest.raises(ValueError,match='budget'):
        make_scene(cfg=UnifiedPhysicsConfig(max_path_states=1))


def test_unconfigured_reflectors_are_not_guessed():
    r,_=make_scene(refl=False)
    assert r._sequences==[]
    assert r.telemetry()['physics_mode'].startswith('unified')


def test_without_reflectors_matches_v019_including_noise():
    signal=(.04*np.sin(2*np.pi*547*np.arange(SR)/SR)).astype(np.float32)
    r,_=make_scene(signal=signal,refl=False,noise=.012)
    baseline=GeometryAwareAcousticRenderer(
        r.config,r.microphones,list(r.emitters),
        {'src':AudioAsset('src',signal,SR)},r.geometry)
    actual=render_all(r,4096,311)
    reference=render_all(baseline,4096,311)
    np.testing.assert_allclose(actual,reference,atol=2.e-7,rtol=1.e-6)


def test_material_reflection_energy_shifts_delayed_signal():
    src=np.zeros(4096,dtype=np.float32);src[0]=.1
    lo=ReflectionMaterial((125,8000),(.2,.2),(0.,0.),'test')
    hi=ReflectionMaterial((125,8000),(.8,.8),(0.,0.),'test')
    rlo,_=make_scene(signal=src,refmat=lo)
    rhi,_=make_scene(signal=src,refmat=hi)
    ylo=render_all(rlo,1024,256)[:,0]
    yhi=render_all(rhi,1024,256)[:,0]
    # Restrict to arrival of reflection, excluding earlier direct component.
    ratio=np.max(abs(yhi[150:250]))/np.max(abs(ylo[150:250]))
    assert abs(ratio-.5)<.03


def test_native_mujoco_hinged_door_contract():
    mujoco=pytest.importorskip('mujoco')
    # Read the actual acceptance script's MJCF in a side-effect-free way.
    from pathlib import Path
    import runpy
    script=Path(__file__).resolve().parents[1]/'tools/validate_v020_mujoco_moving_door.py'
    scope=runpy.run_path(str(script),run_name='onda_native_mujoco_test')
    model=mujoco.MjModel.from_xml_string(scope['MJCF'])
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    tmat=scope['DOOR_TL']
    from onda_hearing.acoustic_scene.physical_adapters import surface_from_mujoco_box
    initial=surface_from_mujoco_box(model,data,'door_panel',tmat)
    collision=AcousticGeometry((initial,))
    source=(-1.45,-.20,1.2); mic=(1.45,.12,1.35)
    assert len(collision.occluders(source,mic))==1
    # In native MuJoCo, set hinge angle and recompute forward kinematics.
    idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'door_hinge')
    data.qpos[model.jnt_qposadr[idx]]=1.45
    mujoco.mj_forward(model,data)
    moved=surface_from_mujoco_box(model,data,'door_panel',tmat)
    collision.put(moved)
    assert len(collision.occluders(source,mic))==0


def test_geometry_change_at_absolute_sample_chunk_invariant():
    x=(.035*np.sin(2*np.pi*487.*np.arange(SR*2)/SR)).astype(np.float32)
    change_at=SR//2
    def experiment(chunk):
        r,g=make_scene(signal=x)
        result=[]
        while r.next_sample<change_at:
            n=min(chunk,change_at-r.next_sample)
            result.append(r.render(r.next_sample,n))
        g.put(wall_y(2.,xmin=20.,xmax=30.))
        while r.next_sample<SR*2:
            n=min(chunk,SR*2-r.next_sample)
            result.append(r.render(r.next_sample,n))
        return np.concatenate(result)
    np.testing.assert_allclose(experiment(129),experiment(2048),rtol=2e-5,atol=2e-6)


def test_native_reflection_path_state_is_bounded():
    r,_=make_scene()
    _=r.render(0,256)
    telem=r.telemetry()
    assert telem['path_states']==4
    assert telem['active_specular_paths_at_last_update']==4
    assert telem['overrange_samples']==0
    assert telem['clipped_samples']==0
