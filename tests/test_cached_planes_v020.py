"""Tests for v0.20.2 safe plane cache and exact acoustic equivalence."""
from unittest.mock import patch

import numpy as np
import pytest

from onda_hearing.acoustic_scene import (Pose3, Binding, Schedule, Emitter,
    MicArray, AudioAsset, RendererConfig)
from onda_hearing.acoustic_scene.physical import (TransmissionMaterial,
    AcousticSurface, AcousticGeometry)
from onda_hearing.acoustic_scene import unified_physics as up
from onda_hearing.acoustic_scene.cached_planes_v020 import make_cached_plane

SR = 16000
TL = TransmissionMaterial((125., 8000.), (35., 35.), 'synthetic test')
REFL = up.ReflectionMaterial((125., 8000.), (.2, .2), (0., 0.), 'synthetic test')


def wall(y=2., sid='wall'):
    return AcousticSurface.quad(sid, ((-3.,y,0.),(3.,y,0.),
                                          (3.,y,3.),(-3.,y,3.)),TL)


def make_renderer(order=1):
    mic = MicArray(Binding(world=Pose3((2., 0., 1.))),
                   ((0.,0.,0.),(0.,.04,0.),(0.,-.04,0.),(.04,0.,0.)))
    e = Emitter('one','clip',Binding(world=Pose3((1.,0.,1.))),
                Schedule(onset_s=0.))
    t = np.arange(6000,dtype=float)/SR
    x = (.04*np.sin(2*np.pi*440*t)+.03*np.sin(2*np.pi*1800*t)).astype(np.float32)
    a = wall()
    b = AcousticSurface.quad('end', ((-3.,-2.,0.),(3.,-2.,0.),
                                           (3.,-2.,3.),(-3.,-2.,3.)),TL)
    scene=AcousticGeometry((a,b))
    return up.UnifiedGeometryAcousticRenderer(
        RendererConfig(sample_rate=SR),mic,[e],{'clip':AudioAsset('clip',x,SR)},
        scene,{'wall':REFL,'end':REFL},
        unified_config=up.UnifiedPhysicsConfig(max_specular_order=order,
            geometry_step_samples=256,transition_samples=256)),scene


def test_cache_hits_and_identical_values():
    cached=make_cached_plane(up._plane, maxsize=8)
    original=up._plane(wall())
    for _ in range(50):
        actual=cached(wall())
        np.testing.assert_array_equal(actual[0],original[0])
        np.testing.assert_array_equal(actual[1],original[1])
        assert actual[0].flags.writeable is False
        assert actual[1].flags.writeable is False
    assert cached.cache_info().hits >= 49
    assert cached.cache_info().misses == 1


def test_geometry_change_same_id_does_not_reuse_old_plane():
    cached=make_cached_plane(up._plane)
    before=cached(wall(2.,'same'))
    after=cached(wall(3.,'same'))
    assert cached.cache_info().misses == 2
    np.testing.assert_array_equal(after[0],up._plane(wall(3.,'same'))[0])
    assert float(before[0][1]) != float(after[0][1])


def test_cache_is_bounded_and_rejects_invalid_sizes():
    for size in (0,-1,9000,1.5):
        with pytest.raises((ValueError,TypeError)):
            make_cached_plane(up._plane,maxsize=size)
    cached=make_cached_plane(up._plane,maxsize=7)
    for i in range(25):
        cached(wall(float(i+1),str(i)))
    assert cached.cache_info().currsize == 7


def test_invalid_nonplanar_surface_still_rejected():
    surface=AcousticSurface('nonplanar',((0.,0.,0.),(1.,0.,0.),
                                           (0.,1.,0.),(1.,1.,.2)),
                            ((0,1,2),(1,3,2)),TL)
    cached=make_cached_plane(up._plane)
    with pytest.raises(ValueError,match='planar'):
        cached(surface)


@pytest.mark.parametrize('order',[1,2])
def test_original_and_cached_pcm_identical_with_moving_reflector(order):
    def run(cache):
        renderer,geo=make_renderer(order=order)
        out=[]
        for index in range(12):
            if index == 5:
                geo.put(wall(2.15))
            if index == 8:
                geo.put(wall(2.))
            at=index*256
            out.append(renderer.render(at,256))
        return np.concatenate(out), renderer.telemetry()
    original_pcm,old_report=run(False)
    accelerated=make_cached_plane(up._plane)
    with patch.object(up,'_plane',accelerated):
        optimized_pcm,new_report=run(True)
    np.testing.assert_array_equal(original_pcm,optimized_pcm)
    assert old_report['active_specular_paths_at_last_update'] == new_report['active_specular_paths_at_last_update']
    assert accelerated.cache_info().hits > 10
