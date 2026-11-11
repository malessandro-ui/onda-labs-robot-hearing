"""Reference-equivalence checks for the v0.20 opt-in scalar ray backend."""
import math
import numpy as np
import pytest

from onda_hearing.acoustic_scene.physical import (
    AcousticGeometry, AcousticSurface, TransmissionMaterial,
)
from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry

TL = TransmissionMaterial((125., 8000.), (15., 30.), 'synthetic_test_only')


def quad_x(x, name):
    return AcousticSurface.quad(name, ((x, -2., -2.), (x, 2., -2.),
                                      (x, 2., 2.), (x, -2., 2.)), TL)


def quad_y(y, name):
    return AcousticSurface.quad(name, ((-2., y, -2.), (2., y, -2.),
                                      (2., y, 2.), (-2., y, 2.)), TL)


def ids(geo, a, b):
    return tuple(s.surface_id for s in geo.occluders(a, b))


def test_surface_intersections_randomized_equivalent():
    surfaces = (quad_x(0., 'door'), quad_y(.4, 'panel'),
                quad_x(1.8, 'back_wall'), quad_y(-1.8, 'front_wall'))
    old, fast = AcousticGeometry(surfaces), FastAcousticGeometry(surfaces)
    rng = np.random.default_rng(20261110)
    for _ in range(3000):
        a = rng.uniform(-3., 3., 3)
        b = rng.uniform(-3., 3., 3)
        assert ids(old, a, b) == ids(fast, a, b)


def test_tolerance_and_endpoint_cases():
    surf = quad_x(0., 'wall')
    old, fast = AcousticGeometry((surf,)), FastAcousticGeometry((surf,))
    rays = [
        ((-1., 0., 0.), (1., 0., 0.)),    # classic crossing
        ((1., 0., 0.), (-1., 0., 0.)),    # back face
        ((-1., 2., 0.), (1., 2., 0.)),    # boundary
        ((-1., 3., 0.), (1., 3., 0.)),    # outside
        ((0., 0., 0.), (1., 0., 0.)),     # start on surface
        ((-1., 0., 0.), (0., 0., 0.)),    # end on surface
        ((-1., 0., 0.), (-1., 0., 0.)),  # zero length
        ((0., -1., 0.), (0., 1., 0.)),    # coplanar
        ((-1.e-10, 0., 0.), (1., 0., 0.)),
        ((-1., 1.e-7, 1.e-7), (1., 1.e-7, 1.e-7)),
    ]
    for a, b in rays:
        assert ids(old, a, b) == ids(fast, a, b), (a, b)


def test_dynamic_replace_remove_enable_and_order():
    a = quad_x(0., 'a')
    b = quad_x(1., 'b')
    old, fast = AcousticGeometry((a, b)), FastAcousticGeometry((a, b))
    start, end = (-1., 0., 0.), (2., 0., 0.)
    for ref, opt in ((old, fast),):
        assert ids(ref, start, end) == ids(opt, start, end) == ('a', 'b')
        opt.put(quad_x(1.5, 'b'))
        ref.put(quad_x(1.5, 'b'))
        assert ids(ref, start, end) == ids(opt, start, end)
        disabled = AcousticSurface('a', a.vertices_m, a.triangles, TL, enabled=False)
        ref.put(disabled)
        opt.put(disabled)
        assert ids(ref, start, end) == ids(opt, start, end) == ('b',)
        ref.remove('a')
        opt.remove('a')
        assert ids(ref, start, end) == ids(opt, start, end) == ('b',)
        assert ref.revision == opt.revision


def test_input_validation_parity():
    old, fast = AcousticGeometry((quad_x(0., 'a'),)), FastAcousticGeometry((quad_x(0., 'a'),))
    bad = ([0., 0.], [float('nan'), 0., 0.], [0., 0., float('inf')])
    for p in bad:
        for geom in (old, fast):
            with pytest.raises(ValueError):
                geom.occluders(p, (1., 2., 3.))
    for geom in (old, fast):
        with pytest.raises(ValueError):
            geom.occluders((0., 0., 0.), (float('nan'), 1., 1.))


def test_optimized_is_subclass_and_returns_original_surface():
    wall = quad_y(0., 'wall')
    geo = FastAcousticGeometry((wall,))
    assert isinstance(geo, AcousticGeometry)
    result = geo.occluders((0., -1., 0.), (0., 1., 0.))
    assert len(result) == 1 and result[0] is wall


def test_rendered_pcm_equivalence():
    from onda_hearing.acoustic_scene.types import Pose3, Binding, Schedule, Emitter, MicArray
    from onda_hearing.acoustic_scene.assets import AudioAsset
    from onda_hearing.acoustic_scene.renderer import RendererConfig
    from onda_hearing.acoustic_scene.unified_physics import (
        UnifiedGeometryAcousticRenderer, UnifiedPhysicsConfig, ReflectionMaterial,
    )
    sr = 16000
    t = np.arange(2048) / sr
    x = (.04 * np.sin(2 * np.pi * 440 * t) +
         .03 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    wall = quad_y(1., 'reflector')
    door = quad_x(1.5, 'door')
    mic = MicArray(Binding(world=Pose3((2., 0., 0.))),
                   ((0., 0., 0.), (0., .035, 0.), (0., -.035, 0.), (.035, 0., 0.)))
    emitter = Emitter('phone', 'test', Binding(world=Pose3((0., 0., 0.))),
                      Schedule(onset_s=0., loop=True))
    refl = ReflectionMaterial((125., 8000.), (.3, .3), (.0, .0), 'synthetic_test_only')
    scenes = []
    for geom in (AcousticGeometry((wall, door)), FastAcousticGeometry((wall, door))):
        renderer = UnifiedGeometryAcousticRenderer(
            RendererConfig(sample_rate=sr), mic, [emitter], {'test': AudioAsset('test', x, sr)},
            geom, {'reflector': refl},
            unified_config=UnifiedPhysicsConfig(max_specular_order=1, geometry_step_samples=256),
        )
        chunks = []
        for pos in range(0, 2048, 256):
            if pos == 1024:
                geom.put(quad_x(3., 'door'))
            chunks.append(renderer.render(pos, 256))
        scenes.append(np.concatenate(chunks))
    assert scenes[0].shape == scenes[1].shape == (2048, 4)
    np.testing.assert_allclose(scenes[0], scenes[1], atol=1.e-6, rtol=1.e-6)
