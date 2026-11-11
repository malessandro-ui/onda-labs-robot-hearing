"""Independent physical and integration tests of Onda v0.19 geometry backend."""
import math
import numpy as np
import pytest

from onda_hearing.acoustic_scene import (
    Pose3, Binding, Schedule, Emitter, MicArray, AudioAsset,
    RendererConfig, AcousticSceneRenderer,
)
from onda_hearing.acoustic_scene.physical import (
    TransmissionMaterial, AcousticSurface, AcousticGeometry,
    GeometryAwareAcousticRenderer, PhysicalConfig,
    minimum_phase_transmission,
)

SR = 16000
MAT = TransmissionMaterial((200., 1000., 4000., 8000.),
                           (6., 12., 28., 36.), "synthetic_test_not_measured")


def door(orientation="closed"):
    if orientation == "closed":
        corners = ((0., -0.8, 0.), (0., 0.8, 0.),
                   (0., 0.8, 2.1), (0., -0.8, 2.1))
    elif orientation == "open":
        # 90-degree door swing about hinge at (0,-0.8,z).
        corners = ((0., -0.8, 0.), (-1.6, -0.8, 0.),
                   (-1.6, -0.8, 2.1), (0., -0.8, 2.1))
    else:
        raise ValueError(orientation)
    return AcousticSurface.quad("door", corners, MAT)


def scene(n=32000, signal=None, geometry=None, config=None, phys=None):
    x = (0.07*np.sin(2*np.pi*440*np.arange(n)/SR)
         + 0.04*np.sin(2*np.pi*3200*np.arange(n)/SR)) if signal is None else signal
    x = np.asarray(x, dtype=np.float32)
    asset = AudioAsset("tone", x, SR)
    emit = Emitter("phone", "tone", Binding(world=Pose3((-1.5, 0., 1.2))),
                   Schedule(onset_s=0.))
    mics = MicArray(Binding(world=Pose3((1.5, 0., 1.2))),
                    ((0., 0., 0.), (0., .04, 0.), (0., -.04, 0.), (.04, 0., 0.)))
    cfg = config or RendererConfig(sample_rate=SR)
    g = geometry or AcousticGeometry((door(),))
    renderer = GeometryAwareAcousticRenderer(cfg, mics, [emit],
                                            {"tone": asset}, g,
                                            physical_config=phys or PhysicalConfig())
    return renderer, g, mics, emit, asset


def stream(r, total, chunk):
    blocks = []
    cursor = 0
    while cursor < total:
        k = min(chunk, total-cursor)
        blocks.append(r.render(cursor, k))
        cursor += k
    return np.concatenate(blocks)


def band_amplitude(x, freq, offset):
    y = np.asarray(x, dtype=np.float64)[offset:offset+SR//2]
    n = np.arange(len(y))
    return abs(2./len(y) * np.dot(y, np.exp(-2j*np.pi*freq*n/SR)))


def test_rejects_unsupported_material_curves():
    with pytest.raises(ValueError):
        TransmissionMaterial((500, 300), (5, 10), "test")
    with pytest.raises(ValueError):
        TransmissionMaterial((300, 500), (-1, 10), "test")
    with pytest.raises(ValueError):
        TransmissionMaterial((300, 500), (5, 10), "")


def test_surface_geometry_validation():
    with pytest.raises(ValueError):
        AcousticSurface.quad("bad", ((0, 0, 0), (1, 0, 0),
                                     (1, 1, 0), (0, 1, 1)), MAT)
    with pytest.raises(ValueError):
        AcousticSurface("bad", ((0,0,0),(1,0,0),(2,0,0)), ((0,1,2),), MAT)


def test_closed_door_hits_and_open_door_does_not():
    geometry = AcousticGeometry((door(),))
    a, b = (-1., 0., 1.), (1., 0., 1.)
    assert [s.surface_id for s in geometry.occluders(a, b)] == ["door"]
    geometry.put(door("open"))
    assert geometry.occluders(a, b) == ()


def test_shared_triangle_edge_does_not_double_attenuate():
    geometry = AcousticGeometry((door(),))
    a, b = (-1., 0., 1.05), (1., 0., 1.05)
    assert len(geometry.occluders(a, b)) == 1


def test_minphase_fir_is_finite_and_identity_when_clear():
    clear = minimum_phase_transmission((), SR)
    assert clear[0] == 1. and np.count_nonzero(clear) == 1
    loss = minimum_phase_transmission((MAT,), SR)
    assert len(loss) == 129 and np.isfinite(loss).all()
    freqs = np.fft.rfftfreq(16384, 1/SR)
    amp = abs(np.fft.rfft(loss, 16384))
    for center, dB in ((440, 7.4), (3200, 24.2)):
        value = float(np.interp(center, freqs, amp))
        measured = -20*math.log10(value)
        assert abs(measured-dB) < 3., (center, measured)


def test_closed_door_muffles_high_frequencies_more():
    closed, _, _, _, _ = scene()
    open_renderer, _, _, _, _ = scene(geometry=AcousticGeometry((door("open"),)))
    y_closed = stream(closed, SR*2, 1600)[:, 0]
    y_open = stream(open_renderer, SR*2, 1600)[:, 0]
    for freq in (440., 3200.):
        a_open = band_amplitude(y_open, freq, SR//2)
        a_closed = band_amplitude(y_closed, freq, SR//2)
        loss = 20*math.log10(max(a_open,1e-12)/max(a_closed,1e-12))
        if freq == 440:
            low_loss = loss
        else:
            high_loss = loss
    assert high_loss > low_loss + 10., (low_loss, high_loss)
    assert low_loss > 2.0


def test_clear_path_matches_original_renderer():
    physical, _, mics, emit, asset = scene(
        geometry=AcousticGeometry(()))
    original = AcousticSceneRenderer(RendererConfig(sample_rate=SR), mics,
                                     [emit], {"tone": asset})
    a = stream(physical, SR, 375)
    b = stream(original, SR, 375)
    np.testing.assert_allclose(a, b, atol=1.e-7, rtol=1.e-6)


def test_chunk_independence_for_static_closed_door():
    a, *_ = scene()
    b, *_ = scene()
    np.testing.assert_allclose(stream(a, SR, 171), stream(b, SR, 4096),
                               atol=1.e-7, rtol=1.e-6)


def test_opening_transition_fades_and_preserves_chunk_independence():
    def sample(chunk):
        renderer, geometry, *_ = scene(n=SR*3)
        first = stream(renderer, SR, chunk)
        geometry.put(door("open"))
        second = []
        cursor = SR
        while cursor < SR*3:
            n = min(chunk, SR*3-cursor)
            second.append(renderer.render(cursor, n))
            cursor += n
        return np.concatenate([first, *second])
    a = sample(512)
    b = sample(3301)
    np.testing.assert_allclose(a, b, atol=1.e-6, rtol=1.e-5)
    before = np.sqrt(np.mean(a[SR//2:SR, 0].astype(float)**2))
    after = np.sqrt(np.mean(a[SR*2:SR*3, 0].astype(float)**2))
    assert after > before*1.5


def test_noise_not_multiplied_per_emitter():
    cfg = RendererConfig(sample_rate=SR, noise_std=.01, seed=123)
    r, _, mics, _, _ = scene(signal=np.zeros(SR, np.float32), config=cfg)
    original = AcousticSceneRenderer(cfg, mics, [], {})
    np.testing.assert_allclose(stream(r, SR, 813), stream(original, SR, 813),
                               atol=1.e-7, rtol=1.e-6)


def test_sequential_and_reset_api():
    r, *_ = scene()
    _ = r.render(0, 256)
    with pytest.raises(ValueError):
        r.render(0, 256)
    with pytest.raises(ValueError):
        r.render(256, SR*3)
    r.reset()
    assert r.next_sample == 0
    assert r.render(0, 256).shape == (256, 4)
    assert r.telemetry()["ray_tested_paths"] == 4


def test_no_legacy_fir_double_counting():
    cfg = RendererConfig(sample_rate=SR, rir_taps=np.ones((2,4), np.float32))
    with pytest.raises(ValueError, match="rir_taps"):
        scene(config=cfg)


def test_optional_pyroomacoustics_static_room():
    pytest.importorskip("pyroomacoustics")
    from onda_hearing.acoustic_scene.room_physics import simulate_shoebox
    impulse = np.zeros(256, np.float32)
    impulse[0] = 0.1
    kw = dict(signals=[impulse], sources_m=np.array([[1., 1., 1.]]),
              microphones_m=np.array([[2., 1., 1.]]),
              dimensions_m=(4., 3., 2.5), sample_rate=SR,
              energy_absorption=.3)
    direct = simulate_shoebox(max_order=0, **kw)
    reflected = simulate_shoebox(max_order=2, **kw)
    assert direct.samples.shape[1] == 1
    assert reflected.samples.shape[1] == 1
    assert np.isfinite(reflected.samples).all()
    # An image-source room should contain strictly more reflected energy.
    idx = int(SR/343.)
    assert np.max(abs(direct.samples[idx-5:idx+6, 0])) > 0
    assert (np.sum(reflected.samples.astype(float)**2) >
            np.sum(direct.samples.astype(float)**2))
    assert reflected.runtime_metadata["clipping_applied"] is False


def test_pyroomacoustics_missing_material_calibration_rejected():
    from onda_hearing.acoustic_scene.room_physics import simulate_shoebox
    with pytest.raises(ValueError, match="explicit energy_absorption"):
        simulate_shoebox([np.zeros(128)], np.array([[1., 1., 1.]]),
                         np.array([[2., 1., 1.]]), (4., 3., 2.5))


def test_convex_face_adapters_accept_quads_and_reject_concave():
    from onda_hearing.acoustic_scene.physical_adapters import triangulate_convex_faces
    square = [(0,0,0),(2,0,0),(2,2,0),(0,2,0)]
    assert len(triangulate_convex_faces(square, [4], [0,1,2,3])) == 2
    concave = [(0,0,0),(2,0,0),(0.6,0.6,0),(0,2,0)]
    with pytest.raises(ValueError, match="nonconvex"):
        triangulate_convex_faces(concave, [4], [0,1,2,3])


def test_legacy_v017_geometry_renderer_unchanged_by_import():
    from onda_hearing.acoustic_scene.physical_adapters import triangulate_convex_faces
    assert hasattr(AcousticSceneRenderer, "render")
