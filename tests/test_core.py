import numpy as np
import pytest
from onda_hearing import (MicrophoneArray, FreeFieldSimulator, SoundSource,
                          AudioBlock, estimate_azimuth, BearingTracker, Direction)
from onda_hearing.tracking import wrap


def test_geometry_rejects_duplicates():
    with pytest.raises(ValueError):
        MicrophoneArray([[0, 0, 0], [0, 0, 0]])


def test_geometry_rank_and_planar_capability():
    a = MicrophoneArray([[0,0,0],[.1,0,0],[0,.1,0]])
    assert a.channels == 3 and a.geometry_rank == 2
    a.validate_azimuth()
    with pytest.raises(ValueError):
        MicrophoneArray([[0,0,0],[.1,0,0]]).validate_azimuth()


def test_audio_contract_immutable():
    b = AudioBlock(np.zeros((20, 3)), 1000, .5, 'array')
    assert b.t1_s == pytest.approx(.52)
    with pytest.raises(ValueError):
        b.pcm[0, 0] = 1


def test_propagation_delays_and_additivity():
    sr = 24000
    a = MicrophoneArray([[0,0,0],[.15,0,0],[0,.1,0]])
    click = np.zeros(2048); click[0] = .1
    source = SoundSource('click', click, [2,0,0], start_s=.02)
    sim = FreeFieldSimulator(sr)
    x = sim.render(a, np.zeros(3), [source], .12)
    assert np.argmax(np.abs(x.pcm[:,1])) < np.argmax(np.abs(x.pcm[:,0]))
    double = sim.render(a, np.zeros(3), [source, source], .12)
    np.testing.assert_allclose(double.pcm, 2*x.pcm, atol=1e-8)


def test_noise_seed_reproducible():
    sim = FreeFieldSimulator(1000)
    a = MicrophoneArray([[0,0,0]])
    x = sim.render(a, np.zeros(3), [], .1, noise_std=.01, seed=123)
    y = sim.render(a, np.zeros(3), [], .1, noise_std=.01, seed=123)
    np.testing.assert_array_equal(x.pcm, y.pcm)


def test_planar_azimuth_impulse():
    sr = 48000
    a = MicrophoneArray([[-.08,-.08,0],[.08,-.08,0],[.08,.08,0],[-.08,.08,0]])
    rng = np.random.default_rng(11)
    impulse = rng.normal(0,.1,10000)
    x = FreeFieldSimulator(sr).render(a, np.zeros(3),
               [SoundSource('broadband', impulse, [3, 1.5, 0], .05)], .36)
    candidates = estimate_azimuth(x,a,angle_step_deg=2)
    assert candidates
    target = np.arctan2(1.5, 3)
    assert abs(wrap(candidates[0].azimuth_rad - target)) < np.deg2rad(9)


def test_tracks_id_persists_and_wraparound():
    t = BearingTracker(confirm_hits=2)
    a = t.update((Direction(np.deg2rad(179), 1., 'a'),), .1)
    b = t.update((Direction(np.deg2rad(-179), 1., 'a'),), .2)
    assert a[0].track_id == b[0].track_id and b[0].confirmed
    with pytest.raises(ValueError):
        t.update((), .2)


def test_tracker_frames_rejected():
    t = BearingTracker()
    t.update((Direction(0,1,'a'),),1)
    with pytest.raises(ValueError):
        t.update((Direction(0,1,'b'),),2)
