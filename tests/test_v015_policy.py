"""Simulator-free regression tests for Onda v0.15 policy observation ABI."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from onda_hearing.policy import (
    AcousticEvidence, HearingObservationBank, PolicyHearingConfig,
    PolicyHearingEncoder, POLICY_FEATURES,
)
from onda_hearing.simulators.mujoco_policy import MuJoCoPolicyBridge


def event(t, deg, score=0.8, accepted=True, frame='world'):
    return AcousticEvidence(t, math.radians(deg) if deg is not None else None,
                            score, frame, accepted)


def test_feature_contract_and_bearing_wrap():
    assert len(POLICY_FEATURES) == 5
    encoder = PolicyHearingEncoder()
    encoder.ingest(event(100, 90))
    np.testing.assert_allclose(encoder.read(100), [1, 0, .8, 1, 0], atol=1e-6)
    assert encoder.read(100).dtype == np.float32
    encoder.ingest(event(200, 180))
    np.testing.assert_allclose(encoder.read(200)[:2], [0, -1], atol=1e-6)


def test_silence_rejection_and_age():
    e = PolicyHearingEncoder()
    np.testing.assert_array_equal(e.read(0), np.zeros(5, np.float32))
    e.ingest(event(100_000_000, 30, 0.8))
    assert e.read(350_000_000)[3] == 1
    assert e.read(401_000_000)[3] == 0
    assert e.read(401_000_000)[4] == pytest.approx(.3)
    e.ingest(event(500_000_000, None, accepted=False))
    assert e.read(500_000_000)[3] == 0
    assert not np.any(e.read(500_000_000)[:4])
    e.reset()
    np.testing.assert_array_equal(e.read(0), np.zeros(5, np.float32))


def test_fails_closed_on_clock_and_frame_errors():
    e = PolicyHearingEncoder()
    with pytest.raises(ValueError):
        e.ingest(event(100, 10, frame='robot'))
    e.ingest(event(100, 10))
    with pytest.raises(ValueError):
        e.ingest(event(100, 10))
    with pytest.raises(ValueError):
        e.read(99)
    with pytest.raises(ValueError):
        e.ingest(event(90, 10))
    for score in (float('nan'), float('inf'), -1):
        with pytest.raises(ValueError):
            event(200, 10, score)
    with pytest.raises(ValueError):
        event(200, None, accepted=True)
    with pytest.raises(ValueError):
        PolicyHearingConfig(max_age_s=0)


def test_independent_batch_state_and_partial_reset():
    bank = HearingObservationBank(3)
    bank.ingest(0, event(0, 30))
    bank.ingest(2, event(0, -40))
    result = bank.read([100, 100, 100])
    assert result.shape == (3, 5)
    assert list(result[:, 3]) == [1, 0, 1]
    assert result[0, 0] > 0 and result[2, 0] < 0
    bank.reset([0])
    next_result = bank.read(200)
    assert list(next_result[:, 3]) == [0, 0, 1]
    with pytest.raises(IndexError):
        bank.ingest(3, event(1, 5))
    with pytest.raises(ValueError):
        bank.read([1])


def test_existing_hearing_observation_and_ambiguous_rejection():
    def old_obs(scores):
        capture = SimpleNamespace(end_ns=100)
        directions = [SimpleNamespace(azimuth_rad=0.2, frame_id='world', score=s)
                      for s in scores]
        spatial = SimpleNamespace(capture=capture, frame_id='world', directions=directions)
        return SimpleNamespace(capture=capture, spatial=spatial)
    e = PolicyHearingEncoder()
    e.ingest_hearing(old_obs([.8, .2]))
    assert e.read(100)[3] == 1
    e.reset()
    e.ingest_hearing(old_obs([.8, .75]))
    assert e.read(100)[3] == 0
    e.reset()
    e.ingest_hearing(old_obs([]))
    assert e.read(100)[3] == 0


def test_loop_updates_only_on_new_decision():
    e = PolicyHearingEncoder()
    row = SimpleNamespace(time_ns=100, estimated_bearing_deg=45,
        evidence_score=.6, acoustic_reason='acoustic_bearing_accepted',
        new_acoustic_decision=True)
    e.ingest_loop(row)
    assert e.read(100)[3] == 1
    row.new_acoustic_decision = False
    e.ingest_loop(row)
    assert e.read(200)[3] == 1
    row.new_acoustic_decision = True
    row.time_ns = 300
    row.acoustic_reason = 'ambiguous_spatial_candidates'
    e.ingest_loop(row)
    assert e.read(300)[3] == 0


def test_mujoco_bridge_runtime_ownership_and_rewind():
    class DummyRuntime:
        def __init__(self): self.calls = 0; self.resets = 0
        def reset(self): self.resets += 1
        def advance(self, snapshot):
            self.calls += 1
            return None
    class DummyPose:
        def snapshot(self, time_ns): return time_ns
    runtime = DummyRuntime()
    bridge = MuJoCoPolicyBridge(runtime, DummyPose())
    np.testing.assert_array_equal(bridge.observe(0), np.zeros(5, np.float32))
    np.testing.assert_array_equal(bridge.observe(10), np.zeros(5, np.float32))
    with pytest.raises(ValueError): bridge.observe(10)
    assert runtime.calls == 2
    bridge.reset()
    assert runtime.resets == 1
    bridge.observe(0)
