"""Contract tests covering actual Onda audio renderer + motion SRP."""
import math
import numpy as np
import pytest

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.policy import (
    SemanticEvent, SemanticResult, SemanticStatus, SemanticPolicyConfig,
    SemanticPolicyEncoder, WindowedHearingPipeline, WindowedPipelineConfig,
    PolicyHearingConfig, AuditoryObservation,
)
from onda_hearing.runtime import HearingRuntime, SceneSnapshot, ScheduledSource
from onda_hearing.experiments.auditory_control import AcousticOnlySelector, AcousticSelectionConfig
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig


def test_semantic_capability_masks():
    enc = SemanticPolicyEncoder(SemanticPolicyConfig(('telephone_ring','glass_break','alarm')))
    unsupported = SemanticResult(SemanticStatus.UNSUPPORTED)
    empty = SemanticResult(SemanticStatus.NO_DETECTION, backend_id='test_backend')
    event = SemanticEvent('telephone_ring', 0.7, 100, 200, 'test_backend',
                          embedding=(0.1, -0.2))
    detected = SemanticResult(SemanticStatus.DETECTED, (event,), backend_id='test_backend')
    assert enc.feature_names == ('semantic_enabled','semantic_detected',
                                 'class:telephone_ring','class:glass_break','class:alarm')
    np.testing.assert_array_equal(enc.encode(unsupported), [0,0,0,0,0])
    np.testing.assert_array_equal(enc.encode(empty), [1,0,0,0,0])
    np.testing.assert_allclose(enc.encode(detected), [1,1,.7,0,0], atol=1e-7)
    assert enc.encode(detected).dtype == np.float32
    assert SemanticEvent('ring', .7, 100, 200, 'test', offset_ns=700).duration_s == pytest.approx(0.0000006)


def test_semantic_contract_rejections():
    with pytest.raises(ValueError):
        SemanticEvent('glass_break', float('nan'), 1, 2, 'test')
    with pytest.raises(ValueError):
        SemanticEvent('glass_break', .5, 2, 1, 'test')
    with pytest.raises(ValueError):
        SemanticEvent('glass_break', .5, 1, 2, 'test', embedding=(float('inf'),))
    with pytest.raises(ValueError):
        SemanticResult(SemanticStatus.UNSUPPORTED, backend_id='model')
    with pytest.raises(ValueError):
        SemanticResult(SemanticStatus.DETECTED, (), backend_id='model')
    with pytest.raises(ValueError):
        SemanticResult(SemanticStatus.NO_DETECTION, (SemanticEvent('a', .5, 1, 2, 'model'),), backend_id='model')


def make_pipeline(signal):
    array = MicrophoneArray.from_positions(
        [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
        array_id='head_array', frame_id='robot/head', geometry_id='asym90mm_test')
    runtime = HearingRuntime(array,(ScheduledSource('beacon',signal,0),),sample_rate_hz=16000)
    selector = AcousticOnlySelector(MotionAwareSRPLocalizer(MotionSRPConfig(
        block_samples=320, hop_samples=160, angular_resolution_deg=2,
        max_candidates=3, min_relative_score=.65, min_prominence=.012)),
        AcousticSelectionConfig(min_score=.08, ambiguous_ratio=.88, min_consistent_windows=2))
    return WindowedHearingPipeline(runtime, selector)


def snapshot(time_ns, bearing=38.):
    a = math.radians(bearing)
    return SceneSnapshot(time_ns, (0.,0.,1.), (0.,0.,0.,1.),
                         {'beacon': (1.5*math.cos(a),1.5*math.sin(a),1.)})


def test_pipeline_zero_audio_silence_and_reset():
    pipeline = make_pipeline(np.zeros(16000, dtype=np.float32))
    f = pipeline.advance(snapshot(0))
    np.testing.assert_array_equal(f.features, np.zeros(5, dtype=np.float32))
    assert f.audio_samples == 0
    got = []
    for n in range(1, 26):
        f = pipeline.advance(snapshot(n * 20_000_000))
        got.append(f)
    assert pipeline.decisions > 0
    assert pipeline.accepted == 0
    assert all(x.features[3] == 0 for x in got)
    assert sum(x.audio_samples for x in got) == 8000
    assert all(x.observation is None or not x.observation.spatial.directions for x in got)
    with pytest.raises(ValueError, match='reset'):
        pipeline.advance(snapshot(0))
    pipeline.reset()
    assert pipeline.advance(snapshot(0)).features[3] == 0
    assert pipeline.decisions == 0


def test_pipeline_rejects_duplicate_clocks_and_double_perception():
    pipeline = make_pipeline(np.zeros(16000, dtype=np.float32))
    pipeline.advance(snapshot(0))
    with pytest.raises(ValueError):
        pipeline.advance(snapshot(0))
    with pytest.raises(ValueError):
        WindowedHearingPipeline(pipeline.runtime, pipeline.selector,
            config=WindowedPipelineConfig(frame_id='robot'),
            policy_config=PolicyHearingConfig(frame_id='world'))


def test_native_mujoco_policy_optional():
    pytest.importorskip('mujoco')
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
    from v015_mujoco_policy_observations import run_episode
    assert run_episode(mode='silence', duration_s=1.0, strict=True)['status'] == 'PASS'


def test_source_tracking_policy_padding_and_freshness():
    from onda_hearing.policy import SourcePolicyEncoder, SourcePolicyConfig
    from onda_hearing.contracts.tracking import SourceTrack, TrackState
    enc = SourcePolicyEncoder(SourcePolicyConfig(max_sources=3, max_age_s=.5))
    tracks = (
        SourceTrack('a', 'world', math.radians(90), TrackState.CONFIRMED, 900_000_000),
        SourceTrack('b', 'world', math.radians(-90), TrackState.TENTATIVE, 950_000_000),
    )
    output = enc.encode(tracks, now_ns=1_000_000_000, frame_id='world')
    assert output.shape == (3, 7) and output.dtype == np.float32
    np.testing.assert_allclose(output[0, [0, 1, 4, 5, 6]], [1,0,.1,1,1], atol=1e-6)
    assert output[1, 0] == -1 and output[1, 5] == 0 and output[1, 6] == 1
    np.testing.assert_array_equal(output[2], 0)
    assert not enc.encode(tracks, now_ns=2_000_000_000, frame_id='world')[:,6].any()
    with pytest.raises(ValueError):
        enc.encode(tracks, now_ns=1, frame_id='world')


def test_shared_mujoco_bridge_on_real_audio_runtime_without_mujoco_dependency():
    from onda_hearing.simulators.mujoco_policy import WindowedMuJoCoPolicyBridge
    class PoseAdapter:
        def snapshot(self, time_ns):
            return snapshot(time_ns)
    pipeline = make_pipeline(np.zeros(16000, dtype=np.float32))
    bridge = WindowedMuJoCoPolicyBridge(pipeline.runtime, PoseAdapter(), pipeline.selector)
    assert bridge.observe(0).features.shape == (5,)
    f = bridge.observe(20_000_000)
    assert f.audio_samples == 320 and not f.new_decision
    with pytest.raises(ValueError):
        bridge.observe(20_000_000)
    bridge.reset()
    assert bridge.observe(0).audio_samples == 0
