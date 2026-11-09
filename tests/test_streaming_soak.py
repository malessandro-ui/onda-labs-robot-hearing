"""Regression tests for sustained-streaming validation infrastructure."""
import math
import numpy as np
import pytest

from onda_hearing.streaming import StreamConfig, PCMBuffer, GCCPHATSRP, Pose, PoseHistory
from onda_hearing.streaming.simulated_capture import DEFAULT_ARRAY_METERS
from tools.onda_soak_common import CyclicSource, LatencyHistogram, angle_error, sample_timestamp_ns, yaw_at_seconds, yaw_xyzw, Counter


def test_absolute_timestamp_stable_over_30_minute_horizon():
    t0=1_000_000_000_000
    assert sample_timestamp_ns(t0, 16_000*1800, 16_000)==t0+1800_000_000_000
    assert sample_timestamp_ns(t0, 16_000*1800+512, 16_000)==t0+1800_032_000_000


def test_fixed_memory_histogram_with_long_samples():
    h=LatencyHistogram(bin_ms=.1,max_ms=100.)
    for i in range(100_000):h.add((i%100)/10.)
    assert h.count==100_000
    assert h.percentile(50)==pytest.approx(5.,abs=.1)
    assert h.percentile(99)==pytest.approx(9.9,abs=.1)
    h.add(120.)
    assert h.overflow==1
    assert h.maximum==120.


def test_cyclic_source_moving_microphone_bearing():
    src=CyclicSource(seed=12)
    loc=GCCPHATSRP(DEFAULT_ARRAY_METERS)
    radians=math.radians(38.)
    target=(1.5*math.cos(radians),1.5*math.sin(radians),0.)
    for angle in (0,30,-30,45):
        pcm=src.render(20000,2048,16000,(0.,0.,0.),yaw_xyzw(angle),target,DEFAULT_ARRAY_METERS)
        result=loc.estimate(pcm)
        assert result.valid
        assert angle_error(result.local_bearing_deg,38.-angle)<6.


def test_bounded_ring_long_run_eviction_is_not_discontinuity():
    # Exercise 10 minutes of source sample indices without accumulating audio.
    cfg=StreamConfig(sample_rate=2000,channels=4,window_samples=256,hop_samples=128,buffer_seconds=.5)
    ring=PCMBuffer(cfg,0)
    chunk=np.ones((200,4),dtype=np.float32)
    for k in range(6000):
        ring.append(chunk,k*200)
    oldest,next_sample,epoch=ring.bounds()
    assert next_sample==1_200_000
    assert next_sample-oldest==1000
    assert epoch==0 and ring.discontinuities==0
    assert ring.overwritten_samples==1_199_000
    assert ring.window(next_sample).samples.shape==(256,4)


def test_explicit_discontinuity_is_not_normal_ring_eviction():
    cfg=StreamConfig(sample_rate=2000,channels=4,window_samples=256,hop_samples=128,buffer_seconds=.5)
    ring=PCMBuffer(cfg,0)
    ring.append(np.zeros((512,4),dtype=np.float32),0)
    ring.append(np.zeros((512,4),dtype=np.float32),1024)
    assert ring.discontinuities==1
    assert ring.bounds()[2]==1
    assert ring.window(512) is None


def test_yaw_schedule_continuous():
    assert yaw_at_seconds(0)==pytest.approx(0)
    assert yaw_at_seconds(6)==pytest.approx(50)
    assert yaw_at_seconds(12)==pytest.approx(0,abs=1e-9)
    assert angle_error(-179,179)==2


def test_reference_sample_window_pose_interpolation():
    poses=PoseHistory()
    poses.add(Pose(0,(0.,0.,0.),yaw_xyzw(0)))
    poses.add(Pose(1_000_000_000,(0.,0.,0.),yaw_xyzw(50)))
    halfway=poses.at(500_000_000,1_000_000_000)
    assert halfway is not None
    assert halfway.quaternion_xyzw[2] == pytest.approx(yaw_xyzw(25)[2],abs=1e-5)


def test_counter_only_counts_unique_bearing_decisions():
    from onda_hearing.streaming import HearingObservation
    c=Counter()
    o=HearingObservation(valid=True,bearing_world_deg=39.,source_sample_end=2048)
    for i in range(9):c.track(o,38.,10.)
    assert c.valid==9 and c.unique_valid==1 and c.bearing_errors==0
    c.track(HearingObservation(valid=False,reason='stale'),38.,10.)
    assert c.stale==1 and c.invalid==1
