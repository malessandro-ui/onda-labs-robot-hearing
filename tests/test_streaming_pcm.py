import math
import time
import numpy as np
import pytest
from onda_hearing.streaming import (StreamConfig,Pose,PoseHistory,PCMBuffer,GCCPHATSRP,
                                    BearingEstimate,StreamingHearing,rotate_xyzw,StreamError)
from onda_hearing.streaming.simulated_capture import render_point_source,DEFAULT_ARRAY_METERS

CFG=StreamConfig(window_samples=1024,hop_samples=256,buffer_seconds=.4)


def yaw(angle):
    a=math.radians(angle)/2
    return (0.,0.,math.sin(a),math.cos(a))


def pose(t_ns,yaw_deg=0):
    return Pose(t_ns,(0.,0.,0.),yaw(yaw_deg))


def test_ring_exact_samples_overflow_and_discontinuity():
    ring=PCMBuffer(CFG,123456789)
    src=np.arange(5120,dtype=np.float32).reshape(-1,1).repeat(4,axis=1)
    for k in range(10):ring.append(src[k*512:(k+1)*512],k*512)
    w=ring.window(5120)
    assert w.samples.shape==(1024,4)
    np.testing.assert_array_equal(w.samples[:,0],np.arange(4096,5120))
    assert w.capture_mid_ns == ring.time_ns(4608)
    assert not w.samples.flags.writeable
    ring.append(np.zeros((1500,4),dtype=np.float32),5120)
    assert ring.overwritten_samples==220
    assert ring.window(1024) is None
    with pytest.raises(StreamError):ring.append(np.zeros((100,4)),6100)
    ring.append(np.zeros((1200,4)),7000)
    assert ring.epoch==1 and ring.discontinuities==1
    assert ring.window(7001) is None


def test_pose_slerp_and_no_extrapolation():
    history=PoseHistory()
    history.add(pose(0,0)); history.add(pose(100_000_000,90))
    midpoint=history.at(50_000_000,120_000_000)
    v=rotate_xyzw(midpoint.quaternion_xyzw,(1,0,0))
    assert math.degrees(math.atan2(v[1],v[0]))==pytest.approx(45,abs=.001)
    assert history.at(101_000_000,120_000_000) is None
    assert history.at(50_000_000,10_000_000) is None
    with pytest.raises(StreamError):history.add(pose(100_000_000,100))


def test_gcc_phat_azimuth_from_multichannel_pcm():
    rng=np.random.default_rng(22)
    mono=rng.standard_normal(10000)
    for degrees in (-68,-38,8,38,68):
        angle=math.radians(degrees)
        src=1.5*np.array([math.cos(angle),math.sin(angle),0])
        pcm=render_point_source(mono,2000,2048,16000,(0,0,0),yaw(0),src)
        result=GCCPHATSRP(DEFAULT_ARRAY_METERS).estimate(pcm)
        assert result.valid
        assert abs((result.local_bearing_deg-degrees+180)%360-180)<=6, (degrees,result)


def test_motion_world_bearing_uses_capture_pose():
    rng=np.random.default_rng(2)
    mono=rng.standard_normal(10000)
    loc=GCCPHATSRP(DEFAULT_ARRAY_METERS)
    src=np.array([1.5*math.cos(math.radians(38)),1.5*math.sin(math.radians(38)),0])
    for rig_yaw in (0,30,-30):
        pcm=render_point_source(mono,2000,2048,16000,(0,0,0),yaw(rig_yaw),src)
        estimate=loc.estimate(pcm)
        assert abs((estimate.local_bearing_deg-(38-rig_yaw)+180)%360-180)<=6
        direction=rotate_xyzw(yaw(rig_yaw),(math.cos(math.radians(estimate.local_bearing_deg)),
                                              math.sin(math.radians(estimate.local_bearing_deg)),0))
        world=math.degrees(math.atan2(direction[1],direction[0]))
        assert abs(world-38)<=6


def test_silence_and_stale_fail_closed():
    class Always:
        def estimate(self, pcm): return BearingEstimate(30.,.8,True)
    stream=StreamingHearing(CFG,0,Always(),start_worker=False)
    stream.push_pose(pose(0));stream.push_pose(pose(100_000_000))
    stream.push_audio(np.zeros((1024,4),dtype=np.float32),0)
    assert stream._one()
    result,_=stream.poll(70_000_000)
    assert not result.valid and result.reason=='silence'
    stream.push_audio(np.full((256,4),.1,dtype=np.float32),1024)
    assert stream._one()
    result,_=stream.poll(70_000_000)
    assert result.valid and result.bearing_world_deg==pytest.approx(30.)
    stale,_=stream.poll(1_000_000_000)
    assert not stale.valid and stale.reason=='stale'
    assert stream.policy_features(1_000_000_000)[0][3]==0


def test_overload_skips_inference_jobs_but_preserves_samples():
    class Always:
        def estimate(self, pcm):return BearingEstimate(15.,.9,True)
    stream=StreamingHearing(CFG,0,Always(),start_worker=False)
    for k in range(100): stream.push_pose(pose(k*10_000_000))
    for k in range(20):stream.push_audio(np.ones((256,4),dtype=np.float32),k*256)
    assert stream._one()
    _,metric=stream.poll(stream.buffer.time_ns(5120))
    assert metric['audio_discontinuities']==0
    assert metric['overwritten_audio_samples']==0
    assert metric['inference_windows_skipped']>0
    assert metric['windows_processed']==1


def test_worker_no_lock_wait_while_estimating():
    class Slow:
        def estimate(self,pcm):
            time.sleep(.075)
            return BearingEstimate(0.,1.,True)
    stream=StreamingHearing(CFG,0,Slow())
    try:
        for k in range(30):stream.push_pose(pose(k*10_000_000))
        stream.push_audio(np.ones((1024,4),dtype=np.float32),0)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline and stream.poll(100_000_000)[1]['windows_processed']<1:
            time.sleep(.005)
        begin=time.perf_counter()
        stream.push_audio(np.ones((256,4),dtype=np.float32),1024)
        duration=time.perf_counter()-begin
        assert duration<.05
        assert stream.poll(100_000_000)[1]['worker_error'] is None
    finally:
        assert stream.close()


def test_explicit_audio_gap_invalidates_epoch():
    class Always:
        def estimate(self,pcm): return BearingEstimate(12.,.7,True)
    s=StreamingHearing(CFG,0,Always(),start_worker=False)
    for k in range(60):s.push_pose(pose(k*10_000_000))
    s.push_audio(np.ones((1024,4)),0);s._one()
    assert s.poll(70_000_000)[0].valid
    s.push_audio(np.ones((1024,4)),1500)
    # Stale previous-epoch results must never be published as valid.
    assert not s.poll(150_000_000)[0].valid
    s._one()
    assert s.poll(150_000_000)[0].valid
    assert s.poll(150_000_000)[1]['audio_discontinuities']==1


def test_pcm_ring_wraparound_retains_exact_order():
    c=StreamConfig(window_samples=256,hop_samples=64,buffer_seconds=.256)
    r=PCMBuffer(c,0)
    block=np.arange(16000,dtype=np.float32)
    for start in range(0,16000,128):
        segment=block[start:start+128]
        r.append(np.stack((segment,segment,segment,segment),axis=1),start)
    latest=r.window(16000)
    np.testing.assert_array_equal(latest.samples[:,0],np.arange(15744,16000))
    assert r.ingested_samples==16000
    assert r.overwritten_samples==16000-r.capacity
    assert r.discontinuities==0


def test_no_pose_never_publishes_bearing():
    class Always:
        def estimate(self,pcm):return BearingEstimate(42.,.9,True)
    s=StreamingHearing(CFG,0,Always(),start_worker=False)
    s.push_audio(np.ones((1024,4),dtype=np.float32),0)
    s._one()
    obs,m=s.poll(100_000_000)
    assert not obs.valid
    assert m['pose_misses']==1
    assert m['windows_processed']==0


def test_thread_failure_is_explicit_and_fails_closed():
    class Explode:
        def estimate(self,pcm):raise RuntimeError('intentional fault')
    s=StreamingHearing(CFG,0,Explode())
    try:
        for i in range(20):s.push_pose(pose(i*10_000_000))
        s.push_audio(np.ones((1024,4),dtype=np.float32),0)
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            obs,meta=s.poll(80_000_000)
            if meta['worker_error']:break
            time.sleep(.01)
        assert meta['worker_error']=='RuntimeError: intentional fault'
        assert not obs.valid and obs.reason=='worker_error'
    finally:
        assert s.close()


def test_invalid_pcm_and_silent_observation_contract():
    r=PCMBuffer(CFG,0)
    with pytest.raises(ValueError):r.append(np.full((64,4),np.nan),0)
    with pytest.raises(ValueError):r.append(np.zeros((64,3)),0)
    assert r.ingested_samples==0


def test_incoherent_noise_is_rejected_by_absolute_coherence_gate():
    loc=GCCPHATSRP(DEFAULT_ARRAY_METERS)
    for seed in range(25):
        independent=np.random.default_rng(seed).standard_normal((2048,4))
        assert not loc.estimate(independent).valid
