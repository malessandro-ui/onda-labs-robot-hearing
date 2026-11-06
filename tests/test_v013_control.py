import math
import numpy as np
import pytest

from onda_hearing.control import AcousticBearing, HeadYawConfig, HeadYawController
from onda_hearing.experiments import AcousticOnlySelector, AcousticSelectionConfig, EvaluationRow, evaluate_episode
from onda_hearing.runtime import HearingRuntime, HearingWindowBuffer, ScheduledSource, SceneSnapshot
from onda_hearing.contracts import MicrophoneArray, AudioPacket, DirectionEstimate, SpatialResult
from onda_hearing.spatial.interfaces import LocalizationFrame, SourceEvidence
from onda_hearing.physics.propagation import PoseTrajectory
from onda_hearing.spatial.motion_srp import MotionAwareSRPLocalizer, MotionSRPConfig


def array():
    return MicrophoneArray.from_positions([[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
             array_id='head_array',frame_id='robot/head',geometry_id='asym90mm_v013')


def snap(t, angle=0., source_deg=38.):
    a=math.radians(angle)
    b=math.radians(source_deg)
    return SceneSnapshot(t,(0,0,1),(0,0,math.sin(a/2),math.cos(a/2)),
                         {'phone':(1.5*math.cos(b),1.5*math.sin(b),1.)})


def test_controller_rate_limit_and_joint_limits():
    c=HeadYawController(HeadYawConfig(max_setpoint_speed_deg_s=30, min_yaw_deg=-60,max_yaw_deg=60))
    c.set_bearing(AcousticBearing(0,120.,.8))
    x=c.command(now_ns=100_000_000, base_world_yaw_rad=0.,dt_s=.1)
    assert math.degrees(x.target_joint_rad)==pytest.approx(3)
    for i in range(2,50):
        x=c.command(now_ns=i*100_000_000, base_world_yaw_rad=0.,dt_s=.1)
    assert math.degrees(x.target_joint_rad)<=60


def test_controller_refuses_stale_and_invalid_evidence():
    c=HeadYawController(HeadYawConfig(min_evidence_score=.3,max_observation_age_s=.2))
    assert not c.command(now_ns=10,base_world_yaw_rad=0.,dt_s=.002).accepted
    c.set_bearing(AcousticBearing(11,60.,.1))
    assert c.command(now_ns=100,base_world_yaw_rad=0.,dt_s=.002).reason=='insufficient_evidence'
    c.set_bearing(AcousticBearing(101,60.,.8))
    assert c.command(now_ns=500_000_000,base_world_yaw_rad=0.,dt_s=.002).reason=='stale_acoustic_observation'
    assert c.command(now_ns=700_000_000,base_world_yaw_rad=0.,dt_s=.002).target_joint_rad==0.


def test_controller_rejects_reordered_times_and_observations():
    c=HeadYawController()
    c.set_bearing(AcousticBearing(50,10,.1))
    with pytest.raises(ValueError):c.set_bearing(AcousticBearing(49,10,.1))
    c.command(now_ns=60,base_world_yaw_rad=0,dt_s=.01)
    with pytest.raises(ValueError):c.command(now_ns=60,base_world_yaw_rad=0,dt_s=.01)


def test_ground_truth_cannot_change_controller_action():
    c1=HeadYawController();c2=HeadYawController()
    for c in (c1,c2):c.set_bearing(AcousticBearing(0,38,.8))
    # Two independent evaluation truths: only the acoustic input is identical.
    truth_a=-70;truth_b=140
    a=c1.command(now_ns=10_000_000,base_world_yaw_rad=0.,dt_s=.01)
    b=c2.command(now_ns=10_000_000,base_world_yaw_rad=0.,dt_s=.01)
    assert truth_a!=truth_b
    assert a==b


def test_controller_compensates_base_yaw():
    c=HeadYawController(HeadYawConfig(max_setpoint_speed_deg_s=1000))
    c.set_bearing(AcousticBearing(0,60,.8))
    x=c.command(now_ns=100_000_000,base_world_yaw_rad=math.radians(20),dt_s=.1)
    assert math.degrees(x.target_joint_rad)==pytest.approx(40)


def test_buffer_exact_audio_samples_and_pose_clock():
    a=array();b=HearingWindowBuffer(a,sample_rate_hz=16000,window_samples=640,capacity_samples=1600)
    b.push_pose(snap(0))
    b.push_pose(snap(20_000_000))
    b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=0,clock_id='simulation',array=a,sequence=0))
    b.push_pose(snap(40_000_000))
    b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=20_000_000,clock_id='simulation',array=a,sequence=1))
    w=b.window()
    assert w.audio.pcm.shape==(640,4)
    assert w.audio.capture.start_ns==0
    assert w.audio.capture.end_ns==40_000_000
    assert w.poses.times_s[0]==0
    assert w.poses.times_s[-1]==pytest.approx(.04)


def test_buffer_rejects_discontinuity_and_sequence_gap():
    a=array();b=HearingWindowBuffer(a,sample_rate_hz=16000,window_samples=640,capacity_samples=1600)
    b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=0,clock_id='simulation',array=a,sequence=0))
    with pytest.raises(ValueError):b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=20_000_000,clock_id='simulation',array=a,sequence=2))
    with pytest.raises(ValueError):b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=30_000_000,clock_id='simulation',array=a,sequence=1))
    with pytest.raises(ValueError):b.push_audio(AudioPacket.from_pcm(np.zeros((320,4),np.float32),16000,start_ns=20_000_000,clock_id='simulation',array=a,sequence=1,discontinuity=True))


class StubLocalizer:
    def __init__(self, angles):self.angles=angles
    def reset(self):pass
    def estimate(self,packet,array,poses,*,pose_clock_id,world_frame_id):
        directions=tuple(DirectionEstimate(math.radians(a),'world',score) for a,score in self.angles)
        return LocalizationFrame(SpatialResult(packet.capture,'world',directions,'stub'),tuple(SourceEvidence(d.azimuth_rad,d.score,1.,1.,4,6) for d in directions),'stub')


def fake_window():
    a=array();p=AudioPacket.from_pcm(np.zeros((640,4),np.float32),16000,start_ns=0,clock_id='simulation',array=a)
    return (type('W',(),{'audio':p,'poses':PoseTrajectory.stationary((0,0,0)),'pose_clock_id':'simulation'})(),a)


def test_selector_rejects_ambiguous_candidates():
    w,a=fake_window()
    sel=AcousticOnlySelector(StubLocalizer([(38,.7),(-45,.65)]),AcousticSelectionConfig(min_consistent_windows=1))
    assert sel.process(w,a).reason=='ambiguous_spatial_candidates'


def test_selector_requires_repeated_evidence():
    w,a=fake_window()
    sel=AcousticOnlySelector(StubLocalizer([(38,.7)]),AcousticSelectionConfig(min_consistent_windows=2))
    assert sel.process(w,a).bearing is None
    assert sel.process(w,a).bearing.world_azimuth_deg==pytest.approx(38)


def test_silence_evaluation_holds_head():
    rows=[EvaluationRow(t,0,0,None,None,'hold','silence',None,None) for t in (0,.1,.2)]
    assert evaluate_episode(rows)['status']=='PASS'
    changed=[*rows[:2],EvaluationRow(.2,8,8,None,None,'turn','silence',None,None)]
    assert evaluate_episode(changed)['status']=='FAIL'


def test_sound_evaluation_does_not_accept_no_detection():
    rows=[EvaluationRow(.1,39,39,None,None,'hold','silent',38,1)]
    assert evaluate_episode(rows)['status']=='FAIL'


def test_full_audio_render_and_motion_localization_single_source():
    fs=16000;a=array();rng=np.random.default_rng(321)
    x=rng.normal(0,.07,size=fs).astype('f')
    spec=np.fft.rfft(x);f=np.fft.rfftfreq(len(x),1/fs);spec[(f<300)|(f>3800)]=0
    x=np.fft.irfft(spec,n=len(x)).astype('f')
    hearing=HearingRuntime(a,(ScheduledSource('phone',x,0),),sample_rate_hz=fs)
    buffer=HearingWindowBuffer(a,sample_rate_hz=fs,window_samples=4096,capacity_samples=8192)
    hearing.advance(snap(0));buffer.push_pose(snap(0))
    for k in range(1,19):
        s=snap(k*20_000_000)
        out=hearing.advance(s)
        buffer.push_pose(s);buffer.push_audio(out.audio)
    window=buffer.window()
    loc=MotionAwareSRPLocalizer(MotionSRPConfig(max_candidates=3,min_relative_score=.65))
    found=loc.estimate(window.audio,a,window.poses,pose_clock_id=window.pose_clock_id)
    assert found.result.directions
    assert abs(math.degrees(found.result.directions[0].azimuth_rad)-38)<8


def test_no_audio_stays_silent_with_real_localizer():
    fs=16000;a=array()
    hearing=HearingRuntime(a,(ScheduledSource('phone',np.zeros(fs),0),),sample_rate_hz=fs)
    buffer=HearingWindowBuffer(a,sample_rate_hz=fs,window_samples=4096,capacity_samples=8192)
    hearing.advance(snap(0));buffer.push_pose(snap(0))
    for k in range(1,15):
        s=snap(k*20_000_000)
        out=hearing.advance(s);buffer.push_pose(s);buffer.push_audio(out.audio)
    sel=AcousticOnlySelector(MotionAwareSRPLocalizer())
    assert sel.process(buffer.window(),a).bearing is None
