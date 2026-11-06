import numpy as np
import pytest
from onda_hearing.contracts import AudioPacket,MicrophoneArray
from onda_hearing.physics.propagation import PoseTrajectory
from onda_hearing.spatial import MotionSRPConfig,MotionAwareSRPLocalizer

OFFSETS=[[-.07,-.05,0],[.07,-.05,0],[.07,.05,0],[-.07,.05,0]]
def packet():
    arr=MicrophoneArray.from_positions(OFFSETS,array_id='test',frame_id='array',geometry_id='rigid')
    rng=np.random.default_rng(7)
    pcm=rng.normal(0,.01,(3200,4)).astype('float32')
    audio=AudioPacket.from_pcm(pcm,16000,start_ns=0,clock_id='sim',array=arr)
    pose=PoseTrajectory.stationary([0,0,0])
    return arr,audio,pose

def test_valid_output_frame():
    array,audio,pose=packet()
    result=MotionAwareSRPLocalizer().estimate(audio,array,pose,pose_clock_id='sim')
    assert result.result.frame_id=='world'
    assert result.method=='motion_srp_candidate'
    assert len(result.evidence)==len(result.result.directions)

def test_clock_rejection():
    array,audio,pose=packet()
    with pytest.raises(ValueError,match='clock'):
        MotionAwareSRPLocalizer().estimate(audio,array,pose,pose_clock_id='host')

def test_geometry_rejection():
    array,audio,pose=packet()
    other=MicrophoneArray.from_positions(OFFSETS,array_id='other',frame_id='array',geometry_id='rigid')
    with pytest.raises(ValueError,match='mismatch'):
        MotionAwareSRPLocalizer().estimate(audio,other,pose,pose_clock_id='sim')

def test_pose_extrapolation_rejection():
    array,audio,_=packet()
    pose=PoseTrajectory(np.array([0,.05]),np.zeros((2,3)),np.repeat(np.eye(3)[None],2,axis=0))
    with pytest.raises(ValueError,match='extrapolation'):
        MotionAwareSRPLocalizer().estimate(audio,array,pose,pose_clock_id='sim')

def test_pose_staleness_rejection():
    array,audio,_=packet()
    pose=PoseTrajectory(np.array([0,.2]),np.zeros((2,3)),np.repeat(np.eye(3)[None],2,axis=0))
    with pytest.raises(ValueError,match='stale'):
        MotionAwareSRPLocalizer().estimate(audio,array,pose,pose_clock_id='sim')

def test_silence_has_no_candidates():
    array,audio,pose=packet()
    zero=AudioPacket.from_pcm(np.zeros_like(audio.pcm),16000,start_ns=0,clock_id='sim',array=array)
    result=MotionAwareSRPLocalizer().estimate(zero,array,pose,pose_clock_id='sim')
    assert not result.result.directions

def test_invalid_window_configuration():
    with pytest.raises(ValueError):MotionSRPConfig(block_samples=64)
    with pytest.raises(ValueError):MotionSRPConfig(block_samples=256,hop_samples=300)

def test_repeatable():
    array,audio,pose=packet()
    m=MotionAwareSRPLocalizer()
    a=m.estimate(audio,array,pose,pose_clock_id='sim')
    b=m.estimate(audio,array,pose,pose_clock_id='sim')
    assert a.result.directions==b.result.directions
