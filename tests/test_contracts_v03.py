import math
import numpy as np
import pytest
from onda_hearing.contracts import (
    AudioPacket, Microphone, MicrophoneArray, CaptureInterval, ClockMapping,
    DirectionEstimate, SpatialResult, SpatialCapabilities, UnsupportedConfigurationError,
    HearingObservation, SourceTrack, TrackState, Pose, run_checked, from_legacy_block,
)
from onda_hearing import AudioBlock


def arr(points=((0,0,0),(.04,0,0),(0,.04,0)),**kw):
    return MicrophoneArray.from_positions(points,**kw)

def audio(a=None,**kw):
    a = arr() if a is None else a
    return AudioPacket.from_pcm(np.zeros((160,a.channels),dtype=np.float32),16000,
                                start_ns=0,clock_id='simulation',array=a,**kw)

def cap(**kw):
    return SpatialCapabilities(backend_id='test',min_channels=3,max_channels=8,
        sample_rates_hz=frozenset({16000}),min_xy_rank=2,**kw)

@pytest.mark.parametrize('count', [1,2,3,4,8,16])
def test_arbitrary_array_count(count):
    a=arr([(i*.02,(i%3)*.03,(i%2)*.01) for i in range(count)])
    assert a.channels==count
    assert audio(a).channel_count==count

def test_channel_order_and_duplicates():
    a=MicrophoneArray('array','head',(Microphone('right',1,(.1,0,0)),Microphone('left',0,(0,0,0))),'geometry')
    assert a.microphones[0].name=='left'
    with pytest.raises(ValueError,match='contiguous'):
        MicrophoneArray('array','head',(Microphone('a',0,(0,0,0)),Microphone('b',2,(1,0,0))),'geometry')
    with pytest.raises(ValueError,match='coincident'):
        arr([(0,0,0),(0,0,0)])
    with pytest.raises(ValueError,match='unique'):
        MicrophoneArray('a','b',(Microphone('mic',0,(0,0,0)),Microphone('mic',1,(1,0,0))),'g')

def test_rank_and_capability_negotiation():
    collinear=arr([(0,0,0),(.1,0,0),(.2,0,0)])
    with pytest.raises(UnsupportedConfigurationError,match='geometry'):
        cap().validate(audio(collinear),collinear)
    good=arr();cap().validate(audio(good),good)
    with pytest.raises(UnsupportedConfigurationError,match='simultaneous'):
        cap().validate(audio(good),good,requested_sources=2)
    with pytest.raises(UnsupportedConfigurationError,match='separation'):
        cap().validate(audio(good),good,require_separation=True)
    with pytest.raises(ValueError,match='max_sources'):
        cap(max_sources=2)

def test_array_identity_and_clock():
    a=arr();p=audio(a)
    assert p.capture.duration_s==pytest.approx(.01)
    with pytest.raises(ValueError,match='mismatch'):
        p.validate_array(arr(array_id='other'))
    with pytest.raises(ValueError,match='interval'):
        AudioPacket(p.pcm,16000,CaptureInterval('sim',0,5000),a.array_id,a.frame_id,a.geometry_id)
    with pytest.raises(ValueError,match='interval'):
        CaptureInterval('sim',5,5)
    assert ClockMapping('sim','hardware',100,10).offset_ns==100

def test_immutability_and_pcm_validation():
    a=arr();data=np.zeros((160,3),dtype=np.float32)
    packet=AudioPacket.from_pcm(data,16000,start_ns=0,clock_id='sim',array=a)
    data[0,0]=.7
    assert packet.pcm[0,0]==0
    with pytest.raises(ValueError):packet.pcm[0,0]=.5
    with pytest.raises(ValueError,match='normalized'):
        AudioPacket.from_pcm(np.ones((160,3))*2,16000,start_ns=0,clock_id='sim',array=a)
    with pytest.raises(ValueError,match='finite'):
        AudioPacket.from_pcm(np.full((160,3),np.nan),16000,start_ns=0,clock_id='sim',array=a)

def test_pose_and_angles():
    p=Pose('world','robot/head',(1,2,3),(0,0,0,1))
    assert p.translation_m==(1.,2.,3.)
    with pytest.raises(ValueError,match='unit quaternion'):
        Pose('world','head',(0,0,0),(0,0,0,0))
    assert DirectionEstimate(3*math.pi,'head',.3).azimuth_rad==pytest.approx(-math.pi)
    with pytest.raises(ValueError):DirectionEstimate(0,'head',float('nan'))

def test_result_and_observation():
    packet=audio();direction=DirectionEstimate(.5,packet.frame_id,.8)
    result=SpatialResult(packet.capture,packet.frame_id,(direction,),'test')
    track=SourceTrack('t1',packet.frame_id,.5,TrackState.TENTATIVE,packet.capture.end_ns)
    observation=HearingObservation(packet.capture,result,(track,))
    assert observation.spatial.directions[0].score==.8
    with pytest.raises(ValueError,match='frame'):
        SpatialResult(packet.capture,'other',(direction,),'test')
    with pytest.raises(ValueError,match='duplicate track'):
        HearingObservation(packet.capture,result,(track,track))

def test_backend_contract():
    packet=audio();a=arr()
    class Backend:
        capabilities=cap()
        def reset(self):pass
        def process(self, audio, geometry):
            return SpatialResult(audio.capture,audio.frame_id,(DirectionEstimate(.1,audio.frame_id,.4),),'test')
    assert len(run_checked(Backend(),packet,a).directions)==1
    with pytest.raises(UnsupportedConfigurationError):run_checked(Backend(),packet,a,requested_sources=2)
    class BadBackend(Backend):
        def process(self,audio,geometry):
            return SpatialResult(audio.capture,'wrong',(), 'test')
    with pytest.raises(RuntimeError,match='contract'):
        run_checked(BadBackend(),packet,a)

def test_legacy_to_new_contract():
    a=arr(frame_id='head',geometry_id='measured_v1')
    legacy=AudioBlock(np.zeros((160,3),dtype=np.float32),16000,1.0,'head')
    packet=from_legacy_block(legacy,a,clock_id='sim')
    packet.validate_array(a)
    assert packet.capture.start_ns==1_000_000_000
    assert packet.capture.end_ns==1_010_000_000
    with pytest.raises(ValueError):from_legacy_block(legacy,arr(frame_id='other'),clock_id='sim')
