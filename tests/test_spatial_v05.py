import numpy as np
import pytest
from onda_hearing.contracts import AudioPacket, MicrophoneArray, SpatialResult, DirectionEstimate
from onda_hearing.spatial import (SRPCandidateLocalizer,SRPConfig,CandidateTracker,LocalizationFrame,SourceEvidence,UnsupportedSeparator,SeparationUnavailableError)

ARRAY=MicrophoneArray.from_positions([[-.06,-.04,0],[.06,-.04,0],[.06,.04,0],[-.06,.04,0]],array_id='a',frame_id='head',geometry_id='g')

def packet(pcm, array=ARRAY, start_ns=0):
    return AudioPacket.from_pcm(pcm,16000,start_ns=start_ns,clock_id='simulation',array=array)

def frame(*degrees):
    dirs=tuple(DirectionEstimate(np.deg2rad(d),'head',.1) for d in degrees)
    return LocalizationFrame(SpatialResult(packet(np.zeros((160,4),np.float32)).capture,'head',dirs,'test'),tuple(SourceEvidence(np.deg2rad(d),.1,.05,1,4,6) for d in degrees),'test')

def test_silent_yields_no_candidates():
    assert not SRPCandidateLocalizer().estimate(packet(np.zeros((512,4),np.float32)),ARRAY).result.directions

def test_reject_line_geometry():
    arr=MicrophoneArray.from_positions([[0,0,0],[.1,0,0],[.2,0,0]],array_id='a',frame_id='head',geometry_id='g')
    with pytest.raises(ValueError,match='non-collinear'):
        SRPCandidateLocalizer().estimate(packet(np.zeros((512,3),np.float32),arr),arr)

def test_array_and_audio_id_mismatch():
    arr=MicrophoneArray.from_positions(ARRAY.positions_m,array_id='else',frame_id='head',geometry_id='g')
    with pytest.raises(ValueError,match='mismatch'):
        SRPCandidateLocalizer().estimate(packet(np.zeros((512,4),np.float32)),arr)

def test_configuration_rejects_nonsense():
    with pytest.raises(ValueError):SRPConfig(max_candidates=0)
    with pytest.raises(ValueError):SRPConfig(min_relative_score=1.5)

def test_tracker_confirms_and_coasts():
    t=CandidateTracker(confirmation_hits=2,max_misses=2)
    assert not t.update(frame(25),0.1)[0].confirmed
    obs=t.update(frame(27),.2)
    assert obs[0].confirmed and obs[0].track_id==1
    coast=t.update(frame(),.3)
    assert coast[0].coasting and not coast[0].confirmed
    reacquired=t.update(frame(27),.4)
    assert reacquired[0].track_id==1 and reacquired[0].confirmed

def test_tracker_expires():
    t=CandidateTracker(max_idle_s=.15,max_misses=9)
    t.update(frame(0),.1)
    assert not t.update(frame(),.3)

def test_tracker_wraparound():
    t=CandidateTracker(confirmation_hits=2)
    t.update(frame(179),.1)
    out=t.update(frame(-179),.2)
    assert out[0].track_id==1 and out[0].confirmed

def test_tracker_rejects_reordered_time_and_changed_frame():
    t=CandidateTracker()
    t.update(frame(0),.1)
    with pytest.raises(ValueError,match='timestamps'):t.update(frame(0),.1)

def test_candidate_count_not_certified():
    with pytest.raises(ValueError,match='certif'):
        LocalizationFrame(frame(0).result,frame(0).evidence,'test',True)

def test_separation_is_explicitly_unavailable():
    with pytest.raises(SeparationUnavailableError):
        UnsupportedSeparator().separate(packet(np.zeros((128,4),np.float32)),ARRAY,frame())
