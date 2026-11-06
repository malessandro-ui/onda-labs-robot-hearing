import numpy as np
import pytest
from onda_hearing import (AudioBlock, BearingTracker, Direction, HearingPipeline,
                          MicrophoneArray, FreeFieldSimulator, SoundSource, estimate_azimuth)


def array():
    return MicrophoneArray([[-.08,-.08,0],[.08,-.08,0],[.08,.08,0],[-.08,.08,0]])


def test_single_source_no_phantom_second_peak():
    sr=24000
    rng=np.random.default_rng(41)
    x=FreeFieldSimulator(sr).render(array(),np.zeros(3),[
        SoundSource('first',rng.normal(0,.1,10000),[3,1.5,0],.02)],.5)
    peaks=estimate_azimuth(x,array(),max_sources=2)
    assert len(peaks)==1
    target=np.arctan2(1.5,3)
    assert abs((peaks[0].azimuth_rad-target+np.pi)%(2*np.pi)-np.pi)<np.deg2rad(10)


def test_reject_silence_and_nonfinite_gcc():
    from onda_hearing.localization import gcc_phat
    assert estimate_azimuth(AudioBlock(np.zeros((1024,4)),24000,0,'microphone_array'),array())==()
    with pytest.raises(ValueError):
        gcc_phat(np.array([1., np.nan]),np.ones(2),24000,.01)


def test_tracker_drops_stale_and_invalid_candidates():
    t=BearingTracker(confirm_hits=2,max_missed=1,max_idle_s=.5,min_observation_score=.1)
    low=t.update((Direction(0,.05,'a'),),.1)
    assert low==()
    first=t.update((Direction(0,.2,'a'),),.2)
    second=t.update((Direction(.01,.2,'a'),),.3)
    assert len(first)==1 and not first[0].confirmed
    assert second[0].confirmed and second[0].track_id==first[0].track_id
    assert t.update((),.4)[0].missed==1
    assert t.update((),.5)==()
    with pytest.raises(ValueError):
        t.update((Direction(float('nan'),.5,'a'),),.6)


def test_tracker_consecutive_confirmation():
    t=BearingTracker(confirm_hits=3,max_missed=2)
    t.update((Direction(.1,.5,'a'),),.1)
    t.update((Direction(.1,.5,'a'),),.2)
    t.update((),.3)
    assert not t.update((Direction(.1,.5,'a'),),.4)[0].confirmed
    assert not t.update((Direction(.1,.5,'a'),),.5)[0].confirmed
    assert t.update((Direction(.1,.5,'a'),),.6)[0].confirmed


def test_pipeline_rejects_out_of_order_audio_and_reset():
    p=HearingPipeline(array())
    chunk=AudioBlock(np.zeros((120,4)),24000,0,'microphone_array')
    p.process(chunk)
    with pytest.raises(ValueError):
        p.process(chunk)
    p.reset()
    p.process(chunk)


def test_candidate_configuration_errors():
    x=AudioBlock(np.ones((100,4)),24000,0,'microphone_array')
    with pytest.raises(ValueError):
        estimate_azimuth(x,array(),min_peak_relative=1.1)
    with pytest.raises(ValueError):
        estimate_azimuth(x,array(),c=0)
