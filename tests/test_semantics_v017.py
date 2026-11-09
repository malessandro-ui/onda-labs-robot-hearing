import numpy as np
import pytest
from onda_hearing.semantics import EventScore,RuleBasedTestBrain,AsyncSemanticHearing,PolicyEncoder,UnifiedHearing
from onda_hearing.semantics.core import SemanticObservation
from onda_hearing.streaming.core import StreamConfig,PCMBuffer,HearingObservation

def test_event_score():
    with pytest.raises(ValueError):EventScore('ring',2.)
    with pytest.raises(ValueError):EventScore('',0.5)

def test_custom_brain_interface():
    class Custom:
        name='customer-brain'
        def classify(self,mono,sample_rate): return (EventScore('trumpet',.8),)
    buffer=PCMBuffer(StreamConfig(channels=2,window_samples=512,hop_samples=256),0)
    worker=AsyncSemanticHearing(buffer,Custom(),window_samples=512,hop_samples=256,start_worker=False)
    buffer.append(np.ones((1024,2),np.float32)*.1,0)
    assert worker._step()
    obs,stats=worker.poll(768*1000000000//16000)
    assert obs.valid and obs.events[0].label=='trumpet'

def test_silence_no_event():
    buffer=PCMBuffer(StreamConfig(channels=2,window_samples=512,hop_samples=256),0)
    worker=AsyncSemanticHearing(buffer,RuleBasedTestBrain(),window_samples=512,hop_samples=256,start_worker=False)
    buffer.append(np.zeros((512,2),np.float32),0)
    worker._step();obs,_=worker.poll(512*1000000000//16000)
    assert not obs.valid and obs.reason=='silence'

def test_policy_no_fabricated_association():
    enc=PolicyEncoder(('telephone','trumpet'))
    spatial=HearingObservation(valid=True,bearing_local_deg=30.,confidence=.7,capture_time_ns=100)
    semantic=SemanticObservation(True,(EventScore('trumpet',.88),),'customer',100,100,0,100,0,'ok')
    x=enc.encode(spatial,semantic,100)
    assert x.shape==(9,) and x[8]==pytest.approx(.88)
    assert x[3]==1 and x[5]==1

def test_policy_invalid_zeroed():
    enc=PolicyEncoder(('ring',))
    x=enc.encode(HearingObservation(),SemanticObservation(),0)
    assert np.all(x==0)

def test_policy_vocab_validation():
    with pytest.raises(ValueError):PolicyEncoder(('x','x'))

def test_unified_audio_duplicates_to_different_window_rings():
    """Regression for v0.16 PCMBuffer's fixed-window contract."""
    from onda_hearing.streaming.core import PCMBuffer,StreamConfig
    class FakeSpatial:
        def __init__(self):
            self.buffer=PCMBuffer(StreamConfig(channels=2,window_samples=512,hop_samples=256),0)
            self.closed=False
        def push_audio(self,data,start):self.buffer.append(data,start)
        def close(self):self.closed=True;return True
        def poll(self,t):return HearingObservation(),{}
    stream=FakeSpatial()
    sem_buffer=PCMBuffer(StreamConfig(channels=2,window_samples=1024,hop_samples=256),0)
    sem=AsyncSemanticHearing(sem_buffer,RuleBasedTestBrain(),window_samples=1024,hop_samples=256,start_worker=False)
    unified=UnifiedHearing(stream,sem,PolicyEncoder(('test_non_silent',)))
    unified.push_audio(np.ones((1536,2),dtype=np.float32)*.1,0)
    assert sem._step()
    assert sem_buffer.bounds()[1]==stream.buffer.bounds()[1]==1536
    assert sem._latest.end_sample-sem._latest.start_sample==1024
    assert unified.close() and stream.closed


def test_unified_rejects_shared_ring():
    class FakeSpatial:
        def __init__(self):self.buffer=PCMBuffer(StreamConfig(channels=2,window_samples=512,hop_samples=256),0)
    stream=FakeSpatial()
    sem=AsyncSemanticHearing(stream.buffer,RuleBasedTestBrain(),window_samples=512,hop_samples=256,start_worker=False)
    with pytest.raises(ValueError):UnifiedHearing(stream,sem,PolicyEncoder(('test',)))

def test_semantic_reference_channel_avoids_phase_cancellation():
    class Brain:
        name='ref'
        def classify(self,mono,sample_rate):
            return (EventScore('audible',.9),) if np.max(np.abs(mono))>.01 else ()
    cfg=StreamConfig(channels=2,window_samples=512,hop_samples=256)
    buf=PCMBuffer(cfg,0)
    sem=AsyncSemanticHearing(buf,Brain(),window_samples=512,hop_samples=256,start_worker=False)
    x=np.zeros((512,2),dtype=np.float32);x[:,0]=.2;x[:,1]=-.2
    buf.append(x,0);sem._step()
    assert sem._latest.valid and sem._latest.events[0].label=='audible'

def test_high_level_attach_custom_brain():
    from onda_hearing.semantics import attach_sound_brain
    class Custom:
        name='custom'
        def classify(self,mono,sample_rate):return (EventScore('alarm',.75),)
    class FakeStream:
        def __init__(self):
            self.config=StreamConfig(channels=2,window_samples=512,hop_samples=256)
            self.buffer=PCMBuffer(self.config,0)
            self.closed=False
        def push_audio(self,x,start):self.buffer.append(x,start)
        def poll(self,ts):return HearingObservation(),{}
        def close(self):self.closed=True;return True
    s=FakeStream()
    unified=attach_sound_brain(s,brain=Custom(),vocabulary=('alarm',),
                               semantic_window_s=.064,semantic_hop_s=.016,start_worker=False)
    unified.push_audio(np.ones((1024,2),dtype=np.float32)*.1,0)
    assert unified.semantic._step()
    f,sp,sem,metrics=unified.poll(1024*1000000000//16000)
    assert f.shape==(8,) and f[-1]==pytest.approx(.75)
    assert unified.close() and s.closed


def test_custom_brain_needs_vocabulary():
    from onda_hearing.semantics import attach_sound_brain
    class Custom:
        name='x'
    class FakeStream:
        def __init__(self):
            self.config=StreamConfig(channels=2,window_samples=512,hop_samples=256)
            self.buffer=PCMBuffer(self.config,0)
    with pytest.raises(ValueError):attach_sound_brain(FakeStream(),brain=Custom())
