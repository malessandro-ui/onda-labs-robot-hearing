"""Dependency-free numerical and policy acceptance utilities regression tests."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from onda_v017_native_common import angle_error,truth_bearing,quantiles,Accumulator,validate_metrics
from onda_hearing.acoustic_scene import Pose3
from onda_hearing.semantics.core import EventScore,SemanticObservation
from types import SimpleNamespace

def test_angle_wrap():
    assert angle_error(179,-179)==2
    assert angle_error(-175,175)==10
    assert angle_error(30,30)==0

def test_truth():
    assert abs(truth_bearing(Pose3((0,0,0)),Pose3((2,1,0)))-26.565051177)<1e-6

def test_metrics_fail_closed():
    acc=Accumulator()
    reasons=validate_metrics({'worker_error':'failed','audio_discontinuities':2},acc)
    assert 'no_valid_semantic_window' in reasons
    assert 'worker_error' in reasons
    assert any('audio_discontinuities=' in s for s in reasons)

def test_no_spatial_semantic_source_association():
    a=Accumulator()
    spatial=SimpleNamespace(valid=True,source_sample_end=2048,capture_time_ns=0,bearing_world_deg=30.)
    semantic=SemanticObservation(valid=True,events=(EventScore('Dial tone',.5),),
              brain_name='test',capture_mid_ns=0,publish_mono_ns=0,
              start_sample=0,end_sample=16000,epoch=0,reason='ok')
    class Stream:
        def poll(self,_):return np.zeros(534,dtype=np.float32),spatial,semantic,{}
    a.observe(Stream(),0,lambda _:30.,534)
    assert a.labels['Dial tone']==.5
    assert a.errors==[0.]
    assert a.association_claimed is False
