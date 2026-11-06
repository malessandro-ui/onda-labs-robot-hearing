"""Contract and regression tests for paired synthetic spatial comparison."""
import sys
from pathlib import Path
import numpy as np
import pytest
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.spatial import SubbandConfig, SubbandSRPLocalizer, SRPCandidateLocalizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from spatial_challenge_v06 import cases,generate,match_one_to_one,evaluate


def test_stress_matrix_has_required_coverage():
    c=cases();names={x['name'] for x in c}
    for sep in [10,15,20,30,45,60,90,120,150,180]:
        for db in [0,-10,-20]: assert f'two_sep{sep}_db{db}' in names
    assert len([x for x in c if len(x['angles'])==3])>=4
    assert {'snr_0','snr_10','snr_20','array_rotation_90','echo_0.8'}<=names


def test_matching_one_to_one_no_reusing_one_detection():
    assert match_one_to_one([-5,5],[0],15)==1
    assert match_one_to_one([-45,45],[-46,46],15)==2
    assert match_one_to_one([-45,45],[0],15)==0
    assert match_one_to_one([],[],15)==0


def test_matching_wraparound():
    assert match_one_to_one([179],[-179],5)==1


def test_saved_input_determinism():
    case=next(x for x in cases() if x['name']=='two_sep90_db-10')
    a,arr_a=generate(case,220);b,arr_b=generate(case,220)
    assert np.array_equal(a,b)
    assert np.array_equal(arr_a.positions_m,arr_b.positions_m)


def test_noise_changes_waveform():
    original=next(x for x in cases() if x['name']=='two_sep90_db-10')
    modified=dict(original,snr=10)
    clean,_=generate(original,222);noisy,_=generate(modified,222)
    assert not np.array_equal(clean,noisy)


def test_rotation_rotates_geometry_without_changing_channel_order():
    base=next(x for x in cases() if x['name']=='array_rotation_0')
    turned=next(x for x in cases() if x['name']=='array_rotation_90')
    _,a=generate(base,10);_,b=generate(turned,10)
    assert a.channels==b.channels
    assert np.allclose(np.linalg.norm(a.positions_m,axis=1),np.linalg.norm(b.positions_m,axis=1))
    assert not np.allclose(a.positions_m,b.positions_m)


def test_subband_rejects_incompatible_geometry():
    a=MicrophoneArray.from_positions([[0,0,0],[.1,0,0]],array_id='a',frame_id='f',geometry_id='g')
    p=AudioPacket.from_pcm(np.zeros((512,2),dtype=np.float32),16000,start_ns=0,clock_id='sim',array=a)
    with pytest.raises(ValueError,match='non-collinear'):
        SubbandSRPLocalizer().estimate(p,a)


def test_subband_silence_returns_no_sources():
    a=MicrophoneArray.from_positions([[-.05,-.05,0],[.05,-.05,0],[.05,.05,0],[-.05,.05,0]])
    p=AudioPacket.from_pcm(np.zeros((1024,4),dtype=np.float32),16000,start_ns=0,clock_id='sim',array=a)
    assert not SubbandSRPLocalizer().estimate(p,a).result.directions


def test_subband_invalid_config():
    for kwargs in [{'bands':1},{'max_candidates':0},{'min_relative_score':1.5},{'min_frequency_hz':8000,'max_frequency_hz':1000}]:
        with pytest.raises(ValueError): SubbandConfig(**kwargs)


def test_paired_files_and_metrics(tmp_path):
    report=evaluate(tmp_path,limit=4)
    assert len(list((tmp_path/'mixtures').glob('*.npz')))==4
    assert (tmp_path/'summary.json').is_file()
    import csv
    with (tmp_path/'results.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert {r['algorithm'] for r in rows}==set(report['metrics'])
    for i in range(4):
        assert len({r['mixture'] for r in rows if int(r['case_id'])==i})==1
