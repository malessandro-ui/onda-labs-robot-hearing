import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks'))
from room_motion_v07 import scenarios, generate, run
import numpy as np


def test_v07_matrix_coverage():
    cases=scenarios()
    assert len(cases)==27
    assert {r['group'] for r in cases}=={'angular_level','noise','room_absorption','continuous_rotation'}
    assert {r['rotation'] for r in cases if r['group']=='continuous_rotation'}=={45,90,180}
    assert {r['n_sources'] for r in cases}=={1,2,3}


def test_v07_repeatable_mixtures():
    for i in (0,13,22,26):
        a,geometry,truth,poses=generate(scenarios()[i],i)
        b,other,truth2,poses2=generate(scenarios()[i],i)
        assert a.shape==(3200,4)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(geometry.positions_m,other.positions_m)
        assert truth==truth2


def test_v07_paired_outputs(tmp_path):
    summary=run(tmp_path,limit=2)
    assert set(summary['overall'])=={'srp_phat_v05','subband_srp_v06'}
    assert summary['overall']['srp_phat_v05']['cases']==2
    assert len(list((tmp_path/'mixtures').glob('*.npz')))==2
    results=(tmp_path/'results.csv').read_text()
    assert 'method' in results and 'srp_phat_v05' in results and 'subband_srp_v06' in results
    assert json.loads((tmp_path/'summary.json').read_text())==summary
