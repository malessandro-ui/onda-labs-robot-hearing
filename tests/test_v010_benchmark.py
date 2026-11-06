from pathlib import Path
import importlib.util
import sys

def test_benchmark_no_odas_results_fabricated(tmp_path):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'benchmarks'))
    from odas_comparison_v010 import run
    result=run(tmp_path,limit=1)
    assert result['methods']['odas_offline_ssl']['evaluated']==0
    assert result['methods']['odas_offline_ssl']['precision'] is None
    assert result['methods']['motion_srp_v09']['evaluated']==1
    assert (tmp_path/'mixtures'/'000.npz').exists()
