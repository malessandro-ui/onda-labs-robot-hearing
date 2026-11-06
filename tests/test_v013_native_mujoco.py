"""Real MuJoCo acceptance gate, skipped where optional MuJoCo isn't installed.

Unlike the synthetic stand-in tests, this runs the real MuJoCo C physics engine.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

mujoco = pytest.importorskip('mujoco', reason='optional real MuJoCo physics dependency not installed')


def test_native_mujoco_hear_and_turn(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'examples' / 'mujoco_hear_and_turn.py'
    spec = importlib.util.spec_from_file_location('onda_real_mujoco_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.run_episode(output_dir=tmp_path, strict=True)
    assert summary['simulator'].startswith('mujoco ')
    assert 'STANDIN' not in summary['simulator']
    assert summary['status'] == 'PASS'
    assert summary['source_truth_exposed_to_controller'] is False
