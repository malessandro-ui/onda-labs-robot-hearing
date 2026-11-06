"""Native MuJoCo safety and opposite-side regressions, never a mock."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

pytest.importorskip('mujoco', reason='native MuJoCo physics required')


def _example():
    p = Path(__file__).resolve().parents[1] / 'examples' / 'mujoco_hear_and_turn.py'
    spec = importlib.util.spec_from_file_location('onda_v014_native_experiment', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_silence_no_spurious_turn(tmp_path):
    summary = _example().run_episode(mode='silence', output_dir=tmp_path, strict=True)
    assert summary['status'] == 'PASS'
    assert summary['acoustic_observation_count'] == 0
    assert summary['maximum_uncommanded_motion_deg'] <= 2


def test_native_opposite_source_reverses_turn(tmp_path):
    summary = _example().run_episode(source_azimuth_deg=-38., output_dir=tmp_path, strict=True)
    assert summary['status'] == 'PASS'
    assert summary['final_head_yaw_deg'] < -20
    assert summary['final_alignment_error_deg'] <= 10
