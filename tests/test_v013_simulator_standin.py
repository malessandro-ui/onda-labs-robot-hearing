"""Synthetic MuJoCo API stand-in for end-to-end code-path tests.

This is deliberately labelled a STAND-IN, not actual MuJoCo physics. The native
MuJoCo acceptance experiment is examples/mujoco_hear_and_turn.py --strict.
"""
from __future__ import annotations
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class FakeModel:
    opt = types.SimpleNamespace(timestep=0.002)

    @classmethod
    def from_xml_string(cls, xml):
        import re
        match = re.search(r'<site name="beacon" pos="([^\"]+)"', xml)
        xyz = tuple(float(v) for v in match.group(1).split())
        result = cls()
        result.source_xyz = xyz
        return result

    def site(self, name):
        return types.SimpleNamespace(id={'head_array': 0, 'beacon': 1}[name])


class FakeData:
    def __init__(self, model):
        self.time = 0.
        self.qpos = np.zeros(1)
        self.ctrl = np.zeros(1)
        self.site_xpos = np.zeros((2, 3))
        self.site_xpos[1] = model.source_xyz
        self.site_xmat = np.zeros((2, 9))
        self.update()

    def update(self):
        a = self.qpos[0]
        c, s = math.cos(a), math.sin(a)
        self.site_xpos[0] = [0, 0, 1]
        self.site_xmat[0] = [c, -s, 0, s, c, 0, 0, 0, 1]
        self.site_xmat[1] = np.eye(3).flatten()


def fake_step(model, data):
    data.qpos[0] += np.clip((data.ctrl[0] - data.qpos[0]) * .08, -.02, .02)
    data.time += model.opt.timestep
    data.update()


@pytest.fixture
def experiment(monkeypatch):
    fake = types.SimpleNamespace(MjModel=FakeModel, MjData=FakeData,
                                 mj_step=fake_step,
                                 mj_forward=lambda model, data: data.update(),
                                 __version__='STANDIN_NOT_REAL_MUJOCO')
    monkeypatch.setitem(sys.modules, 'mujoco', fake)
    file = Path(__file__).resolve().parents[1] / 'examples' / 'mujoco_hear_and_turn.py'
    spec = importlib.util.spec_from_file_location('onda_v013_example', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standin_hearing_turns_to_signal_derived_direction(experiment, tmp_path):
    result = experiment.run_episode(output_dir=tmp_path, strict=True)
    assert result['status'] == 'PASS'
    assert result['simulator'].endswith('STANDIN_NOT_REAL_MUJOCO')
    assert result['source_truth_exposed_to_controller'] is False
    assert result['acoustic_observation_count'] > 0
    assert (tmp_path / 'telemetry.csv').exists()
    assert (tmp_path / 'microphones.wav').exists()


def test_standin_silence_prevents_head_motion(experiment, tmp_path):
    result = experiment.run_episode(mode='silence', output_dir=tmp_path, strict=True)
    assert result['status'] == 'PASS'
    assert result['maximum_uncommanded_motion_deg'] < 1
    assert result['acoustic_observation_count'] == 0


def test_standin_opposite_source_reverses_turn(experiment, tmp_path):
    result = experiment.run_episode(source_azimuth_deg=-38., output_dir=tmp_path)
    assert result['status'] == 'PASS'
    assert result['final_head_yaw_deg'] < -20
