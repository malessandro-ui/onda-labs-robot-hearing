"""Isaac SDK bridge tests with a synthetic plant, never labelled native Isaac Sim."""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.experiments import IsaacHearingLoop, HearingLoopConfig
from onda_hearing.runtime import SceneSnapshot, ScheduledSource
from onda_hearing.simulators import IsaacSingleArticulationYaw


class Plant:
    def __init__(self, source_deg=38., exposure_token=None):
        self.yaw = 0.
        self.target = 0.
        self.source_deg = source_deg
        self.exposure_token = exposure_token
        self.commands = []

    def get_joint_yaw_rad(self):
        return self.yaw

    def get_base_yaw_rad(self):
        return 0.0

    def apply_target_rad(self, target_rad):
        self.target = target_rad
        self.commands.append(target_rad)

    def physics_step(self):
        self.yaw += float(np.clip((self.target - self.yaw) * .08, -.02, .02))

    def snapshot(self, time_ns):
        yaw = self.yaw
        b = math.radians(self.source_deg)
        # Marker never reaches the controller: it is intentionally unused.
        return SceneSnapshot(time_ns, (0., 0., 1.),
                             (0., 0., math.sin(yaw / 2), math.cos(yaw / 2)),
                             {'beacon': (1.5 * math.cos(b), 1.5 * math.sin(b), 1.)})


def array():
    return MicrophoneArray.from_positions(
        [[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],
        array_id='head_array', frame_id='robot/head', geometry_id='asym90mm_v014')


def excitation(seconds=3., fs=16000):
    rng = np.random.default_rng(202613)
    n = round(seconds * fs)
    s = rng.normal(size=n)
    z = np.fft.rfft(s)
    f = np.fft.rfftfreq(n, 1 / fs)
    z[(f < 300) | (f > 3800)] = 0
    y = np.fft.irfft(z, n=n)
    return (0.085 * y / max(1, float(np.std(y)) * 3)).astype(np.float32)


def run_fake(source_deg=38., *, silence=False):
    plant = Plant(source_deg)
    sound = np.zeros(48000, dtype=np.float32) if silence else excitation()
    loop = IsaacHearingLoop(array=array(), sources=(ScheduledSource('beacon', sound, 0),),
                             poses=plant, actuator=plant)
    loop.reset()
    for i in range(1, 1201):
        loop.before_step(i)
        plant.physics_step()
        loop.after_step(i)
    return plant, loop


def test_bridge_real_audio_srp_moves_synthetic_plant():
    plant, loop = run_fake()
    assert loop.new_decisions >= 10
    assert loop.accepted_decisions > 0
    assert len(loop.microphone_pcm()) == 38400
    assert loop.microphone_pcm().shape[1] == 4
    assert abs(math.degrees(plant.yaw) - 38) <= 10
    assert any(row.estimated_bearing_deg is not None for row in loop.observations)


def test_bridge_silence_does_not_move_synthetic_plant():
    plant, loop = run_fake(silence=True)
    assert loop.accepted_decisions == 0
    assert abs(plant.yaw) < 1e-10
    assert all(not r.new_acoustic_decision or r.estimated_bearing_deg is None for r in loop.observations)


def test_bridge_opposite_side_uses_microphone_pcm():
    plant, loop = run_fake(source_deg=-38.)
    assert loop.accepted_decisions > 0
    assert math.degrees(plant.yaw) < -25


def test_bridge_enforces_step_order():
    p = Plant()
    loop = IsaacHearingLoop(array=array(), sources=(ScheduledSource('beacon', np.zeros(16000), 0),),
                            poses=p, actuator=p)
    with pytest.raises(RuntimeError):
        loop.before_step(1)
    loop.reset()
    with pytest.raises(ValueError):
        loop.after_step(1)
    loop.before_step(1)
    with pytest.raises(ValueError):
        loop.before_step(2)
    loop.after_step(1)
    with pytest.raises(ValueError):
        loop.after_step(1)
    loop.reset()
    loop.before_step(1)
    p.physics_step()
    loop.after_step(1)


def test_bridge_rejects_bad_cadence():
    with pytest.raises(ValueError):
        HearingLoopConfig(sensor_interval_steps=7, perception_interval_steps=50)
    with pytest.raises(ValueError):
        HearingLoopConfig(window_samples=100)


def test_isaac_articulation_action_is_not_teleportation(monkeypatch):
    class FakeAction:
        def __init__(self, joint_positions, joint_indices):
            self.joint_positions = joint_positions
            self.joint_indices = joint_indices

    types_mod = types.ModuleType('isaacsim.core.utils.types')
    types_mod.ArticulationAction = FakeAction
    monkeypatch.setitem(sys.modules, 'isaacsim', types.ModuleType('isaacsim'))
    monkeypatch.setitem(sys.modules, 'isaacsim.core', types.ModuleType('isaacsim.core'))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils', types.ModuleType('isaacsim.core.utils'))
    monkeypatch.setitem(sys.modules, 'isaacsim.core.utils.types', types_mod)

    class Articulation:
        def __init__(self):
            self.calls = []
            self.set_joint_positions = lambda *args: pytest.fail('teleport API used')
        def get_dof_index(self, name):
            return 3 if name == 'head_yaw' else None
        def get_joint_positions(self):
            return np.array([0.,0.,0.,0.3])
        def apply_action(self, action):
            self.calls.append(action)

    fake = Articulation()
    actuator = IsaacSingleArticulationYaw(fake, joint_name='head_yaw')
    assert actuator.get_joint_yaw_rad() == pytest.approx(.3)
    actuator.apply_target_rad(.65)
    assert len(fake.calls) == 1
    assert fake.calls[0].joint_positions.tolist() == [.65]
    assert fake.calls[0].joint_indices.tolist() == [3]
    with pytest.raises(ValueError):
        actuator.apply_target_rad(float('nan'))


def test_mujoco_xml_explicit_radians_regression():
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'examples' / 'mujoco_hear_and_turn.py'
    tree = ast.parse(source.read_text())
    xml = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'XML' for t in node.targets))
    assert '<compiler angle="radian"/>' in xml
    assert 'range="-1.570796 1.570796"' in xml


def test_mujoco_compiled_joint_limits_when_installed():
    mujoco = pytest.importorskip('mujoco')
    import ast
    source = Path(__file__).resolve().parents[1] / 'examples' / 'mujoco_hear_and_turn.py'
    tree = ast.parse(source.read_text())
    xml = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'XML' for t in node.targets))
    m = mujoco.MjModel.from_xml_string(xml)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'head_yaw')
    assert np.allclose(m.jnt_range[jid], [-math.pi/2, math.pi/2], atol=1e-5)
