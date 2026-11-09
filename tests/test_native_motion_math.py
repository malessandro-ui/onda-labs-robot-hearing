"""CPU-only regression tests; do not import Isaac Lab/Kit."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "tools/native_isaaclab_motion/rotation_math.py"
spec = importlib.util.spec_from_file_location("rotation_math", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_world_direction_rotates_into_local_frame():
    array = [2.0, 0, 2.0]
    source = np.array(array) + [np.cos(np.deg2rad(38)), np.sin(np.deg2rad(38)), 0]
    for yaw in [0, 30, -30, 45, -45, 90]:
        actual = module.local_bearing_deg(array, module.yaw_quaternion_wxyz(yaw), source)
        assert module.angular_error_deg(actual, 38-yaw) < 1e-9


def test_wraparound_and_sign():
    assert module.angular_error_deg(179, -179) == 2
    assert module.angular_error_deg(-179, 179) == 2
    assert abs(module.bearing_from_features([0.5, np.sqrt(0.75), 0.4, 1, 0]) - 30.0) < 1e-8


def test_zero_quaternion_rejected():
    import pytest
    with pytest.raises(ValueError):
        module.inverse_rotate_wxyz([0, 0, 0, 0], [1, 0, 0])
