"""Optional Isaac Sim articulation actuator. No Isaac imports until an action is sent.

Joint angles returned by Isaac Core are radians; USDA authored joint angles may
be degrees. Use `apply_action`, NEVER `set_joint_positions` (teleportation).
"""
from __future__ import annotations

import math
import numpy as np


class IsaacSingleArticulationYaw:
    """Thin adapter for an initialized Isaac Sim SingleArticulation.

    Parameters
    ----------
    articulation: An initialized Isaac Sim SingleArticulation, with get_dof_index,
        get_joint_positions and apply_action methods.
    joint_name: Name of a revolute joint whose positive rotation is world +Z.
    base_yaw_rad: Fixed robot-base yaw. Dynamic bases should supply a custom
        adapter with get_base_yaw_rad() reading measured base orientation.
    """

    def __init__(self, articulation, *, joint_name: str, base_yaw_rad: float = 0.0):
        if not isinstance(joint_name, str) or not joint_name:
            raise ValueError('joint_name is required')
        if not math.isfinite(base_yaw_rad):
            raise ValueError('base yaw must be finite')
        self.articulation = articulation
        self.joint_name = joint_name
        idx = articulation.get_dof_index(joint_name)
        if type(idx) not in (int, np.int32, np.int64) or int(idx) < 0:
            raise ValueError(f'joint not found: {joint_name}')
        self.joint_index = int(idx)
        self._base_yaw_rad = float(base_yaw_rad)

    def get_joint_yaw_rad(self) -> float:
        values = np.asarray(self.articulation.get_joint_positions(), dtype=float).reshape(-1)
        if self.joint_index >= len(values) or not math.isfinite(float(values[self.joint_index])):
            raise RuntimeError('invalid joint position reported by Isaac articulation')
        return float(values[self.joint_index])

    def get_base_yaw_rad(self) -> float:
        return self._base_yaw_rad

    def apply_target_rad(self, target_rad: float) -> None:
        if not math.isfinite(target_rad):
            raise ValueError('target must be finite')
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError as exc:
            raise RuntimeError('An active Isaac Sim Python runtime is required to command articulation') from exc
        action = ArticulationAction(
            joint_positions=np.array([float(target_rad)], dtype=np.float64),
            joint_indices=np.array([self.joint_index], dtype=np.int32),
        )
        self.articulation.apply_action(action)
