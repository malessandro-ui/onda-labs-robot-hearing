"""Pose adapter contracts and an optional MuJoCo integration.

Importing the Onda core does not import MuJoCo or Isaac Sim.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from .core import MicrophoneArray


class PoseProvider(Protocol):
    def world_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return translation [3] and 3x3 rotation local-to-world."""
        ...


@dataclass
class MuJoCoPoseProvider:
    model: object
    data: object

    def world_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        import mujoco  # optional dependency; import only when invoked
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"MuJoCo site not found: {name}")
        return (np.asarray(self.data.site_xpos[site_id], dtype=float).copy(),
                np.asarray(self.data.site_xmat[site_id], dtype=float).reshape(3, 3).copy())


def array_world_positions(array: MicrophoneArray, origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(origin, dtype=float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Invalid rigid transform")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
        raise ValueError("Rotation matrix is not proper orthonormal")
    return (R @ array.positions_m.T).T + t
