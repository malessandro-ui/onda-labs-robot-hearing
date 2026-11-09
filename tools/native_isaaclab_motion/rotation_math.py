"""Small pure-Python/Numpy quaternion helpers for Onda native-motion validation.

All Isaac Lab quaternions are scalar-first WXYZ. Angles are degrees.
The reference calculations are evaluation-only: never feed them to Onda.
"""
from __future__ import annotations

import math
import numpy as np


def normalize_degrees(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def angular_error_deg(estimate: float, reference: float) -> float:
    return abs(normalize_degrees(estimate - reference))


def yaw_quaternion_wxyz(yaw_deg: float) -> np.ndarray:
    half = math.radians(float(yaw_deg)) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def inverse_rotate_wxyz(quaternion, vector) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("Expected quaternion shape (4,) and vector shape (3,)")
    n = np.linalg.norm(q)
    if not np.isfinite(n) or not np.isclose(n, 1.0, atol=1e-4):
        raise ValueError("Quaternion must be finite, normalized WXYZ")
    if not np.isfinite(v).all():
        raise ValueError("Vector must be finite")
    w = q[0]
    inverse_xyz = -q[1:]
    uv = np.cross(inverse_xyz, v)
    uuv = np.cross(inverse_xyz, uv)
    return v + 2.0 * (w * uv + uuv)


def local_bearing_deg(array_pos_w, array_quat_wxyz, source_pos_w) -> float:
    relative = np.asarray(source_pos_w, dtype=np.float64) - np.asarray(array_pos_w, dtype=np.float64)
    local = inverse_rotate_wxyz(array_quat_wxyz, relative)
    if np.linalg.norm(local[:2]) < 1e-8:
        raise ValueError("Source lies at array origin in XY")
    return normalize_degrees(math.degrees(math.atan2(local[1], local[0])))


def bearing_from_features(feature) -> float:
    feature = np.asarray(feature, dtype=np.float64)
    if feature.shape != (5,) or not np.isfinite(feature).all():
        raise ValueError("Expected five finite policy features")
    if not np.isclose(feature[3], 1.0):
        raise ValueError("Acoustic observation not valid")
    magnitude = math.hypot(float(feature[0]), float(feature[1]))
    if not 0.9 <= magnitude <= 1.1:
        raise ValueError("Invalid sine/cosine pair")
    return normalize_degrees(math.degrees(math.atan2(feature[0], feature[1])))
