#!/usr/bin/env python3
"""Opt-in fast geometry + immutable specular-plane caching trial.

Runs the installed native MuJoCo experiment and its original pass criteria.
The trial is strictly process-scoped, with original Onda source untouched.
Do not run concurrent renders in the same Python interpreter while patched.
"""
from __future__ import annotations

from unittest.mock import patch
import sys

from onda_hearing.acoustic_scene import unified_physics
from onda_hearing.acoustic_scene.cached_planes_v020 import make_cached_plane
from validate_v020_mujoco_moving_door_fast import main as run_fast


def main():
    base = unified_physics._plane
    cached = make_cached_plane(base, maxsize=512)
    print('Geometry backend: fast + cached immutable specular planes', file=sys.stderr)
    with patch.object(unified_physics, '_plane', cached):
        result = run_fast()
    print('Reflection plane cache:', cached.cache_info(), file=sys.stderr)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
