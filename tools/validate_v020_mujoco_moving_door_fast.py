#!/usr/bin/env python3
"""Run the existing, unmodified v0.20 MuJoCo acceptance with fast geometry.

Uses the *installed* original script and its current MJCF (including any
local fixes). Changes only the class used to instantiate AcousticGeometry.
It does NOT alter existing Onda files, door physics, criteria, or materials.
Accepts the same CLI flags as validate_v020_mujoco_moving_door.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry


def main():
    original = Path(__file__).with_name('validate_v020_mujoco_moving_door.py')
    if not original.is_file():
        raise SystemExit(f'Original native validator missing: {original}')
    spec = importlib.util.spec_from_file_location('_onda_v020_native_original', original)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import original validator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Explicit opt-in class binding. All other imported symbols and behavior
    # are retained from the actual local original validator.
    assert module.AcousticGeometry.__name__ == 'AcousticGeometry'
    module.AcousticGeometry = FastAcousticGeometry
    print('Geometry backend: fast_scalar_moller_trumbore_v020', file=sys.stderr)
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
