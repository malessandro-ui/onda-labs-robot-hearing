#!/usr/bin/env python3
"""Paired reference vs fast geometry microbenchmark with output equivalence."""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from onda_hearing.acoustic_scene.physical import AcousticGeometry, AcousticSurface, TransmissionMaterial
from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=int, default=12000)
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    if not 100 <= args.queries <= 200000:
        p.error('queries must be 100..200000')
    mat = TransmissionMaterial((125., 8000.), (10., 25.), 'synthetic_benchmark')
    surfaces = []
    for k in range(5):
        x = float(k) - 2.
        surfaces.append(AcousticSurface.quad(
            f'wall_x{k}', ((x, -3., 0.), (x, 3., 0.),
                           (x, 3., 3.), (x, -3., 3.)), mat))
    for k in range(5):
        y = float(k) - 2.
        surfaces.append(AcousticSurface.quad(
            f'wall_y{k}', ((-3., y, 0.), (3., y, 0.),
                           (3., y, 3.), (-3., y, 3.)), mat))
    ref = AcousticGeometry(tuple(surfaces))
    opt = FastAcousticGeometry(tuple(surfaces))
    rng = np.random.default_rng(args.seed)
    rays = tuple((rng.uniform(-3., 3., 3), rng.uniform(-3., 3., 3))
                 for _ in range(args.queries))
    output = []
    for cls in (ref, opt):
        start = time.perf_counter()
        results = [tuple(s.surface_id for s in cls.occluders(a, b)) for a, b in rays]
        duration = time.perf_counter() - start
        output.append((results, duration))
    identical = output[0][0] == output[1][0]
    report = {
        'equivalent': identical,
        'queries': args.queries,
        'reference_wall_s': output[0][1],
        'optimized_wall_s': output[1][1],
        'occlusion_speedup': output[0][1] / max(output[1][1], 1.e-12),
        'note': 'Microbenchmark only. Use native MuJoCo and PCM comparison for end-to-end assessment.',
    }
    print(json.dumps(report, indent=2))
    if not identical:
        raise SystemExit('FAIL: optimized geometry disagrees with reference')

if __name__ == '__main__':
    main()
