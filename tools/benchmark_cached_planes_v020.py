#!/usr/bin/env python3
"""CPU microbenchmark of repeated reflection plane construction.

This is NOT an end-to-end MuJoCo throughput measurement.
"""
import argparse
import json
import time
import numpy as np

from onda_hearing.acoustic_scene.physical import AcousticSurface, TransmissionMaterial
from onda_hearing.acoustic_scene import unified_physics
from onda_hearing.acoustic_scene.cached_planes_v020 import make_cached_plane


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=30000)
    args=parser.parse_args()
    if not 100 <= args.iterations <= 1000000:
        parser.error('iterations must be 100..1000000')
    material=TransmissionMaterial((125.,8000.),(45.,60.),'synthetic benchmark')
    surfaces=[AcousticSurface.quad(f'wall{i}',((-2.,float(i),0.),(2.,float(i),0.),
            (2.,float(i),3.),(-2.,float(i),3.)),material) for i in range(6)]
    cache=make_cached_plane(unified_physics._plane)

    # Both arms receive identical surfaces and ordering.
    t0=time.perf_counter()
    direct=[unified_physics._plane(surfaces[i%6]) for i in range(args.iterations)]
    t1=time.perf_counter()
    opt=[cache(surfaces[i%6]) for i in range(args.iterations)]
    t2=time.perf_counter()
    same=all(np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1])
             for x,y in zip(direct,opt))
    report={'equivalent':same,'iterations':args.iterations,
       'reference_wall_s':t1-t0,'cached_wall_s':t2-t1,
       'speedup':(t1-t0)/max(t2-t1,1e-12),
       'cache_hits':cache.cache_info().hits,'cache_misses':cache.cache_info().misses,
       'scope':'repeated plane extraction only; not render throughput'}
    print(json.dumps(report,indent=2))
    return 0 if same else 1

if __name__=='__main__':
    raise SystemExit(main())
