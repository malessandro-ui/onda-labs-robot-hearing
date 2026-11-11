#!/usr/bin/env python3
"""Fail-closed comparison of reference and optimized native test outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

WAV_NAMES = (
    'mujoco_door_unified_4ch_float.wav',
    'mujoco_door_direct_only_4ch_float.wav',
    'mujoco_door_reflections_only_4ch_float.wav',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--optimized', required=True, type=Path)
    p.add_argument('--atol', type=float, default=1.e-6)
    args = p.parse_args()
    if not (0. <= args.atol <= 1.e-3):
        p.error('atol must be between 0 and 1e-3')
    a = json.loads((args.reference / 'metrics.json').read_text())
    b = json.loads((args.optimized / 'metrics.json').read_text())
    assert a['result'] == b['result'] == 'PASS', (a['result'], b['result'])
    for field in ('door_motion_span_rad', 'active_specular_paths_max',
                  'audio_samples_per_channel'):
        assert abs(a[field] - b[field]) < 1.e-9, (field, a[field], b[field])
    for f in ('440.0', '3000.0'):
        aa = a['open_minus_closed_band_db'][f]
        bb = b['open_minus_closed_band_db'][f]
        assert abs(aa - bb) < 0.01, (f, aa, bb)
    max_abs = 0.
    for name in WAV_NAMES:
        sr_a, x = wavfile.read(args.reference / name)
        sr_b, y = wavfile.read(args.optimized / name)
        assert sr_a == sr_b == 16000, (name, sr_a, sr_b)
        assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 4, name
        assert np.isfinite(x).all() and np.isfinite(y).all(), name
        diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        max_abs = max(max_abs, diff)
        print(f'{name}: max_abs_diff={diff:.3g}')
        assert diff <= args.atol, (name, diff, args.atol)
    seconds = b['physics_and_render_wall_s']
    ref_seconds = a['physics_and_render_wall_s']
    print(json.dumps({
        'result': 'PASS',
        'max_abs_pcm_difference': max_abs,
        'reference_wall_s': ref_seconds,
        'optimized_wall_s': seconds,
        'wall_speedup': ref_seconds / seconds,
        'warning': 'Cross-run runtime is affected by CPU contention and warm caches; repeat unprofiled.',
    }, indent=2))

if __name__ == '__main__':
    main()
