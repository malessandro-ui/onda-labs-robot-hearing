# Onda v0.20.1 opt-in geometric occlusion acceleration

## Evidence from Shadeform `cProfile` (second-order reflections)

- `render`: 320 calls, 48.351 s cumulative.
- `_update_paths`: 320 calls, 45.569 s cumulative.
- `find_specular_path`: 20,480 calls, 45.032 s cumulative.
- `AcousticGeometry.occluders`: 48,640 calls, 40.071 s cumulative.
- `_triangle_intersects_segment`: 502,814 calls, 32.158 s cumulative.
- `numpy.cross`: 665,052 calls, 30.129 s cumulative.
- Profiling slowed the entire run to ~50 s; the separate nonprofiled native run took 23.825 s.

**Cumulative times overlap.** Do not sum them. NumPy's dispatcher and array-shape logic dominates short 3-element cross products. The immediate target is the ray/triangle intersection kernel, not the audio source interpolator.

## Implementation

`FastAcousticGeometry` subclasses the existing `AcousticGeometry` so the v0.19/v0.20 renderer accepts it unchanged. It uses the same double-sided Moller–Trumbore segment test, same tolerances, same per-panel AABB tests, and the same one-hit-per-panel semantics. Triangles and their static edges are compiled to Python float tuples at `put()`. `remove()` also updates the prepared cache. No jit, compiled dependencies, CUDA kernel, material coefficient changes, or shortened simulation run.

The fast wrapper loads the *installed* original `validate_v020_mujoco_moving_door.py` and supplies the accelerated geometry class only at scene creation. It preserves local MJCF edits (including `<compiler angle="radian"/>`) and the original acceptance criteria.

## Performance acceptance

1. Tests and microbenchmarks must show identical hit sets for the same inputs.
2. Run the original native validation and optimized wrapper unprofiled, with different output directories.
3. Compare float WAV samples (all three outputs, four channels) with `compare_v020_fast_outputs.py`; require maximum absolute deviation <= 1e-6 and <= 0.01 dB differences at 440 and 3000 Hz.
4. Compare full-run wall time repeatedly under comparable machine load. No predicted speedup is a verified speedup.
5. Re-run the full Onda test suite and keep the accelerated backend opt-in until this acceptance is complete.

## Important limits

This changes CPU implementation, **not acoustic physics**. It does not add diffraction, room diffusion, transmission through solids, structural vibration or real-world calibrated materials. For large scenes, a proper BVH/triangle batching implementation will be necessary. Reflection candidate generation is still exhaustive, and nonmoving reflective planes are recomputed at every acoustic update. Further acceleration should be separately benchmarked and regression-tested.
