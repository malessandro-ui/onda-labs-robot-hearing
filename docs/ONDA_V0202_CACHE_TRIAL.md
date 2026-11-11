# Onda v0.20.2 — Opt-in immutable reflection-plane cache

## Motivation

The user-supplied Shadeform v0.20.1 `cProfile` output reports about 9.09 s cumulative in 320 unified render calls, 7.15 s in `_update_paths`, 6.69 s in 20,480 `find_specular_path` calls, 3.09 s in 33,283 `_plane` calls, and 2.07 s in 44,912 NumPy `cross` calls. These are nested cumulative durations; they must not be added. It also reports 1.50 s in fast occluder queries. These values include profiling overhead and must not be treated as unprofiled wall-clock timings.

## Exact code change

The opt-in trial wraps the existing `_plane(AcousticSurface)` function with `functools.lru_cache(maxsize=512)`, keyed on the entire immutable, frozen `AcousticSurface` value. The reference code still calculates each plane, performs the same validation and raises on invalid geometry. Subsequent calls for the same surface receive the same plane. A moved reflector, even with an unchanged `surface_id`, has changed vertices and therefore a different cache key. Bounded memory, no persistent cache across runs, read-only returned vectors. No material curves, path search, sound filtering or acoustic tolerances changed.

The demonstration wrapper uses a process-local, temporary patch of `unified_physics._plane` for one native MuJoCo command. Because this patch affects the module namespace, it is **not suitable for production code or concurrent rendering**. A future production implementation should pass an explicit plane/cache provider into the renderer instance and track per-object revisions rather than patching a module global.

## Validation requirements

1. Run the new tests; they verify value equivalence, moving geometry invalidation, cache bounds, failed planarity, and exact PCM equivalence with first- and second-order reflections.
2. Compare unprofiled native MuJoCo fast baseline and cached runs under the same fixed-radian scene and the same audio material coefficients.
3. Run `tools/compare_v020_fast_outputs.py` on the baseline and optimized output directories; maximum per-sample PCM difference should be zero, but it must be **verified on Shadeform**.
4. Repeat unprofiled timings at least three times because system load varies. Compare medians.
5. Re-run the full SDK regression suite.

No measured-room acoustic accuracy is claimed, and cached planes cannot add diffraction, late reverberation, seal leakage or realistic impedance.
