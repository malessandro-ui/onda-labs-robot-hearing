"""Onda v0.20.2 opt-in plane-cache performance trial.

Caches reflection plane `(origin, unit normal)` for immutable AcousticSurface
objects. Coordinates, winding, planarity checks and tolerances are entirely
owned by the installed reference `_plane()` implementation. This adds no
acoustic approximation and is intentionally NOT globally enabled.

The surface dataclass (and its nested material) is frozen and contains tuples;
its content-based hash ensures replacing a surface with moved vertices cannot
reuse a stale plane. The cache is bounded to avoid long-scene memory growth.

Do not mutate cached NumPy arrays; they are returned read-only. The only
supported setup is a single-process opt-in test harness while render threads
are not running. Production integration should use an explicit dependency
injection or renderer-owned plane cache rather than runtime monkey-patching.
"""
from __future__ import annotations

from functools import lru_cache, wraps
from typing import Callable

from .physical import AcousticSurface


def make_cached_plane(reference_plane: Callable, *, maxsize: int = 512):
    """Wrap the exact reference plane calculation with a bounded LRU cache.

    Raises on invalid inputs exactly as the reference normally does, except
    cached arrays are immutable to prevent accidental cross-call mutations.
    """
    if not callable(reference_plane):
        raise TypeError('reference_plane must be callable')
    if not isinstance(maxsize, int) or not 1 <= maxsize <= 8192:
        raise ValueError('maxsize must be 1..8192')

    @lru_cache(maxsize=maxsize)
    @wraps(reference_plane)
    def cached(surface: AcousticSurface):
        if not isinstance(surface, AcousticSurface):
            raise TypeError('surface must be AcousticSurface')
        a, n = reference_plane(surface)
        # Ensure nobody corrupts a future result via in-place modification.
        a.setflags(write=False)
        n.setflags(write=False)
        return a, n

    return cached
