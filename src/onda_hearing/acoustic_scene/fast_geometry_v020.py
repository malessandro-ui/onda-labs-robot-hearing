"""Opt-in, API-compatible accelerated v0.19/v0.20 acoustic occlusion queries.

No acoustic approximation is introduced: ray/surface intersection is the same
finite-segment, double-sided Moller-Trumbore predicate used by v0.19, with
identical tolerance constants. The difference is that tiny 3-vector cross
products run as Python scalar arithmetic, not hundreds of thousands of NumPy
function calls. Static per-triangle edges and surface AABBs are precomputed.

This is an *experimental CPU performance backend*; validation against the
reference AcousticGeometry must precede enabling it in production. It does
NOT introduce diffraction, watertight solid geometry, or mesh acceleration
beyond existing per-surface AABB culling. Geometry must be changed via
put()/remove() and may not be mutated concurrently with render().
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .physical import AcousticGeometry, AcousticSurface


@dataclass(frozen=True, slots=True)
class _PreparedSurface:
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    # (ax,ay,az,e1x,e1y,e1z,e2x,e2y,e2z)
    triangles: tuple[tuple[float, ...], ...]


def _segment_hits_triangle(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    tri: tuple[float, ...],
) -> bool:
    """Same double-sided segment-test tolerances as physical.py v0.19."""
    (ax, ay, az, e1x, e1y, e1z, e2x, e2y, e2z) = tri
    # p = cross(direction, edge2)
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) <= 1.e-11:
        return False
    inv = 1. / det
    tx, ty, tz = ox - ax, oy - ay, oz - az
    u = (tx * px + ty * py + tz * pz) * inv
    if u < -1.e-9 or u > 1. + 1.e-9:
        return False
    # q = cross(tvec, edge1)
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < -1.e-9 or u + v > 1. + 1.e-9:
        return False
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    return 1.e-7 < t < 1. - 1.e-7


class FastAcousticGeometry(AcousticGeometry):
    """AcousticGeometry drop-in for Onda v0.19/v0.20 ray occlusion.

    Surface identity, ordering, scene revision increments, and return types
    remain compatible. Use this subclass at scene *construction*, not by
    monkey-patching v0.19's global ray intersection function.
    """

    def __init__(self, surfaces: tuple[AcousticSurface, ...] = ()):
        self._fast_prepared: dict[str, _PreparedSurface] = {}
        super().__init__(surfaces)

    def put(self, surface: AcousticSurface):
        # Let the reference implementation validate surfaces and increment
        # revisions; never bypass its geometry or material contracts.
        super().put(surface)
        v, t, bmin, bmax = self._prepared[surface.surface_id]
        prepared = []
        for row in t:
            a = v[int(row[0])]
            e1 = v[int(row[1])] - a
            e2 = v[int(row[2])] - a
            prepared.append(tuple(float(k) for k in (*a, *e1, *e2)))
        self._fast_prepared[surface.surface_id] = _PreparedSurface(
            tuple(float(x) for x in bmin),
            tuple(float(x) for x in bmax),
            tuple(prepared),
        )

    def remove(self, surface_id: str):
        super().remove(surface_id)
        del self._fast_prepared[surface_id]

    def occluders(self, start: object, end: object) -> tuple[AcousticSurface, ...]:
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('start must have shape (3,), all finite')
        if b.shape != (3,) or not np.isfinite(b).all():
            raise ValueError('end must have shape (3,), all finite')
        ox, oy, oz = float(a[0]), float(a[1]), float(a[2])
        dx, dy, dz = float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2])
        if dx * dx + dy * dy + dz * dz <= 1.e-18:
            return ()
        ex, ey, ez = ox + dx, oy + dy, oz + dz
        low_x, high_x = min(ox, ex), max(ox, ex)
        low_y, high_y = min(oy, ey), max(oy, ey)
        low_z, high_z = min(oz, ez), max(oz, ez)
        crossed = []
        for sid, surface in self.surfaces.items():
            if not surface.enabled:
                continue
            item = self._fast_prepared[sid]
            lo, hi = item.box_min, item.box_max
            if (high_x < lo[0] or low_x > hi[0] or
                high_y < lo[1] or low_y > hi[1] or
                high_z < lo[2] or low_z > hi[2]):
                continue
            for tri in item.triangles:
                if _segment_hits_triangle(ox, oy, oz, dx, dy, dz, tri):
                    crossed.append(surface)
                    break  # One transmission loss per physical panel.
        return tuple(crossed)
