"""Lazy world-mesh adapters for Onda v0.19 explicit transmission surfaces.

These extract geometry only. Neither USD nor MuJoCo material visual labels are
assumed to imply a calibrated acoustic transmission loss.
"""
from __future__ import annotations

import numpy as np
from .physical import AcousticSurface, TransmissionMaterial


def triangulate_convex_faces(vertices_m, face_counts, face_indices):
    """Triangulate planar CONVEX faces; reject concave/invalid polygons.

    Complex and concave USD meshes require a proper triangulated import/export
    path; this helper intentionally refuses to silently fill holes or concavities.
    """
    v = np.asarray(vertices_m, dtype=float)
    count = np.asarray(face_counts, dtype=int)
    indices = np.asarray(face_indices, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("invalid world-space vertices")
    if (count.ndim != 1 or not len(count) or np.any(count < 3) or
            len(indices) != int(count.sum()) or
            np.any(indices < 0) or np.any(indices >= len(v))):
        raise ValueError("invalid mesh topology")
    result = []
    cursor = 0
    for n in count:
        face = indices[cursor:cursor+n]
        cursor += n
        if len(set(face.tolist())) != n:
            raise ValueError("duplicate vertices in polygon")
        poly = v[face]
        normals = np.cross(np.roll(poly, -1, axis=0)-poly,
                           np.roll(poly, -2, axis=0)-np.roll(poly, -1, axis=0))
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1.e-11):
            raise ValueError("degenerate polygon edge")
        normal = np.sum(normals, axis=0)
        if np.linalg.norm(normal) <= 1.e-11:
            raise ValueError("nonconvex or degenerate polygon")
        normal = normal / np.linalg.norm(normal)
        if np.any(np.sum(normals * normal, axis=1) <= 1.e-10):
            raise ValueError("nonconvex polygon: triangulate externally")
        if np.any(abs((poly-poly[0]) @ normal) > 1.e-6):
            raise ValueError("nonplanar polygon: triangulate externally")
        for i in range(1, int(n)-1):
            result.append((int(face[0]), int(face[i]), int(face[i+1])))
    return tuple(result)


def surface_from_usd_mesh(stage, prim_path: str, material: TransmissionMaterial,
                          *, time_code=None, surface_id=None, enabled=True):
    """Read world-space USD Mesh at an explicit stage time (meters)."""
    try:
        from pxr import Usd, UsdGeom, Gf
    except ImportError as exc:
        raise ImportError("USD geometry capture requires Isaac's pxr environment") from exc
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise KeyError(f"USD prim not found: {prim_path}")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"USD prim is not a Mesh: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    time = Usd.TimeCode.Default() if time_code is None else time_code
    positions = mesh.GetPointsAttr().Get(time)
    counts = mesh.GetFaceVertexCountsAttr().Get(time)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if positions is None or counts is None or indices is None:
        raise ValueError("USD mesh requires points and face vertex indices")
    xf = UsdGeom.XformCache(time).GetLocalToWorldTransform(prim)
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid USD stage meters-per-unit")
    world = np.asarray(
        [tuple(xf.Transform(Gf.Vec3d(*map(float, p)))) for p in positions],
        dtype=float,
    ) * scale
    triangles = triangulate_convex_faces(world, counts, indices)
    return AcousticSurface(surface_id or prim_path, tuple(map(tuple, world)),
                           triangles, material, enabled)


def surface_from_mujoco_box(model, data, geom_name: str,
                            material: TransmissionMaterial,
                            *, surface_id=None, enabled=True):
    """Capture a MuJoCo box geom as one thin-screen barrier in world meters.

    The box has multiple polygon faces, but crossing the box counts as one
    material layer; this is a simplification for acoustically thin obstacles.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError("MuJoCo geometry capture requires mujoco") from exc
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if idx < 0:
        raise KeyError(f"MuJoCo geom not found: {geom_name}")
    if model.geom_type[idx] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("only MuJoCo box geoms are supported by this adapter")
    half = np.asarray(model.geom_size[idx], dtype=float)
    center = np.asarray(data.geom_xpos[idx], dtype=float)
    rotation = np.asarray(data.geom_xmat[idx], dtype=float).reshape(3, 3)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("invalid box half-size")
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(rotation).all():
        raise ValueError("invalid MuJoCo geometry pose")
    # Counter-clockwise loops for six faces (double-sided intersection anyway).
    corners = np.array([
        (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
        (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1),
    ], dtype=float) * half
    world = (rotation @ corners.T).T + center
    faces = ((0,1,2,3), (4,5,6,7), (0,1,5,4),
             (1,2,6,5), (2,3,7,6), (3,0,4,7))
    triangles = tuple((face[0], face[i], face[i+1])
                      for face in faces for i in (1,2))
    return AcousticSurface(surface_id or f"mujoco:{geom_name}",
                           tuple(map(tuple, world)), triangles, material, enabled)
