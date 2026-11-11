"""Onda v0.21: bounded, conservative USD -> acoustic-scene converter.

Importable without Isaac Sim. USD/pxr is imported only inside USD methods.
Never replaces a visual mesh with a solid AABB or guesses acoustic measurements.
Uses exact mesh topology for shortlisted meshes. The current v0.20 renderer
supports nonplanar occluders but only *planar* specular reflectors.

This is a static USD snapshot adapter, not a Fabric/PhysX pose synchronizer.
Call ``build`` again on scene changes; a future runtime adapter must update
moving geometry from physics state, not assume USD transforms are current.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping
import hashlib
import math

import numpy as np

from onda_hearing.acoustic_scene.physical import AcousticSurface, TransmissionMaterial
from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry
from onda_hearing.acoustic_scene.unified_physics import ReflectionMaterial, _plane
from onda_hearing.acoustic_scene.physical_adapters import triangulate_convex_faces

CATEGORIES = frozenset(("floor", "wall", "ceiling", "rack_shelf", "pallet", "forklift", "column", "other"))


def category_from_path(path: str) -> str:
    """Name-based heuristic for first-pass selection, *not* acoustic truth."""
    s = path.lower()
    if any(t in s for t in ("floor", "groundplane")): return "floor"
    if any(t in s for t in ("wall", "partition")): return "wall"
    if any(t in s for t in ("ceiling", "roof")): return "ceiling"
    if any(t in s for t in ("rack", "shelf", "shelving")): return "rack_shelf"
    if "pallet" in s or "crate" in s: return "pallet"
    if "forklift" in s: return "forklift"
    if "column" in s or "pillar" in s: return "column"
    return "other"


def distance_point_aabb(point, minimum, maximum) -> float:
    p = np.asarray(point, dtype=np.float64)
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    if p.shape != (3,) or lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("point and bounds must be xyz vectors")
    if not np.isfinite(p).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("invalid bounding box")
    return float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0.)))


def scene_identifier(stage) -> str:
    """Stable identifier based on scene's root layer, not a transient stage pointer."""
    ident = str(stage.GetRootLayer().identifier)
    if not ident:
        raise ValueError("USD scene has no stable root layer identifier")
    return hashlib.sha256(ident.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class AcousticMaterialProfile:
    transmission: TransmissionMaterial
    reflection: ReflectionMaterial | None = None


def demo_materials() -> dict[str, AcousticMaterialProfile]:
    """EXPLICITLY UNCALIBRATED example curves, for engineering smoke tests only."""
    bands = (125., 500., 1000., 4000.)
    out = {}
    for category in CATEGORIES:
        # Deliberately generic coefficients: do NOT mistake these for a
        # physically accurate warehouse material dictionary.
        tl = TransmissionMaterial(bands, (8., 12., 16., 20.),
                                  "ONDA_DEMO_SYNTHETIC_UNCALIBRATED")
        refl = ReflectionMaterial(bands, (0.12, 0.15, 0.20, 0.28),
                                  (0.05, 0.08, 0.10, 0.15),
                                  "ONDA_DEMO_SYNTHETIC_UNCALIBRATED")
        out[category] = AcousticMaterialProfile(tl, refl)
    return out


@dataclass(frozen=True)
class SceneLoaderConfig:
    # A local *region of interest*, not the entire 677k-triangle warehouse.
    region_center_m: tuple[float, float, float] = (-1., 3.6, 1.5)
    region_radius_m: float = 12.
    categories: tuple[str, ...] = ("floor", "wall", "ceiling", "rack_shelf", "pallet", "column", "forklift")
    max_surfaces: int = 96
    max_triangles_total: int = 6000
    max_triangles_per_mesh: int = 256
    max_reflectors: int = 12
    min_surface_area_m2: float = 0.025
    skip_collision_meshes: bool = True
    skip_hidden_meshes: bool = True
    # Supplying measured material profiles is strongly preferred.
    allow_uncalibrated_demo_materials: bool = False
    # Explicitly include prims by path if needed, still subject to geometry budgets.
    include_prim_paths: tuple[str, ...] = ()
    exclude_path_substrings: tuple[str, ...] = ()

    def __post_init__(self):
        a = np.asarray(self.region_center_m, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError("region_center_m must be finite xyz")
        if not math.isfinite(self.region_radius_m) or self.region_radius_m <= 0:
            raise ValueError("region_radius_m must be positive")
        if not 1 <= self.max_surfaces <= 2048 or not 1 <= self.max_triangles_per_mesh <= 10000:
            raise ValueError("invalid mesh/surface budgets")
        if self.max_triangles_total < self.max_triangles_per_mesh:
            raise ValueError("total triangle budget must cover one mesh")
        if not 0 <= self.max_reflectors <= 24 or self.max_reflectors > self.max_surfaces:
            raise ValueError("max_reflectors must be <= min(max_surfaces,24)")
        if not math.isfinite(self.min_surface_area_m2) or self.min_surface_area_m2 < 0:
            raise ValueError("invalid area threshold")
        if any(x not in CATEGORIES for x in self.categories):
            raise ValueError("unknown category in categories")
        if len(self.include_prim_paths) > 256 or len(self.exclude_path_substrings) > 256:
            raise ValueError("too many include/exclude selectors")
        if any(not isinstance(p, str) or not p.startswith('/') or len(p) > 1024
               for p in self.include_prim_paths):
            raise ValueError("include_prim_paths must be absolute USD prim paths")
        if any(not isinstance(s, str) or not s or len(s) > 128
               for s in self.exclude_path_substrings):
            raise ValueError("invalid exclusion substring")
        object.__setattr__(self, "region_center_m", tuple(float(x) for x in a))


@dataclass
class AcousticSceneBuild:
    geometry: FastAcousticGeometry
    reflectors: dict[str, ReflectionMaterial]
    scene_id: str
    source_layer: str
    metadata: dict
    warnings: list[str] = field(default_factory=list)

    def public_summary(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "source_layer": self.source_layer,
            "surfaces": len(self.geometry.surfaces),
            "triangles": sum(len(s.triangles) for s in self.geometry.surfaces.values()),
            "reflectors": len(self.reflectors),
            "geometry_revision": self.geometry.revision,
            "metadata": self.metadata,
            "warnings": self.warnings[:30],
        }


@dataclass(frozen=True)
class _Candidate:
    prim: object
    path: str
    category: str
    distance_m: float
    triangles: int


def _mesh_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b-a, c-a), axis=1).sum())


class IsaacSceneLoader:
    """Convert selected composed, visible USD meshes to Onda geometry.

    This method must be called on Isaac Sim's application thread. It reads
    USD mesh points and world transforms; it does not mutate the stage.

    Intentional limitations reported as warnings: unresolved point-instancer
    children, instance-proxy geometry, large/concave meshes, dynamic PhysX
    bodies whose poses do not propagate to USD (Fabric mode).
    """

    def __init__(self, config: SceneLoaderConfig,
                 materials: Mapping[str, AcousticMaterialProfile] | None = None):
        self.config = config
        if materials is None:
            if not config.allow_uncalibrated_demo_materials:
                raise ValueError("supply explicit acoustic materials or opt in to demo synthetic materials")
            materials = demo_materials()
        self.materials = dict(materials)
        for name, profile in self.materials.items():
            if name not in CATEGORIES or not isinstance(profile, AcousticMaterialProfile):
                raise TypeError(f"invalid acoustic material profile: {name}")

    def open_scene(self, asset_path: str):
        """Open a USD scene in an *already running* Isaac Sim 5.1 app.

        The caller must initialize ``SimulationApp`` before this method and
        allow Kit updates to finish loading referenced assets before ``build``.
        Never call while handling a Kit update event (it would replace the
        current stage). For an already opened USD scene, call build(stage).
        """
        if not isinstance(asset_path, str) or not asset_path.strip():
            raise ValueError("asset_path required")
        try:
            from isaacsim.storage.native import get_assets_root_path
            from isaacsim.core.utils.stage import open_stage, get_current_stage
        except ImportError as exc:
            raise RuntimeError("Isaac Sim 5.1 stage utilities unavailable") from exc
        path = asset_path
        if path.startswith('/Isaac/'):
            root = get_assets_root_path()
            if not root:
                raise RuntimeError("Isaac Sim asset root unavailable")
            path = root.rstrip('/') + path
        opened = open_stage(path)
        ok = opened[0] if isinstance(opened, tuple) else opened
        if not ok:
            raise RuntimeError(f"unable to open Isaac Sim USD stage: {asset_path}")
        stage = get_current_stage()
        if stage is None:
            raise RuntimeError("USD stage did not become available")
        return stage

    def build(self, stage, *, time_code=None) -> AcousticSceneBuild:
        try:
            from pxr import Usd, UsdGeom, Gf
        except ImportError as exc:
            raise RuntimeError("USD scene loading requires Isaac Sim / pxr") from exc
        if stage is None:
            raise ValueError("no USD stage is loaded")
        time = Usd.TimeCode.Default() if time_code is None else time_code
        scale = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid USD metersPerUnit")
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        bounds_cache = UsdGeom.BBoxCache(time, purposes, useExtentsHint=True)
        transforms = UsdGeom.XformCache(time)
        skipped = Counter()
        candidates = []
        seen_instancers = 0
        seen_instances = 0
        center = self.config.region_center_m
        for prim in stage.Traverse():
            if prim.IsInstance():
                seen_instances += 1
            if prim.IsA(UsdGeom.PointInstancer):
                seen_instancers += 1
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if path.startswith("/OndaMicrophoneRig/") or path == "/OndaMicrophoneRig":
                skipped["onda_viewport_marker"] += 1
                continue
            category = category_from_path(path)
            forced = path in self.config.include_prim_paths
            if not forced and category not in self.config.categories:
                skipped["excluded_category"] += 1
                continue
            if any(t.lower() in path.lower() for t in self.config.exclude_path_substrings):
                skipped["excluded_path"] += 1
                continue
            if self.config.skip_collision_meshes and "collisionmesh" in path.lower():
                skipped["collision_mesh"] += 1
                continue
            if self.config.skip_hidden_meshes:
                imageable = UsdGeom.Imageable(prim)
                visible = imageable.ComputeVisibility(time)
                if visible == UsdGeom.Tokens.invisible:
                    skipped["hidden"] += 1
                    continue
                if imageable.ComputePurpose() == UsdGeom.Tokens.guide:
                    skipped["usd_guide_mesh"] += 1
                    continue
            try:
                world_range = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                low = np.asarray(tuple(world_range.GetMin()), dtype=float) * scale
                high = np.asarray(tuple(world_range.GetMax()), dtype=float) * scale
                if (not np.isfinite(low).all() or not np.isfinite(high).all()
                        or np.any(low > high)):
                    # Some valid USD meshes omit the extent attribute. Never
                    # silently discard those just because BBoxCache has no
                    # usable bound; calculate exact world-space point AABB.
                    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(time)
                    if pts is None or len(pts) < 3:
                        raise ValueError("mesh lacks valid points and extent")
                    xf = transforms.GetLocalToWorldTransform(prim)
                    world_pts = np.asarray(
                        [tuple(xf.Transform(Gf.Vec3d(*map(float, p)))) for p in pts],
                        dtype=np.float64,
                    ) * scale
                    low, high = world_pts.min(axis=0), world_pts.max(axis=0)
                    skipped["computed_missing_extent"] += 1
                d = distance_point_aabb(center, low, high)
                if d > self.config.region_radius_m and not forced:
                    skipped["outside_region"] += 1
                    continue
                mesh = UsdGeom.Mesh(prim)
                counts = mesh.GetFaceVertexCountsAttr().Get(time)
                if counts is None or len(counts) == 0:
                    skipped["empty_mesh"] += 1
                    continue
                tri_count = sum(max(0, int(n)-2) for n in counts)
                if tri_count > self.config.max_triangles_per_mesh:
                    skipped["needs_acoustic_proxy_too_complex"] += 1
                    continue
                if tri_count <= 0:
                    skipped["zero_triangles"] += 1
                    continue
                candidates.append(_Candidate(prim, path, category, d, tri_count))
            except (ValueError, RuntimeError, TypeError) as exc:
                skipped["bbox_or_topology_error"] += 1
        # Stable, spatially meaningful ranking. Floor/wall are useful early
        # reflectors; exact geometry still governs actual ray intersection.
        priority = {"floor": 0, "wall": 1, "ceiling": 2, "rack_shelf": 3,
                    "column": 4, "pallet": 5, "forklift": 6, "other": 7}
        candidates.sort(key=lambda c: (c.distance_m, priority[c.category], c.triangles, c.path))
        surfaces = []
        profiles = {}
        ids = set()
        triangles_used = 0
        for cand in candidates:
            if len(surfaces) >= self.config.max_surfaces:
                skipped["surface_budget"] += 1
                continue
            if triangles_used + cand.triangles > self.config.max_triangles_total:
                skipped["triangle_budget"] += 1
                continue
            profile = self.materials.get(cand.category)
            if profile is None:
                skipped["no_acoustic_material"] += 1
                continue
            try:
                mesh = UsdGeom.Mesh(cand.prim)
                pts = mesh.GetPointsAttr().Get(time)
                counts = mesh.GetFaceVertexCountsAttr().Get(time)
                indices = mesh.GetFaceVertexIndicesAttr().Get(time)
                if pts is None or counts is None or indices is None:
                    raise ValueError("mesh is missing points/topology")
                world = transforms.GetLocalToWorldTransform(cand.prim)
                vertices = np.asarray(
                    [tuple(world.Transform(Gf.Vec3d(*map(float, p)))) for p in pts],
                    dtype=np.float64,
                ) * scale
                triangles = triangulate_convex_faces(vertices, counts, indices)
                if len(triangles) != cand.triangles:
                    raise ValueError("unexpected triangulation count")
                if _mesh_area(vertices, np.asarray(triangles, dtype=int)) < self.config.min_surface_area_m2:
                    skipped["too_small"] += 1
                    continue
                surface = AcousticSurface(cand.path, tuple(map(tuple, vertices)),
                                          triangles, profile.transmission)
                if surface.surface_id in ids:
                    raise ValueError("duplicate acoustic surface id")
                ids.add(surface.surface_id)
                surfaces.append(surface)
                profiles[surface.surface_id] = profile
                triangles_used += len(triangles)
            except (ValueError, TypeError, IndexError, OverflowError):
                skipped["invalid_or_concave_mesh"] += 1
        if not surfaces:
            raise RuntimeError("no usable acoustic surfaces in region; inspect scene bounds, categories and budgets")
        reflectors = {}
        # Already ordered by distance. No specular surface is inferred from a
        # nonplanar, thick mesh: its full mesh remains an occluder only.
        for surface in surfaces:
            if len(reflectors) >= self.config.max_reflectors:
                break
            prof = profiles[surface.surface_id]
            if prof.reflection is None:
                continue
            try:
                _plane(surface)
            except ValueError:
                skipped["nonplanar_reflector_candidate"] += 1
                continue
            reflectors[surface.surface_id] = prof.reflection
        warnings = []
        if skipped["needs_acoustic_proxy_too_complex"]:
            warnings.append(f"{skipped['needs_acoustic_proxy_too_complex']} complex meshes omitted; explicit acoustic proxies required")
        if seen_instances or seen_instancers:
            warnings.append(f"{seen_instances} instance prims and {seen_instancers} point instancers found; instance-proxy geometry is not expanded")
        if not reflectors:
            warnings.append("No planar specular reflectors selected; scene will render direct/occluded audio only")
        warnings.append("Static USD snapshot only; PhysX/Fabric motion must be synchronized separately")
        if self.config.allow_uncalibrated_demo_materials:
            warnings.append("ONDA_DEMO_SYNTHETIC_UNCALIBRATED coefficients; not real-world acoustic validation")
        return AcousticSceneBuild(
            geometry=FastAcousticGeometry(tuple(surfaces)),
            reflectors=reflectors,
            scene_id=scene_identifier(stage),
            source_layer=str(stage.GetRootLayer().identifier),
            metadata={
                "meters_per_unit": scale,
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                "region_center_m": list(center),
                "region_radius_m": self.config.region_radius_m,
                "selected_by_category": dict(Counter(category_from_path(s.surface_id) for s in surfaces)),
                "skipped": dict(skipped),
                "candidate_meshes": len(candidates),
                "geometry_mode": "exact_bounded_mesh_no_solid_bbox_proxies",
                "material_mode": "demo_uncalibrated" if self.config.allow_uncalibrated_demo_materials else "explicit_user_profiles",
            },
            warnings=warnings,
        )
