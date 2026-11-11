"""Read-only, bounded USD scene browser and selection inspector for v0.21B.

Never call these functions in the FastAPI worker; all USD calls belong on Kit.
The catalog is cached once per loaded stage and served with bounded pagination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np

from onda_hearing.simulators.isaac_scene_loader import category_from_path
from onda_hearing.simulators.isaac_emitter_registry import validate_prim_path


@dataclass(frozen=True)
class IndexedObject:
    path: str
    type: str
    category: str
    label: str


def catalog_from_stage(stage, *, max_prims=100000) -> list[IndexedObject]:
    if stage is None: return []
    from pxr import UsdGeom
    catalog=[]
    for index, prim in enumerate(stage.Traverse()):
        if index >= max_prims:
            raise ValueError("USD stage exceeds catalog bound; inspect stage composition")
        if not prim.IsA(UsdGeom.Xformable) or not prim.IsActive(): continue
        path=str(prim.GetPath())
        # Ignore Onda's session-layer visualization elements.
        if path.startswith("/OndaMicrophoneRig/"): continue
        if len(path)>1024: continue
        catalog.append(IndexedObject(path, prim.GetTypeName(), category_from_path(path),prim.GetName()))
    catalog.sort(key=lambda x:(x.path.lower(),x.type))
    return catalog


def page_catalog(catalog: list[IndexedObject], *, q="",offset=0,limit=40,category="") -> dict:
    if not isinstance(q,str) or len(q)>256: raise ValueError("search query too long")
    if not isinstance(category,str) or len(category)>80: raise ValueError("category filter too long")
    if type(offset)!=int or not 0<=offset<=100000: raise ValueError("invalid offset")
    if type(limit)!=int or not 1<=limit<=100: raise ValueError("invalid limit")
    needle=q.casefold().strip()
    items=[o for o in catalog if (not needle or needle in o.path.casefold())
           and (not category or o.category==category)]
    return {"total":len(items),"offset":offset,"limit":limit,
            "items":[vars(o).copy() for o in items[offset:offset+limit]]}


def prim_details(stage, path:str) -> dict:
    from pxr import Usd, UsdGeom
    from onda_hearing.acoustic_scene.adapters import IsaacUSDSceneSnapshot
    validate_prim_path(path)
    prim=stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid(): raise KeyError("USD prim not found")
    parent=prim.GetParent()
    out={"prim_path":path,"name":prim.GetName(),"type":prim.GetTypeName(),
         "category_guess":category_from_path(path),"active":bool(prim.IsActive()),
         "parent_path":str(parent.GetPath()) if parent and parent.IsValid() else None,
         "child_count":sum(1 for _ in prim.GetChildren()),
         "has_payload":bool(prim.HasPayload()),"is_instance":bool(prim.IsInstance()),
         "applied_apis":list(prim.GetAppliedSchemas())[:24],
         "emitter_eligible":bool(prim.IsActive() and prim.IsA(UsdGeom.Xformable))}
    if prim.IsA(UsdGeom.Xformable):
        try:
            pose=IsaacUSDSceneSnapshot(stage).capture(path)
            out["position_m"]=[float(v) for v in pose.xyz]
            out["rotation_xyzw"]=[float(v) for v in pose.quat_xyzw]
        except (ValueError, KeyError) as exc:
            out["pose_warning"]=str(exc)[:240]
            out["emitter_eligible"]=False
    if prim.IsA(UsdGeom.Imageable):
        try:
            im=UsdGeom.Imageable(prim)
            out["visibility"]=str(im.ComputeVisibility(Usd.TimeCode.Default()))
            out["purpose"]=str(im.ComputePurpose())
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_,UsdGeom.Tokens.render],useExtentsHint=True)
            r=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            scale=float(UsdGeom.GetStageMetersPerUnit(stage))
            lo=np.asarray(tuple(r.GetMin()),dtype=float)*scale
            hi=np.asarray(tuple(r.GetMax()),dtype=float)*scale
            if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi>=lo):
                out["world_bounds_m"]={"min":lo.tolist(),"max":hi.tolist(),
                                       "size":(hi-lo).tolist()}
        except (ValueError,RuntimeError,TypeError,OverflowError) as exc:
            out["bounds_warning"]=str(exc)[:180]
    if prim.IsA(UsdGeom.Mesh):
        mesh=UsdGeom.Mesh(prim)
        counts=mesh.GetFaceVertexCountsAttr().Get()
        if counts is not None:
            out["polygons"]=len(counts)
            out["estimated_triangles_if_fan_triangulated"]=sum(max(0,int(n)-2) for n in counts)
    return out
