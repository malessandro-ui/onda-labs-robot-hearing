"""USD/Isaac Sim stage binding, no import-time Isaac or pxr dependency.

Caller owns the Isaac Sim SimulationApp, physics stepping and USD stage.
Snapshot reads world transforms from prims after each physics step.
"""
from __future__ import annotations
import numpy as np
from onda_hearing.runtime import SceneSnapshot,xyzw_from_rotation

class IsaacSimSceneAdapter:
    def __init__(self,stage,*,array_prim:str,source_prims:dict[str,str]):
        if not array_prim.startswith('/') or not isinstance(source_prims,dict):
            raise ValueError('absolute USD array prim and source mapping required')
        if any(not isinstance(k,str) or not k or not isinstance(v,str) or not v.startswith('/') for k,v in source_prims.items()):
            raise ValueError('source prim paths must be absolute')
        try:
            from pxr import Usd,UsdGeom
        except ImportError as exc:
            raise RuntimeError('Isaac Sim adapter requires pxr in an Isaac Sim Python runtime') from exc
        self.stage=stage;self.Usd=Usd;self.UsdGeom=UsdGeom
        self.array_prim=self._prim(array_prim)
        self.source_prims={key:self._prim(path) for key,path in source_prims.items()}

    def _prim(self,path):
        prim=self.stage.GetPrimAtPath(path)
        if not prim.IsValid():raise ValueError(f'USD prim is missing: {path}')
        return prim

    def _transform(self,prim,cache):
        mat=cache.GetLocalToWorldTransform(prim)
        t=mat.ExtractTranslation()
        q=mat.ExtractRotationQuat()
        imag=q.GetImaginary()
        quaternion=(float(imag[0]),float(imag[1]),float(imag[2]),float(q.GetReal()))
        # Pose is local-to-world; quaternion reflects the USD world transform.
        return (float(t[0]),float(t[1]),float(t[2])),quaternion

    def snapshot(self,time_ns:int)->SceneSnapshot:
        if type(time_ns) is not int or time_ns<0:raise ValueError('invalid time_ns')
        cache=self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        pos,quat=self._transform(self.array_prim,cache)
        sources={name:self._transform(prim,cache)[0] for name,prim in self.source_prims.items()}
        return SceneSnapshot(time_ns,pos,quat,sources)
