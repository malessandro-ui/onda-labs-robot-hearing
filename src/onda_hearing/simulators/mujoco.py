"""MuJoCo site pose adapter; works with live mjData after mj_forward/mj_step.

The MuJoCo package is NOT imported by the SDK core. Caller supplies live model/data.
Site coordinates are copied immediately; no mjData-backed mutable array escapes.
"""
from __future__ import annotations
import numpy as np
from onda_hearing.runtime import SceneSnapshot,xyzw_from_rotation

class MuJoCoSceneAdapter:
    def __init__(self,model,data,*,array_site:str,source_sites:dict[str,str]):
        if not isinstance(array_site,str) or not array_site:raise ValueError('array_site required')
        if not isinstance(source_sites,dict) or any(not k or not v for k,v in source_sites.items()):
            raise ValueError('source_sites must be a nonempty-ID mapping')
        self.model=model;self.data=data
        self.array_site_id=int(model.site(array_site).id)
        self.source_site_ids={source_id:int(model.site(name).id) for source_id,name in source_sites.items()}

    def snapshot(self,time_ns:int)->SceneSnapshot:
        if type(time_ns) is not int or time_ns<0:raise ValueError('time_ns must be nonnegative int')
        pos=np.array(self.data.site_xpos[self.array_site_id],dtype=float,copy=True)
        rot=np.array(self.data.site_xmat[self.array_site_id],dtype=float,copy=True).reshape(3,3)
        sources={name:tuple(np.array(self.data.site_xpos[idx],dtype=float,copy=True)) for name,idx in self.source_site_ids.items()}
        return SceneSnapshot(time_ns,tuple(pos),xyzw_from_rotation(rot),sources)
