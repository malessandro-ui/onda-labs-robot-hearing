"""Adapters return *world-space* object poses at simulation seconds.

Simulator-specific imports are deferred. Physics backends cannot generally
rewind arbitrary object trajectories, so for moving objects use PoseTimeline:
record poses during simulation and interpolate the past for acoustic travel.
"""
from __future__ import annotations
from typing import Protocol
from types import MappingProxyType
import bisect
import numpy as np
from .types import Pose3, Binding, compose_pose

class PoseResolver(Protocol):
    def pose_at(self, object_path: str, time_s: float) -> Pose3: ...


def slerp(a,b,f):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    dot=float(a@b)
    if dot<0: b=-b;dot=-dot
    dot=min(1.,max(-1.,dot))
    if dot>.9995: q=a+f*(b-a)
    else:
        angle=np.arccos(dot); q=(np.sin((1-f)*angle)*a+np.sin(f*angle)*b)/np.sin(angle)
    q=q/np.linalg.norm(q)
    return tuple(q)

class PoseTimeline:
    """Bounded indexed pose history with O(log N) lookup and quaternion SLERP.

    No arbitrary future extrapolation; a tiny, explicit hold-last allowance
    accommodates interpolation grid boundaries at the latest simulator tick.
    Past the bounded history window, earliest value is NOT used silently.
    """
    def __init__(self,maxlen=8192,max_hold_s=.01):
        if maxlen<2:raise ValueError('maxlen >= 2 required')
        if max_hold_s<0 or not np.isfinite(max_hold_s):raise ValueError('invalid max_hold_s')
        self._maxlen=int(maxlen);self.max_hold_s=float(max_hold_s)
        self._times:dict[str,list[float]]={}
        self._poses:dict[str,list[Pose3]]={}

    def append(self,object_path:str,time_s:float,pose:Pose3):
        if not isinstance(object_path,str) or not object_path or time_s<0 or not np.isfinite(time_s):
            raise ValueError('invalid path or pose time')
        if not isinstance(pose,Pose3):raise TypeError('expected Pose3')
        times=self._times.setdefault(object_path,[])
        if times and time_s<=times[-1]:raise ValueError('pose times must strictly increase')
        times.append(float(time_s))
        self._poses.setdefault(object_path,[]).append(pose)
        if len(times)>self._maxlen:
            times.pop(0);self._poses[object_path].pop(0)

    def pose_at(self,object_path:str,time_s:float)->Pose3:
        times=self._times.get(object_path)
        if not times:raise KeyError(f'no poses recorded for {object_path}')
        poses=self._poses[object_path]
        if time_s<times[0] and times[0]>0:
            raise ValueError(f'history expired for {object_path} at {time_s:.6f}s')
        if time_s<=times[0]:return poses[0]
        if time_s>times[-1]+self.max_hold_s+1e-9:
            raise ValueError(f'missing future pose for {object_path} at {time_s:.6f}s')
        if time_s>=times[-1]:return poses[-1]
        i=bisect.bisect_left(times,time_s)
        if i<len(times) and abs(times[i]-time_s)<1e-9:return poses[i]
        f=(time_s-times[i-1])/(times[i]-times[i-1])
        pos=tuple((1-f)*np.asarray(poses[i-1].xyz)+f*np.asarray(poses[i].xyz))
        return Pose3(pos,slerp(poses[i-1].quat_xyzw,poses[i].quat_xyzw,f))

    def sample_arrays(self,object_path:str,times_s):
        """Vectorized world poses for retarded-time acoustic queries.

        Interpolates only within the history, with <=max_hold_s at the last
        pose and startup hold for t<0 if first sample was recorded at t=0.
        """
        query=np.asarray(times_s,dtype=np.float64)
        if query.ndim!=1 or not np.isfinite(query).all():raise ValueError('invalid times')
        ts=self._times.get(object_path)
        if not ts:raise KeyError(object_path)
        if np.min(query)<ts[0] and ts[0]>0:
            raise ValueError(f'history expired for {object_path}')
        if np.max(query)>ts[-1]+self.max_hold_s+1e-9:
            raise ValueError(f'missing future pose for {object_path}')
        # Extract only nearby physics ticks, not the full 8192-tick history.
        lo=max(0,bisect.bisect_left(ts,float(np.min(query)))-1)
        hi=min(len(ts),bisect.bisect_right(ts,float(np.max(query)))+1)
        data=self._poses[object_path][lo:hi]
        t=np.asarray(ts[lo:hi],dtype=np.float64)
        pos=np.asarray([p.xyz for p in data],dtype=np.float64)
        quat=np.asarray([p.quat_xyzw for p in data],dtype=np.float64)
        right=np.searchsorted(t,query,side='left').clip(0,len(t)-1)
        left=np.maximum(0,right-1)
        # force exact last sample for allowable hold; startup uses first sample.
        left=np.where(query>=t[-1],len(t)-1,left)
        right=np.where(query>=t[-1],len(t)-1,right)
        denom=t[right]-t[left]
        alpha=np.divide(query-t[left],denom,out=np.zeros_like(query),where=denom>0)
        alpha=np.clip(alpha,0,1)[:,None]
        p0=pos[left];p1=pos[right]
        q0=quat[left];q1=quat[right]
        dot=np.sum(q0*q1,axis=1,keepdims=True)
        q1=np.where(dot<0,-q1,q1)
        theta=np.arccos(np.clip(np.abs(dot),0.,1.))
        sin_theta=np.sin(theta)
        a=np.where(sin_theta>1e-7,np.sin((1-alpha)*theta)/np.maximum(sin_theta,1e-12),1-alpha)
        b=np.where(sin_theta>1e-7,np.sin(alpha*theta)/np.maximum(sin_theta,1e-12),alpha)
        q=a*q0+b*q1
        q/=np.linalg.norm(q,axis=1,keepdims=True)
        return p0*(1-alpha)+p1*alpha,q

    def reset(self):self._times.clear();self._poses.clear()


class StaticPoseResolver:
    def __init__(self, poses:dict[str,Pose3]): self.poses=MappingProxyType(dict(poses))
    def pose_at(self,object_path,time_s): return self.poses[object_path]


def resolve(binding:Binding, resolver:PoseResolver | None,time_s:float)->Pose3:
    if binding.world is not None: return binding.world
    if resolver is None: raise ValueError('object attachment requires a pose resolver')
    return compose_pose(resolver.pose_at(binding.object_path,time_s),
                        binding.local_offset_m,binding.local_rotation_xyzw)


class MuJoCoPoseSnapshot:
    """Capture world-space body/site poses from mjModel/mjData at the *current* tick.
    Feed results into PoseTimeline for moving bodies; no time travel is implied.
    """
    def __init__(self, model,data): self.model=model; self.data=data
    def capture(self, kind:str,name:str)->Pose3:
        import mujoco
        if kind not in ('body','site'): raise ValueError('kind must be body or site')
        mjtype = mujoco.mjtObj.mjOBJ_BODY if kind=='body' else mujoco.mjtObj.mjOBJ_SITE
        idx = mujoco.mj_name2id(self.model,mjtype,name)
        if idx<0: raise KeyError(name)
        if kind=='body':
            xyz=np.asarray(self.data.xpos[idx]).copy(); q=np.asarray(self.data.xquat[idx]).copy()
            return Pose3(tuple(xyz),tuple(q[[1,2,3,0]]))
        rot=np.asarray(self.data.site_xmat[idx]).reshape(3,3)
        q=np.empty(4,dtype=float)
        mujoco.mju_mat2Quat(q,rot.reshape(-1))
        return Pose3(tuple(self.data.site_xpos[idx]),tuple(q[[1,2,3,0]]))


class IsaacUSDSceneSnapshot:
    """Read a USD prim's composed world pose at the current Isaac tick.
    Feed snapshots into PoseTimeline; USD coordinates converted to meters.
    """
    def __init__(self,stage): self.stage=stage
    def capture(self,prim_path:str)->Pose3:
        from pxr import Usd,UsdGeom
        prim=self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid(): raise KeyError(prim_path)
        xf=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        # World transforms often include scaling (and sometimes shear), which
        # makes direct ExtractRotationQuat() invalid. USD documents that method
        # as intended for rotation-only matrices. Strip scale/shear FIRST.
        t = xf.ExtractTranslation()
        meters = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        if not np.isfinite(meters) or meters <= 0.:
            raise ValueError(f"invalid USD meters-per-unit: {meters}")
        determinant = float(xf.GetDeterminant3())
        if not np.isfinite(determinant) or abs(determinant) < 1e-12:
            raise ValueError(f"degenerate USD world transform for {prim_path}: determinant={determinant}")
        rotation_xf = xf.RemoveScaleShear()
        # RemoveScaleShear() can return its input if decomposition fails. Never
        # silently invent a pose from a matrix that is still non-orthogonal.
        basis = np.array([[float(rotation_xf[i][j]) for j in range(3)]
                          for i in range(3)], dtype=np.float64)
        if (not np.isfinite(basis).all()
                or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5, rtol=1e-5)
                or not np.isclose(np.linalg.det(basis), 1., atol=1e-5, rtol=1e-5)):
            raise ValueError(f"USD world rotation could not be isolated for {prim_path}")
        q = rotation_xf.ExtractRotationQuat()
        im = q.GetImaginary()
        xyzw = np.array((float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())),
                        dtype=np.float64)
        norm = float(np.linalg.norm(xyzw))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"invalid USD world orientation for {prim_path}")
        xyzw /= norm  # Preserve Pose3's strict unit-quaternion validation.
        return Pose3((float(t[0])*meters, float(t[1])*meters, float(t[2])*meters),
                     tuple(float(v) for v in xyzw))
