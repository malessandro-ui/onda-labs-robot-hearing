"""Deterministic, scene-driven, multi-source PCM renderer for robot microphones.

Reference acoustic model: delay, inverse-distance pressure, simple source
cardioid, optional stationary-channel FIR and deterministic noise. Direct path
only: no claims of realistic occlusion/diffraction. Sim-time renderer (not audio
playback). Position/pose queried on a fixed grid; within-grid linearized motion
and two-step retarded emission time. Does NOT expose source truth to the policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from .types import Emitter,MicArray,Pose3,rotate
from .assets import AudioAsset
from .adapters import PoseResolver,resolve,StaticPoseResolver,PoseTimeline

@dataclass(frozen=True)
class RendererConfig:
    sample_rate:int=16000
    sound_speed_m_s:float=343.
    reference_distance_m:float=1.
    min_distance_m:float=.10
    pose_step_samples:int=32
    noise_std:float=0.
    seed:int=0
    clip_output:bool=False
    rir_taps:np.ndarray|None=None # [taps,channels], causal stationary approximation
    def __post_init__(self):
        if not 1000<=self.sample_rate<=192000: raise ValueError('invalid rate')
        for name in ('sound_speed_m_s','reference_distance_m','min_distance_m'):
            x=getattr(self,name)
            if not np.isfinite(x) or x<=0: raise ValueError(f'invalid {name}')
        if not 1<=self.pose_step_samples<=self.sample_rate: raise ValueError('invalid pose resolution')
        if not np.isfinite(self.noise_std) or self.noise_std<0: raise ValueError('invalid noise')
        if self.rir_taps is not None:
            r=np.asarray(self.rir_taps,dtype=np.float32)
            if r.ndim!=2 or not 1<=len(r)<=16384 or not np.isfinite(r).all():
                raise ValueError('RIR must be finite [taps, channels]')
            object.__setattr__(self,'rir_taps',r.copy())


def _stateless_noise(sample_indices, channels, seed, std):
    """Deterministic by absolute sample index and channel, independent of chunks."""
    if std==0: return np.zeros((len(sample_indices),channels),dtype=np.float64)
    counter=np.asarray(sample_indices,dtype=np.uint64)[:,None]*np.uint64(65537)+np.arange(channels,dtype=np.uint64)[None,:]*np.uint64(11400714819323198485)+np.uint64(seed%2**64)
    def mix(x):
        with np.errstate(over='ignore'):
            x=x+np.uint64(0x9e3779b97f4a7c15)
            x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
            x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return x^(x>>np.uint64(31))
    a=(mix(counter)>>np.uint64(11)).astype(np.float64)/(2**53)
    b=(mix(counter+np.uint64(7))>>np.uint64(11)).astype(np.float64)/(2**53)
    return std*np.sqrt(-2*np.log(np.maximum(a,1e-12)))*np.cos(2*np.pi*b)

class AcousticSceneRenderer:
    def __init__(self, config:RendererConfig, microphones:MicArray, emitters:list[Emitter],
                 assets:Mapping[str,AudioAsset], resolver:PoseResolver|None=None):
        self.config=config; self.microphones=microphones; self.emitters=tuple(emitters)
        self.assets=dict(assets);self.resolver=resolver
        if len({e.emitter_id for e in emitters})!=len(emitters): raise ValueError('duplicate emitter id')
        for e in emitters:
            if e.asset_id not in assets: raise KeyError(f'asset {e.asset_id} missing')
            if assets[e.asset_id].sample_rate!=config.sample_rate:
                raise ValueError(f'asset {e.asset_id} rate does not match renderer; resample explicitly')
        if config.rir_taps is not None and config.rir_taps.shape[1]!=len(microphones.local_positions_m):
            raise ValueError('RIR channel count mismatch')
        self.next_sample=0
        self.total_overrange_samples=0
        self.clipped_samples=0
        self.max_abs_sample=0.
        self._scheduled = {e.emitter_id: (None if e.schedule.onset_s is None else int(round(e.schedule.onset_s*config.sample_rate))) for e in emitters}
        self._stop_at = {e.emitter_id: None for e in emitters}
        self._gain_overrides = {}
        self._static_parameters=self._compile_static_path()
        self._tail=np.zeros((max(0,0 if config.rir_taps is None else len(config.rir_taps)-1),
                             len(microphones.local_positions_m)),dtype=np.float64)

    def _compile_static_path(self):
        """Precompute fixed-distance/direction geometry for static offline scenes.
        Only safe for explicit world bindings or immutable StaticPoseResolver.
        """
        resolver=self.resolver
        if resolver is not None and not isinstance(resolver,StaticPoseResolver):
            return None
        if self.microphones.binding.world is None and resolver is None:return None
        try:
            array=resolve(self.microphones.binding,resolver,0.)
            mics=[np.asarray(array.xyz)+rotate(array.quat_xyzw,loc) for loc in self.microphones.local_positions_m]
            rows={}
            for e in self.emitters:
                source=resolve(e.binding,resolver,0.)
                vals=[]
                for m in mics:
                    d=np.asarray(m)-np.asarray(source.xyz)
                    dist=float(np.linalg.norm(d))
                    attenuation=self.config.reference_distance_m/max(self.config.min_distance_m,dist)
                    if e.directivity=='cardioid':
                        fwd=rotate(source.quat_xyzw,(1.,0.,0.))
                        attenuation*=max(0.,.5*(1.+float(np.dot(fwd,d))/max(self.config.min_distance_m,dist)))
                    vals.append((dist*self.config.sample_rate/self.config.sound_speed_m_s,attenuation))
                rows[e.emitter_id]=tuple(vals)
            return rows
        except KeyError:
            return None

    def reset(self):
        """Reset PCM, source triggers, gains, and optional FIR state per episode."""
        self.next_sample=0
        self.total_overrange_samples=0
        self.clipped_samples=0
        self.max_abs_sample=0.
        self._scheduled={e.emitter_id: (None if e.schedule.onset_s is None else int(round(e.schedule.onset_s*self.config.sample_rate))) for e in self.emitters}
        self._stop_at={e.emitter_id: None for e in self.emitters}
        self._gain_overrides.clear()
        self._tail[:]=0

    def _check_emitter(self,emitter_id):
        if emitter_id not in self._scheduled:raise KeyError(f'unknown emitter: {emitter_id}')

    def trigger(self,emitter_id:str,at_sample:int):
        """Start/restart a source at a not-yet-rendered emission sample."""
        self._check_emitter(emitter_id)
        if not isinstance(at_sample,int) or at_sample<self.next_sample:raise ValueError('trigger must be >= next sample')
        self._scheduled[emitter_id]=at_sample
        self._stop_at[emitter_id]=None

    def stop(self,emitter_id:str,at_sample:int):
        """Stop emission; already-emitted sound still has acoustic flight time."""
        self._check_emitter(emitter_id)
        if not isinstance(at_sample,int) or at_sample<self.next_sample:raise ValueError('stop must be >= next sample')
        self._stop_at[emitter_id]=at_sample

    def set_gain_db(self,emitter_id:str,gain_db:float):
        self._check_emitter(emitter_id)
        if not np.isfinite(gain_db) or not -120<=gain_db<=40:raise ValueError('invalid gain')
        self._gain_overrides[emitter_id]=float(gain_db)


    def _poses(self,binding,times):
        if isinstance(self.resolver,PoseTimeline) and binding.object_path is not None:
            p,q=self.resolver.sample_arrays(binding.object_path,times)
            n=len(p)
            if binding.local_offset_m!=(0.,0.,0.):
                p=p+self._rotate_batch(q,np.broadcast_to(np.asarray(binding.local_offset_m),(n,3)))
            if binding.local_rotation_xyzw!=(0.,0.,0.,1.):
                x,y,z,w=q[:,0],q[:,1],q[:,2],q[:,3]
                X,Y,Z,W=binding.local_rotation_xyzw
                q=np.stack((w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,
                            w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z),axis=1)
                q/=np.linalg.norm(q,axis=1,keepdims=True)
            return p,q
        if binding.world is not None:
            return (np.broadcast_to(np.asarray(binding.world.xyz,dtype=np.float64),(len(times),3)),
                    np.broadcast_to(np.asarray(binding.world.quat_xyzw,dtype=np.float64),(len(times),4)))
        # Fallback for custom PoseResolver: precomputed pose grid.
        # Same knots for any render chunk => chunk boundary independence.
        step=self.config.pose_step_samples/self.config.sample_rate
        t=np.asarray(times,dtype=np.float64)
        i=np.floor(np.maximum(t,0)/step).astype(np.int64)
        keys=np.unique(np.concatenate([i,i+1]))
        table={int(k):resolve(binding,self.resolver,float(k)*step) for k in keys}
        lo=np.asarray([table[int(k)].xyz for k in i]); hi=np.asarray([table[int(k+1)].xyz for k in i])
        a=(np.maximum(t,0)/step-i)[:,None]
        # Position interpolated, quaternion via normalized lerp (sufficient for short grid intervals).
        q0=np.asarray([table[int(k)].quat_xyzw for k in i]); q1=np.asarray([table[int(k+1)].quat_xyzw for k in i])
        # Vectorized quaternion SLERP, including antipodal-sign correction.
        dots=np.sum(q0*q1,axis=1,keepdims=True)
        q1=np.where(dots<0,-q1,q1)
        theta=np.arccos(np.clip(np.abs(dots),0.,1.))
        denom=np.sin(theta)
        scale0=np.where(denom>1e-7,np.sin((1-a)*theta)/np.maximum(denom,1e-12),1-a)
        scale1=np.where(denom>1e-7,np.sin(a*theta)/np.maximum(denom,1e-12),a)
        q=scale0*q0+scale1*q1
        q/=np.linalg.norm(q,axis=1,keepdims=True)
        return lo*(1-a)+hi*a,q

    @staticmethod
    def _rotate_batch(q,v):
        qv=q[:,:3]; t=2*np.cross(qv,v)
        return v+q[:,3:]*t+np.cross(qv,t)

    def _emitted(self,emitter,indices):
        asset=self.assets[emitter.asset_id]
        start=self._scheduled[emitter.emitter_id]
        if start is None: return np.zeros_like(indices,dtype=np.float64)
        relative=np.asarray(indices,dtype=np.float64)-start
        valid=relative>=0
        stop=self._stop_at[emitter.emitter_id]
        if stop is not None:valid &= np.asarray(indices,dtype=np.float64)<stop
        if emitter.schedule.duration_s is not None:
            valid &= relative<emitter.schedule.duration_s*self.config.sample_rate
        if emitter.schedule.loop:
            query=np.mod(relative,len(asset.samples))
        else:
            valid &= relative<len(asset.samples)
            query=relative
        values=np.interp(query,np.arange(len(asset.samples)),asset.samples,left=0,right=0)
        return np.where(valid,values,0.)

    def render(self,start_sample:int, frames:int)->np.ndarray:
        if not isinstance(start_sample,int) or start_sample!=self.next_sample:
            raise ValueError('render chunks must be sequential; reset for a new episode')
        if not 1<=frames<=self.config.sample_rate*2: raise ValueError('invalid chunk size (max 2 seconds)')
        c=self.config
        samples=np.arange(start_sample,start_sample+frames,dtype=np.int64)
        time_s=samples.astype(np.float64)/c.sample_rate
        out=np.zeros((frames,len(self.microphones.local_positions_m)),dtype=np.float64)
        if self._static_parameters is not None:
            for emitter in self.emitters:
                if not emitter.enabled:continue
                volume=10**(self._gain_overrides.get(emitter.emitter_id,emitter.gain_db)/20)
                for ch,(delay,attenuation) in enumerate(self._static_parameters[emitter.emitter_id]):
                    out[:,ch]+=volume*attenuation*self._emitted(emitter,samples-delay)
        else:
            receiver_xyz,receiver_q=self._poses(self.microphones.binding,time_s)
            for ch,local in enumerate(self.microphones.local_positions_m):
                mic_world=receiver_xyz+self._rotate_batch(receiver_q,np.broadcast_to(np.asarray(local),(frames,3)))
                for emitter in self.emitters:
                    if not emitter.enabled: continue
                    emission_time=time_s.copy()
                    # Fixed point retarded-time approximation using continuous source trajectory.
                    for _ in range(3):
                        source_xyz,_=self._poses(emitter.binding,emission_time)
                        distance=np.linalg.norm(source_xyz-mic_world,axis=1)
                        emission_time=time_s-distance/c.sound_speed_m_s
                    source_xyz,source_q=self._poses(emitter.binding,emission_time)
                    direction=mic_world-source_xyz
                    distance=np.linalg.norm(direction,axis=1)
                    distance_safe=np.maximum(distance,c.min_distance_m)
                    gain=(10**(self._gain_overrides.get(emitter.emitter_id,emitter.gain_db)/20))*c.reference_distance_m/distance_safe
                    if emitter.directivity=='cardioid':
                        forward=self._rotate_batch(source_q,np.broadcast_to(np.array([1.,0.,0.]),(frames,3)))
                        gain *= .5*(1.+np.sum(forward*direction,axis=1)/distance_safe)
                        gain=np.maximum(gain,0.)
                    out[:,ch]+=gain*self._emitted(emitter,emission_time*c.sample_rate)
        if c.rir_taps is not None:
            taps=c.rir_taps
            result=np.zeros_like(out)
            for ch in range(out.shape[1]):
                extended=np.concatenate([self._tail[:,ch],out[:,ch]])
                result[:,ch]=np.convolve(extended,taps[:,ch],mode='full')[len(self._tail):len(self._tail)+frames]
                if self._tail.shape[0]: self._tail[:,ch]=extended[-len(self._tail):]
            out=result
        out+=_stateless_noise(samples,out.shape[1],c.seed,c.noise_std)
        if not np.isfinite(out).all():
            raise FloatingPointError('nonfinite rendered PCM')
        peak=float(np.max(np.abs(out)))
        self.max_abs_sample=max(self.max_abs_sample,peak)
        overrange=int(np.count_nonzero(np.abs(out)>1.))
        self.total_overrange_samples+=overrange
        if c.clip_output:
            np.clip(out,-1.,1.,out=out)
            self.clipped_samples+=overrange
        self.next_sample+=frames
        return np.ascontiguousarray(out,dtype=np.float32)

    def telemetry(self)->dict:
        return {'samples_rendered_per_channel':self.next_sample,
                'max_abs_sample':self.max_abs_sample,
                'overrange_samples':self.total_overrange_samples,
                'clipped_samples':self.clipped_samples}
