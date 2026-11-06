"""Bounded, auditable acoustic RIR contracts and first-order shoebox image paths.

The image-source solver below models specular paths for static source/receiver
pairs in a rectangular room. Reflection gains use frequency-dependent energy
absorption, transformed into an *approximate* linear-phase FIR filter. It does
not model scattering, diffraction, phase-varying wall impedance, or higher-order
reflections; measured RIRs are the preferred realism validation input.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from onda_hearing.physics.propagation import convolve_rir

BANDS_HZ = (125., 250., 500., 1000., 2000., 4000., 8000.)

def _vec3(v,name):
    a=np.asarray(v,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError(f'{name} must be finite XYZ in meters')
    return a.copy()

@dataclass(frozen=True)
class RoomMaterial:
    """Octave-band energy absorption [0,1], NOT pressure reflectivity."""
    absorption: tuple[float,...]
    def __post_init__(self):
        a=np.asarray(self.absorption,dtype=float)
        if a.shape!=(len(BANDS_HZ),) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
            raise ValueError(f'absorption must have {len(BANDS_HZ)} finite values in [0,1]')
        object.__setattr__(self,'absorption',tuple(float(x) for x in a))

@dataclass(frozen=True)
class ShoeboxRoom:
    dimensions_m: tuple[float,float,float]
    walls: tuple[RoomMaterial,...]
    def __post_init__(self):
        d=_vec3(self.dimensions_m,'dimensions')
        if np.any(d<=0):raise ValueError('room dimensions must be positive')
        if len(self.walls)!=6 or not all(isinstance(w,RoomMaterial) for w in self.walls):
            raise ValueError('exactly six room materials required: x0,x1,y0,y1,z0,z1')
        object.__setattr__(self,'dimensions_m',tuple(d))

@dataclass(frozen=True)
class MeasuredRIR:
    """Immutable [taps,source,channel] RIR with provenance and channel ordering.

    `rir` represents the COMPLETE source-to-microphone transfer function,
    including direct arrival. Must not be convolved with already-propagated PCM.
    """
    rir: np.ndarray
    sample_rate_hz: int
    microphone_names: tuple[str,...]
    source_ids: tuple[str,...]
    provenance: str
    def __post_init__(self):
        h=np.array(self.rir,dtype=np.float64,copy=True)
        if h.ndim!=3 or min(h.shape)<1 or not np.isfinite(h).all() or np.max(np.abs(h))>1e6:
            raise ValueError('RIR must be finite nonempty [taps,sources,channels]')
        if not isinstance(self.sample_rate_hz,int) or not 8000<=self.sample_rate_hz<=384000:
            raise ValueError('invalid sample rate')
        if (len(self.microphone_names)!=h.shape[2] or len(self.source_ids)!=h.shape[1] or
            len(set(self.microphone_names))!=len(self.microphone_names) or
            len(set(self.source_ids))!=len(self.source_ids) or
            not all(isinstance(s,str) and s for s in (*self.microphone_names,*self.source_ids))):
            raise ValueError('RIR source and microphone order must be explicitly unique')
        if not self.provenance or not isinstance(self.provenance,str):raise ValueError('RIR provenance required')
        h.setflags(write=False);object.__setattr__(self,'rir',h)
    def render(self,source_pcm,*,sample_rate_hz:int,source_ids, microphone_names,output_mode='full'):
        if sample_rate_hz!=self.sample_rate_hz:raise ValueError('RIR sample rate differs: explicit resampling required')
        if tuple(source_ids)!=self.source_ids or tuple(microphone_names)!=self.microphone_names:
            raise ValueError('RIR ordering mismatch; cannot silently reorder sources or microphones')
        return convolve_rir(source_pcm,self.rir,output_mode=output_mode)
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,rir=self.rir,sample_rate_hz=self.sample_rate_hz,
                            microphone_names=np.asarray(self.microphone_names),source_ids=np.asarray(self.source_ids),
                            provenance=np.asarray(self.provenance))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(z['rir'],int(z['sample_rate_hz']),tuple(str(v) for v in z['microphone_names']),
                       tuple(str(v) for v in z['source_ids']),str(z['provenance']))

def _fractional_impulse(delay_samples,length,half_width=24):
    """Zero-extended windowed sinc; centered at the requested causal arrival."""
    k=np.arange(length,dtype=float);d=k-delay_samples
    h=np.sinc(d)*np.where(np.abs(d)<half_width,0.5+0.5*np.cos(np.pi*d/half_width),0.)
    return h

def _reflection_fir(material:RoomMaterial,fs:int,taps=65):
    """Approximate zero-phase magnitude response -> linear-phase windowed FIR.

    Energy absorption alpha corresponds to pressure amplitude sqrt(1-alpha).
    This ignores frequency-dependent impedance phase; noncausal ideal FIR is
    centered with a physical group-delay allowance when inserted into RIR.
    """
    if taps%2!=1 or taps<9:raise ValueError('filter taps must be odd >= 9')
    f=np.fft.rfftfreq(2048,1/fs)
    a=np.sqrt(1-np.asarray(material.absorption))
    mag=np.interp(f,BANDS_HZ,a,left=a[0],right=a[-1])
    spectral=np.fft.irfft(mag,n=2048)
    half=taps//2
    kernel=np.concatenate((spectral[-half:],spectral[:half+1]))*np.hanning(taps)
    return kernel

def image_source_rir(room:ShoeboxRoom,sources_m, microphones_m,*,sample_rate_hz=16000,
                     speed_of_sound_mps=343.,max_duration_s=.25,reference_distance_m=1.,
                     max_order=1):
    """First-order image-source RIR, [taps, source, microphone].

    Mirror paths assume six infinite planes clipped conceptually to shoebox.
    No visibility/occlusion calculation. Delay is windowed sinc; reflections
    have causal group-delay compensation in the convolution representation.
    No normalization or clipping. This is a *reference approximation*, not a
    calibrated room solver; max_order 0 (direct only) or 1 (first reflections).
    """
    if max_order not in (0,1):raise ValueError('only order 0 or 1 is implemented')
    if type(sample_rate_hz)!=int or not 8000<=sample_rate_hz<=192000:raise ValueError('invalid sample rate')
    if not (200<speed_of_sound_mps<400) or not (0<reference_distance_m<=10):raise ValueError('invalid acoustic constants')
    if not np.isfinite(max_duration_s) or not .01<=max_duration_s<=5.:raise ValueError('invalid RIR length')
    src=np.asarray(sources_m,dtype=float);mics=np.asarray(microphones_m,dtype=float)
    if src.ndim!=2 or src.shape[1]!=3 or not len(src) or mics.ndim!=2 or mics.shape[1]!=3 or not len(mics) or not np.isfinite(src).all() or not np.isfinite(mics).all():raise ValueError('sources/mics must be finite [N,3]')
    dims=np.asarray(room.dimensions_m)
    if np.any((src<=0)|(src>=dims)) or np.any((mics<=0)|(mics>=dims)):
        raise ValueError('sources and microphones must be strictly inside the room')
    n=int(np.ceil(max_duration_s*sample_rate_hz));h=np.zeros((n,len(src),len(mics)),dtype=float)
    kernels=[_reflection_fir(w,sample_rate_hz) for w in room.walls]
    mid=len(kernels[0])//2
    for si,s in enumerate(src):
        images=[(s,None)]
        if max_order:
            for axis in range(3):
                for side in range(2):
                    q=s.copy();q[axis]=-s[axis] if side==0 else 2*dims[axis]-s[axis]
                    images.append((q,2*axis+side))
        for mi,m in enumerate(mics):
            for q,wall in images:
                distance=float(np.linalg.norm(q-m))
                if distance<.01:raise ValueError('source-microphone distance must exceed 1 cm')
                delay=distance/speed_of_sound_mps*sample_rate_hz
                gain=reference_distance_m/distance
                if wall is None:
                    h[:,si,mi]+=gain*_fractional_impulse(delay,n)
                else:
                    # Reflection filter relative to its central tap, preserves physical delay.
                    impulse=_fractional_impulse(delay,n)
                    full=np.convolve(impulse,kernels[wall],mode='full')
                    h[:,si,mi]+=gain*full[mid:mid+n]
    return h.astype(np.float64)
