"""Reference acoustic propagation for reproducible offline validation.

Signal convention: source samples are amplitude at 1 m; direct gain = 1/max(r, min_distance).
All positions in meters, all times seconds, array-frame local offsets translated via
world rotation. PCM is float64 during mixing; never clip before delivering float32.
Not a room solver, no diffraction/scattering or automatic hardware calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def _point(x, label):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f'{label} must be finite XYZ in meters')
    return a.copy()


def _rotation(m):
    a = np.asarray(m, dtype=np.float64)
    if a.shape != (3,3) or not np.isfinite(a).all():
        raise ValueError('rotation must be finite 3x3')
    if not np.allclose(a.T @ a, np.eye(3), atol=1e-6) or np.linalg.det(a) < .999999:
        raise ValueError('rotation must be a proper orthonormal rotation')
    return a.copy()


@dataclass(frozen=True)
class PoseTrajectory:
    """World-space array or sound-source poses at monotonically increasing timestamps.

    Linear translation / SO(3) rotation interpolation via normalized quaternion SLERP.
    For static pose, pass a single timestamp. Outside the given range: hold endpoint pose.
    """
    times_s: np.ndarray
    positions_m: np.ndarray
    rotations: np.ndarray | None = None

    def __post_init__(self):
        t = np.array(self.times_s, dtype=np.float64, copy=True)
        p = np.array(self.positions_m, dtype=np.float64, copy=True)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('trajectory timestamps must be nonempty, finite, strictly increasing')
        if p.shape != (len(t), 3) or not np.isfinite(p).all():
            raise ValueError('trajectory positions must have shape [T,3] meters')
        r = np.repeat(np.eye(3)[None], len(t), axis=0) if self.rotations is None else np.array(self.rotations, dtype=np.float64, copy=True)
        if r.shape != (len(t),3,3): raise ValueError('rotations must have shape [T,3,3]')
        for one in r: _rotation(one)
        t.setflags(write=False); p.setflags(write=False); r.setflags(write=False)
        object.__setattr__(self, 'times_s', t)
        object.__setattr__(self, 'positions_m', p)
        object.__setattr__(self, 'rotations', r)

    @classmethod
    def stationary(cls, position_m, rotation=None):
        return cls(np.array([0.]), np.asarray([_point(position_m,'position')]),
                   None if rotation is None else np.asarray([_rotation(rotation)]))

    def at(self, t: float):
        if not np.isfinite(t): raise ValueError('time must be finite')
        idx = int(np.searchsorted(self.times_s, t, side='right'))
        if idx == 0: return self.positions_m[0], self.rotations[0]
        if idx >= len(self.times_s): return self.positions_m[-1], self.rotations[-1]
        lo, hi = idx-1,idx
        a = float((t-self.times_s[lo])/(self.times_s[hi]-self.times_s[lo]))
        p = (1-a)*self.positions_m[lo] + a*self.positions_m[hi]
        r0,r1=self.rotations[lo],self.rotations[hi]
        # Proper interpolation on SO(3) via axis-angle relative rotation.
        rel=r0.T @ r1
        cos_theta=np.clip((np.trace(rel)-1)/2,-1,1)
        theta=float(np.arccos(cos_theta))
        if theta < 1e-7:
            rot=r0
        elif abs(math.pi-theta)<1e-5:
            # Near-pi: eigenvector of relative rotation with eigenvalue +1.
            vals,vecs=np.linalg.eig(rel)
            axis=np.real(vecs[:,np.argmin(np.abs(vals-1))]);axis/=np.linalg.norm(axis)
            rot=r0 @ _axis_angle(axis,a*theta)
        else:
            axis=np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])/(2*np.sin(theta))
            rot=r0 @ _axis_angle(axis,a*theta)
        return p, rot


def _axis_angle(v,angle):
    x,y,z=v; k=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


@dataclass(frozen=True)
class AcousticConfig:
    sample_rate_hz: int = 24000
    sound_speed_mps: float = 343.0
    reference_distance_m: float = 1.0
    minimum_distance_m: float = 0.1
    max_samples: int = 2_000_000
    interpolation_half_width: int = 12

    def __post_init__(self):
        if type(self.sample_rate_hz) is not int or not 8000<=self.sample_rate_hz<=384000:
            raise ValueError('sample_rate_hz must be 8000..384000')
        if not (200 < self.sound_speed_mps < 400): raise ValueError('sound_speed_mps must be in (200,400)')
        if not (0 < self.minimum_distance_m <= self.reference_distance_m):
            raise ValueError('require 0 < minimum_distance <= reference_distance')
        if type(self.max_samples) is not int or self.max_samples < 1: raise ValueError('invalid sample limit')
        if type(self.interpolation_half_width) is not int or not 4<=self.interpolation_half_width<=64:
            raise ValueError('interpolation_half_width must be 4..64')


@dataclass(frozen=True)
class PhysicalSource:
    id: str
    signal: np.ndarray
    trajectory: PoseTrajectory
    start_s: float = 0.
    amplitude: float = 1.

    def __post_init__(self):
        a=np.asarray(self.signal,dtype=np.float64)
        if not isinstance(self.id,str) or not self.id or a.ndim!=1 or len(a)<1 or not np.isfinite(a).all():
            raise ValueError('source must have ID and finite nonempty mono signal')
        if not np.isfinite([self.start_s,self.amplitude]).all() or self.amplitude<0:
            raise ValueError('invalid start or amplitude')
        if not isinstance(self.trajectory,PoseTrajectory): raise TypeError('trajectory required')
        a=a.copy();a.setflags(write=False);object.__setattr__(self,'signal',a)


def _sinc_samples(signal, indices, half_width):
    """Finite-support Hann-windowed sinc interpolation with zero extension.
    At integer positions returns exact input samples, including at signal boundaries.
    Does not provide ideal bandlimited interpolation for nonbandlimited content.
    """
    ix=np.asarray(indices,dtype=np.float64)
    base=np.floor(ix).astype(np.int64)
    offsets=np.arange(-half_width+1,half_width+1,dtype=np.int64)
    k=base[:,None]+offsets[None,:]
    delta=ix[:,None]-k
    window=np.where(np.abs(delta)<half_width, .5+.5*np.cos(np.pi*delta/half_width),0)
    coeff=np.sinc(delta)*window
    valid=(k>=0)&(k<len(signal))
    values=signal[np.clip(k,0,len(signal)-1)]
    return (coeff*values*valid).sum(axis=1)


class DirectPathRenderer:
    """Offline free-field renderer. Arbitrary array geometry; trajectory-aware retarded time.

    Supports stationary and moving receivers, and moving sources with subsonic motion.
    Uses fixed-point iteration to solve emission time te=t-receiver_to_source(te)/c.
    No reflection, occlusion, microphone response, self-noise or Doppler beyond delay variation.
    """
    def __init__(self, config: AcousticConfig): self.config=config

    def render(self, microphone_offsets_m, array_trajectory:PoseTrajectory,
               sources:list[PhysicalSource], duration_s:float, *, start_s:float=0.,
               sensor_noise_std:float=0., seed:int=0) -> np.ndarray:
        offsets=np.asarray(microphone_offsets_m,dtype=np.float64)
        if offsets.ndim!=2 or offsets.shape[1]!=3 or len(offsets)<1 or not np.isfinite(offsets).all():
            raise ValueError('microphones must be finite [channels,3] local offsets')
        if not isinstance(array_trajectory,PoseTrajectory):raise TypeError('array trajectory required')
        if not np.isfinite([duration_s,start_s,sensor_noise_std]).all() or duration_s<=0 or sensor_noise_std<0:
            raise ValueError('invalid duration, start, or noise')
        n=round(duration_s*self.config.sample_rate_hz)
        if n<1 or n>self.config.max_samples:raise ValueError('render duration exceeds sample limit')
        if len({s.id for s in sources}) != len(sources): raise ValueError('duplicate source IDs')
        fs=self.config.sample_rate_hz;c=self.config.sound_speed_mps
        times=start_s+np.arange(n,dtype=np.float64)/fs
        output=np.zeros((n,len(offsets)),dtype=np.float64)
        # Batch chunks keep interpolation memory bounded.
        for begin in range(0,n,1024):
            ts=times[begin:begin+1024]
            mpos=np.empty((len(ts),len(offsets),3),dtype=np.float64)
            for j,t in enumerate(ts):
                origin,rot=array_trajectory.at(float(t))
                mpos[j]=origin+offsets@rot.T
            for source in sources:
                for channel in range(len(offsets)):
                    receiver=mpos[:,channel,:]
                    te=ts.copy()
                    for _ in range(20):
                        source_positions=np.array([source.trajectory.at(float(t))[0] for t in te])
                        d=np.linalg.norm(receiver-source_positions,axis=1)
                        new_te=ts-d/c
                        if np.max(np.abs(new_te-te))<1e-10:
                            te=new_te;break
                        te=new_te
                    else:
                        raise ValueError('retarded time did not converge; check source motion')
                    distance=np.maximum(d,self.config.minimum_distance_m)
                    sample_ix=(te-source.start_s)*fs
                    waveform=_sinc_samples(source.signal,sample_ix,self.config.interpolation_half_width)
                    output[begin:begin+len(ts),channel]+=source.amplitude*self.config.reference_distance_m*waveform/distance
        if sensor_noise_std:
            output+=np.random.default_rng(seed).normal(0,sensor_noise_std,size=output.shape)
        return output.astype(np.float32)


def convolve_rir(pcm, rir, *, output_mode='full') -> np.ndarray:
    """Linear multichannel convolution; pcm [frames,sources], rir [taps,sources,mics].
    The RIR *includes* all propagation and gain: do not pass already propagated mic PCM.
    """
    x=np.asarray(pcm,dtype=np.float64);h=np.asarray(rir,dtype=np.float64)
    if x.ndim!=2 or h.ndim!=3 or x.shape[1]!=h.shape[1] or not len(x) or not len(h):
        raise ValueError('pcm must be [frames,sources]; RIR [taps,sources,microphones]')
    if not np.isfinite(x).all() or not np.isfinite(h).all():raise ValueError('nonfinite RIR or samples')
    if output_mode not in ('full','same'):raise ValueError('output_mode must be full or same')
    result=np.zeros((len(x)+len(h)-1,h.shape[2]),dtype=np.float64)
    for s in range(x.shape[1]):
        for m in range(h.shape[2]):
            result[:,m]+=np.convolve(x[:,s],h[:,s,m],mode='full')
    if output_mode=='same': result=result[:len(x)]
    return result.astype(np.float32)


@dataclass(frozen=True)
class AcousticProfile:
    rt60_s: float
    noise_std: float
    channel_gains: np.ndarray
    seed: int
    def __post_init__(self):
        gains=np.array(self.channel_gains,dtype=np.float64,copy=True)
        if gains.ndim!=1 or len(gains)<1 or np.any(gains<=0) or not np.isfinite(gains).all():
            raise ValueError('invalid channel gains')
        gains.setflags(write=False);object.__setattr__(self,'channel_gains',gains)


def sample_acoustic_profile(channels:int, *, seed:int,
                            rt60_range_s=(.2,.6), noise_std_range=(0.,.002),
                            gain_db_range=(-1.,1.)) -> AcousticProfile:
    """Seeded parameter sampling, not itself a physical room simulation.
    Ranges must come from measured or justified calibration sets.
    """
    if type(channels) is not int or not 1<=channels<=256:raise ValueError('invalid channel count')
    for name,(lo,hi) in [('rt60',rt60_range_s),('noise',noise_std_range),('gain',gain_db_range)]:
        if not np.isfinite([lo,hi]).all() or hi<lo:raise ValueError(f'invalid {name} range')
    if rt60_range_s[0]<0 or noise_std_range[0]<0:raise ValueError('RT60/noise must be nonnegative')
    rng=np.random.default_rng(seed)
    rt60=float(rng.uniform(*rt60_range_s));noise=float(rng.uniform(*noise_std_range))
    gains=10**(rng.uniform(*gain_db_range,size=channels)/20)
    return AcousticProfile(rt60,noise,gains,seed)
