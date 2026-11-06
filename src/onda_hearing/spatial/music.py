"""Experimental narrowband covariance MUSIC baseline.

Narrowband far-field planar model, no motion compensation, no calibration,
no source separation. Uses MDL to estimate order without ground-truth count.
Not expected to reliably resolve coherent sources or reverberant mixtures.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, DirectionEstimate, SpatialResult

@dataclass(frozen=True)
class MUSICConfig:
    fft_samples: int = 512
    hop_samples: int = 128
    min_frequency_hz: float = 500.
    max_frequency_hz: float = 3500.
    resolution_deg: float = 3.
    max_sources: int = 3
    min_separation_deg: float = 16.
    min_relative_peak: float = .45
    regularization: float = 1e-4
    speed_mps: float = 343.
    def __post_init__(self):
        if (type(self.fft_samples) is not int or self.fft_samples<128 or
            type(self.hop_samples) is not int or not 1<=self.hop_samples<=self.fft_samples or
            type(self.max_sources) is not int or not 1<=self.max_sources<=16):
            raise ValueError('invalid MUSIC window / source count')
        if not all(map(math.isfinite,(self.min_frequency_hz,self.max_frequency_hz,self.resolution_deg,
            self.min_separation_deg,self.min_relative_peak,self.regularization,self.speed_mps))):
            raise ValueError('nonfinite MUSIC configuration')
        if not (0<self.min_frequency_hz<self.max_frequency_hz and 0<self.resolution_deg<=15 and
                0<self.min_separation_deg<=180 and 0<self.min_relative_peak<=1 and
                0<self.regularization<1 and 200<self.speed_mps<400):
            raise ValueError('invalid MUSIC configuration')

class CovarianceMUSICLocalizer:
    def __init__(self,config: MUSICConfig=MUSICConfig()):self.config=config
    def reset(self)->None:pass
    @property
    def maximum_candidates(self)->int:return self.config.max_sources
    def estimate(self,audio:AudioPacket,array:MicrophoneArray)->SpatialResult:
        audio.validate_array(array)
        cfg=self.config
        if array.xy_rank<2 or array.channels<3:
            raise ValueError('MUSIC requires at least 3 non-collinear XY microphone positions')
        n,m=audio.pcm.shape
        if n<cfg.fft_samples*3:return SpatialResult(audio.capture,array.frame_id,(),'music_covariance_v010')
        if cfg.max_frequency_hz>=audio.sample_rate_hz/2:
            raise ValueError('MUSIC frequency range exceeds Nyquist')
        x=np.asarray(audio.pcm,dtype=np.float64)
        if np.sqrt(np.mean(x*x))<1e-7:return SpatialResult(audio.capture,array.frame_id,(),'music_covariance_v010')
        blocks=np.lib.stride_tricks.sliding_window_view(x,cfg.fft_samples,axis=0)[::cfg.hop_samples]
        blocks=np.transpose(blocks,(0,2,1))
        spectra=np.fft.rfft((blocks-blocks.mean(axis=1,keepdims=True))*np.hanning(cfg.fft_samples)[None,:,None],axis=1)
        bins=np.fft.rfftfreq(cfg.fft_samples,1/audio.sample_rate_hz)
        mask=(bins>=cfg.min_frequency_hz)&(bins<=cfg.max_frequency_hz)
        ids=np.flatnonzero(mask)
        if len(ids)<3:raise ValueError('too few frequency bins for MUSIC')
        angles=np.deg2rad(np.arange(-180,180,cfg.resolution_deg))
        rays=np.stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))],axis=1)
        steering_delays=(rays@array.positions_m.T)/cfg.speed_mps
        pooled=np.zeros(len(angles),dtype=np.float64)
        orders=[];used=0
        # Avoid per-frequency oracle source counts. Each frequency chooses order by MDL.
        for k in ids:
            snapshots=spectra[:,k,:]
            cov=(snapshots.conj().T@snapshots)/len(snapshots)
            # Numeric regularization stabilizes eigendecomposition but should not become acoustic evidence.
            cov+=np.eye(m)*cfg.regularization*max(float(np.trace(cov).real)/m,1e-12)
            values,vectors=np.linalg.eigh(cov)
            values=np.maximum(values.real,1e-15)
            nobs=len(snapshots)
            mdl=[]
            for p in range(min(cfg.max_sources,m-1)+1):
                noise=values[:m-p]
                a=np.mean(noise)
                g=np.exp(np.mean(np.log(noise)))
                mdl.append(-nobs*(m-p)*np.log(max(g/a,1e-15))+.5*p*(2*m-p)*np.log(nobs))
            order=int(np.argmin(mdl))
            if order==0:continue
            orders.append(order)
            noise_vectors=vectors[:,:m-order]
            steering=np.exp(-2j*np.pi*bins[k]*steering_delays)
            # Noise-subspace projection; magnitude invariant to conjugate convention for symmetric geometry.
            projection=steering.conj()@noise_vectors
            residual=np.sum(np.abs(projection)**2,axis=1)
            pseudospectrum=1/np.maximum(residual,1e-9)
            pooled+=pseudospectrum/max(float(np.max(pseudospectrum)),1e-9)
            used+=1
        if used==0:return SpatialResult(audio.capture,array.frame_id,(),'music_covariance_v010')
        pooled/=used
        # Conservative empirical order. Does not certify independence.
        source_count=min(cfg.max_sources,max(1,int(np.median(orders))))
        peaks=np.where((pooled>np.roll(pooled,1))&(pooled>=np.roll(pooled,-1)))[0]
        sorted_peaks=sorted(peaks,key=lambda i:(-pooled[i],int(i)))
        selected=[]
        for i in sorted_peaks:
            if pooled[i]<pooled[sorted_peaks[0]]*cfg.min_relative_peak:break
            theta=float(angles[i])
            if any(abs((theta-p+np.pi)%(2*np.pi)-np.pi)<np.deg2rad(cfg.min_separation_deg) for p in selected):continue
            selected.append(theta)
            if len(selected)>=source_count:break
        return SpatialResult(audio.capture,array.frame_id,
            tuple(DirectionEstimate(float(a),array.frame_id,float(pooled[np.argmin(abs(angles-a))])) for a in selected),'music_covariance_v010')
