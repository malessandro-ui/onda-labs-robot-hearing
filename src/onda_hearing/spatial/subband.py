"""Experimental frequency-diverse SRP-PHAT candidate inference.

Different estimator from v0.5: compute per-frequency-band spatial spectra,
normalise each band independently, and aggregate direction scores with a
high-percentile statistic. A source dominating a smaller frequency band can
survive global PHAT aggregation. This is NOT source separation, does not
estimate calibrated source probabilities, and cannot certify source count.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, DirectionEstimate, SpatialResult
from .interfaces import LocalizationFrame, SourceEvidence

@dataclass(frozen=True)
class SubbandConfig:
    speed_of_sound_mps: float = 343.0
    angular_resolution_deg: float = 3.0
    max_candidates: int = 3
    min_separation_deg: float = 20.0
    bands: int = 6
    min_relative_score: float = 0.72
    min_prominence: float = 0.07
    silence_rms: float = 1e-6
    min_frequency_hz: float = 300.0
    max_frequency_hz: float = 6500.0

    def __post_init__(self):
        numeric = (self.speed_of_sound_mps,self.angular_resolution_deg,self.min_separation_deg,
                   self.min_relative_score,self.min_prominence,self.silence_rms,
                   self.min_frequency_hz,self.max_frequency_hz)
        if not all(math.isfinite(x) for x in numeric): raise ValueError('nonfinite subband configuration')
        if not (200<self.speed_of_sound_mps<400 and 0<self.angular_resolution_deg<=15
                and 0<self.min_separation_deg<=180 and 0<=self.min_relative_score<=1
                and 0<=self.min_prominence<=1 and self.silence_rms>=0
                and self.min_frequency_hz>=0 and self.max_frequency_hz>self.min_frequency_hz
                and type(self.bands) is int and 2<=self.bands<=24
                and type(self.max_candidates) is int and 1<=self.max_candidates<=8):
            raise ValueError('invalid subband configuration')

class SubbandSRPLocalizer:
    """Frequency-diverse offline planar azimuth candidate estimator; no track IDs."""
    def __init__(self,config:SubbandConfig=SubbandConfig()): self.config=config
    @property
    def maximum_candidates(self): return self.config.max_candidates
    def reset(self): pass

    def estimate(self,audio:AudioPacket,array:MicrophoneArray)->LocalizationFrame:
        audio.validate_array(array)
        if array.xy_rank<2: raise ValueError('non-collinear planar microphone geometry required')
        x=audio.pcm.astype(np.float64)
        empty=LocalizationFrame(SpatialResult(audio.capture,audio.frame_id,(),'subband_srp_v06'),(),'subband_srp_candidate')
        if len(x)<128 or np.sqrt(np.mean(x*x))<self.config.silence_rms:return empty
        fs=audio.sample_rate_hz
        nyquist=fs/2
        if self.config.min_frequency_hz>=nyquist:
            raise ValueError('minimum band frequency is above Nyquist')
        max_hz=min(self.config.max_frequency_hz,nyquist*.94)
        if max_hz<=self.config.min_frequency_hz:raise ValueError('frequency band empty after Nyquist limit')
        nfft=1<<(2*len(x)-2).bit_length()
        freq=np.fft.rfftfreq(nfft,1/fs)
        edges=np.linspace(self.config.min_frequency_hz,max_hz,self.config.bands+1)
        angles=np.deg2rad(np.arange(-180,180,self.config.angular_resolution_deg))
        dirs=np.column_stack((np.cos(angles),np.sin(angles),np.zeros(len(angles))))
        spectra=np.fft.rfft(x-x.mean(axis=0),nfft,axis=0)
        band_scores=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(freq>=lo)&(freq<hi)
            if np.sum(mask)<4:continue
            pair_scores=[]
            for i,j in combinations(range(array.channels),2):
                cross=spectra[:,i]*np.conj(spectra[:,j])
                mag=np.abs(cross)
                active=mask&(mag>1e-12)
                if active.sum()<4:continue
                spec=np.zeros_like(cross)
                spec[active]=cross[active]/mag[active]
                cc=np.fft.irfft(spec,nfft)
                lag=-(dirs@(array.positions_m[i]-array.positions_m[j]))*fs/self.config.speed_of_sound_mps
                periodic=np.mod(lag,nfft)
                pair_scores.append(np.interp(periodic,np.arange(nfft+1),np.r_[cc,cc[0]]))
            if len(pair_scores)<2:continue
            response=np.mean(pair_scores,axis=0)
            # Independent band normalization prevents wide, loud bands from owning the entire score.
            spread=np.max(response)-np.median(response)
            if spread<=1e-12:continue
            band_scores.append((response-np.median(response))/spread)
        if len(band_scores)<2:return empty
        scores=np.stack(band_scores)
        # Top-quartile aggregation retains peaks supported by several bands while
        # accommodating sources with differing spectral content.
        score=np.quantile(scores,.75,axis=0)
        local=np.flatnonzero((score>np.roll(score,1))&(score>=np.roll(score,-1)))
        if not len(local):return empty
        ordered=sorted(local,key=lambda idx:(-float(score[idx]),int(idx)))
        best=float(score[ordered[0]])
        if best<=0:return empty
        directions=[]; evidence=[]
        for k in ordered:
            val=float(score[k]); prominence=val-float(np.median(score))
            if val<self.config.min_relative_score*best or prominence<self.config.min_prominence:continue
            theta=float(angles[k])
            if any(abs((theta-d.azimuth_rad+np.pi)%(2*np.pi)-np.pi)<math.radians(self.config.min_separation_deg) for d in directions):continue
            directions.append(DirectionEstimate(theta,audio.frame_id,max(0,val)))
            supporting=int(np.sum(scores[:,k]>self.config.min_relative_score*np.max(scores,axis=1)))
            evidence.append(SourceEvidence(theta,val,prominence,val/best,supporting,len(band_scores)))
            if len(directions)>=self.config.max_candidates:break
        return LocalizationFrame(SpatialResult(audio.capture,audio.frame_id,tuple(directions),'subband_srp_v06'),tuple(evidence),'subband_srp_candidate')
