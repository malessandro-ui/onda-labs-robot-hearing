"""Experimental far-field planar SRP-PHAT candidate detector.

Multiple peaks are *hypotheses*, NOT certified source-count estimates. Uses
channel-pair coherence, local peak prominence, an angular exclusion zone and
explicit silence rejection. Designed for offline benchmarks, not production SSL.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, DirectionEstimate, SpatialResult
from .interfaces import LocalizationFrame, SourceEvidence

@dataclass(frozen=True)
class SRPConfig:
    speed_of_sound_mps: float = 343.
    angular_resolution_deg: float = 2.
    max_candidates: int = 3
    min_separation_deg: float = 20.
    min_relative_score: float = .55
    min_prominence: float = .012
    min_pair_support_fraction: float = .5
    silence_rms: float = 1e-6
    def __post_init__(self):
        v=(self.speed_of_sound_mps,self.angular_resolution_deg,self.min_separation_deg,self.min_relative_score,self.min_prominence,self.min_pair_support_fraction,self.silence_rms)
        if not all(math.isfinite(x) for x in v) or not (200<self.speed_of_sound_mps<400) or not (0<self.angular_resolution_deg<=15) or not (0<self.min_separation_deg<=180) or not (0<=self.min_relative_score<=1) or self.min_prominence<0 or not (0<=self.min_pair_support_fraction<=1) or self.silence_rms<0 or type(self.max_candidates)!=int or not 1<=self.max_candidates<=8:
            raise ValueError('invalid SRP configuration')

class SRPCandidateLocalizer:
    """Offline candidate generator; does not perform source separation or identity estimation."""
    def __init__(self, config:SRPConfig=SRPConfig()): self.config=config
    @property
    def maximum_candidates(self): return self.config.max_candidates
    def reset(self): pass
    def estimate(self,audio:AudioPacket,array:MicrophoneArray)->LocalizationFrame:
        audio.validate_array(array)
        if array.xy_rank<2: raise ValueError('SRP planar azimuth requires non-collinear XY geometry')
        pcm=audio.pcm.astype(np.float64)
        result_empty=SpatialResult(audio.capture,audio.frame_id,(), 'srp_candidate_v05')
        if len(pcm)<64 or np.sqrt(np.mean(pcm*pcm))<self.config.silence_rms:
            return LocalizationFrame(result_empty,(), 'srp_phat_candidate')
        angles=np.deg2rad(np.arange(-180.,180.,self.config.angular_resolution_deg))
        dirs=np.column_stack((np.cos(angles),np.sin(angles),np.zeros_like(angles)))
        score=np.zeros(len(angles));support=np.zeros(len(angles));npairs=0
        fs=audio.sample_rate_hz
        nfft=1<<(2*len(pcm)-2).bit_length()
        for i,j in combinations(range(array.channels),2):
            x=pcm[:,i]-np.mean(pcm[:,i]);y=pcm[:,j]-np.mean(pcm[:,j])
            if min(np.linalg.norm(x),np.linalg.norm(y))<1e-9:continue
            cross=np.fft.rfft(x,nfft)*np.conj(np.fft.rfft(y,nfft))
            cross/=np.maximum(np.abs(cross),1e-10)
            cc=np.fft.irfft(cross,nfft)
            lags=-(dirs@(array.positions_m[i]-array.positions_m[j]))*fs/self.config.speed_of_sound_mps
            periodic=np.mod(lags,nfft)
            val=np.interp(periodic,np.arange(nfft+1),np.r_[cc,cc[0]])
            score+=val
            support+=(val>0)
            npairs+=1
        if npairs<2:return LocalizationFrame(result_empty,(), 'srp_phat_candidate')
        score/=npairs;support/=npairs
        local=np.flatnonzero((score>np.roll(score,1))&(score>=np.roll(score,-1)))
        if not len(local):return LocalizationFrame(result_empty,(), 'srp_phat_candidate')
        ordered=sorted(local,key=lambda idx:(-float(score[idx]),int(idx)))
        best=float(score[ordered[0]]);median=float(np.median(score))
        if best<=0:return LocalizationFrame(result_empty,(), 'srp_phat_candidate')
        directions=[];evidence=[]
        for k in ordered:
            strength=float(score[k]);prominence=strength-median
            if strength<best*self.config.min_relative_score or prominence<self.config.min_prominence:continue
            if support[k]<self.config.min_pair_support_fraction:continue
            theta=float(angles[k])
            if any(abs((theta-z.azimuth_rad+np.pi)%(2*np.pi)-np.pi)<math.radians(self.config.min_separation_deg) for z in directions):continue
            directions.append(DirectionEstimate(theta,audio.frame_id,max(0,strength)))
            evidence.append(SourceEvidence(theta,strength,prominence,strength/best,int(round(support[k]*npairs)),npairs))
            if len(directions)>=self.config.max_candidates:break
        return LocalizationFrame(SpatialResult(audio.capture,audio.frame_id,tuple(directions),'srp_candidate_v05'),tuple(evidence),'srp_phat_candidate')
