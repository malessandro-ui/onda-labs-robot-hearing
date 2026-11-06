"""Experimental motion-compensated planar SRP-PHAT in a world coordinate frame.

Array poses are sampled at each analysis block midpoint. The resulting
steering delays follow the moving array; waveform blocks are never treated
as though captured by one stationary array pose. This is NOT proof of source
count, waveform separation, calibrated probability, or world-track identity.

Assumptions: far-field, horizontal azimuth, rigid synchronized array, known
array extrinsics, known speed of sound, approximately static pose within each
short FFT block. Independent source wavefronts can still mask one another.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
from onda_hearing.contracts import AudioPacket, MicrophoneArray, SpatialResult, DirectionEstimate
from onda_hearing.physics.propagation import PoseTrajectory
from onda_hearing.spatial.interfaces import LocalizationFrame, SourceEvidence

@dataclass(frozen=True)
class MotionSRPConfig:
    block_samples: int = 320
    hop_samples: int = 160
    angular_resolution_deg: float = 3.0
    speed_of_sound_mps: float = 343.0
    max_candidates: int = 3
    min_separation_deg: float = 20.
    min_relative_score: float = .6
    min_prominence: float = .012
    min_pair_support_fraction: float = .5
    silence_rms: float = 1e-6
    max_pose_gap_s: float = .025
    def __post_init__(self):
        for key in ('block_samples','hop_samples','max_candidates'):
            value=getattr(self,key)
            if type(value) is not int or value < 1:raise ValueError(f'invalid {key}')
        if self.block_samples<128 or self.hop_samples>self.block_samples or self.max_candidates>8:
            raise ValueError('invalid window/candidate configuration')
        values=(self.angular_resolution_deg,self.speed_of_sound_mps,self.min_separation_deg,
                self.min_relative_score,self.min_prominence,self.min_pair_support_fraction,
                self.silence_rms,self.max_pose_gap_s)
        if not all(math.isfinite(v) for v in values) or not (0<self.angular_resolution_deg<=15) or not (200<self.speed_of_sound_mps<400) or not (0<self.min_separation_deg<=180) or not (0<=self.min_relative_score<=1) or self.min_prominence<0 or not (0<=self.min_pair_support_fraction<=1) or self.silence_rms<0 or not (0<self.max_pose_gap_s<=1):
            raise ValueError('invalid motion SRP configuration')

class MotionAwareSRPLocalizer:
    """Signal-derived world-frame candidates; caller supplies synchronized poses.

    If pose_clock_id differs from audio clock, caller must synchronize clocks
    beforehand. In contrast to static algorithms, array.positions_m must be
    rigid BODY-frame offsets, not already-transformed world coordinates.
    """
    def __init__(self, config:MotionSRPConfig=MotionSRPConfig()):self.config=config
    @property
    def maximum_candidates(self):return self.config.max_candidates
    def reset(self):pass

    def estimate(self,audio:AudioPacket,array:MicrophoneArray,poses:PoseTrajectory,*,pose_clock_id:str,world_frame_id:str='world')->LocalizationFrame:
        cfg=self.config
        audio.validate_array(array)
        if pose_clock_id!=audio.capture.clock_id:raise ValueError('pose/audio clock mismatch; explicit synchronization required')
        if array.xy_rank<2:raise ValueError('non-collinear XY array geometry required')
        if not world_frame_id or world_frame_id==array.frame_id:raise ValueError('world and array frames must differ')
        x=audio.pcm.astype(np.float64)
        empty=LocalizationFrame(SpatialResult(audio.capture,world_frame_id,(),'motion_srp_v08'),(),'motion_srp_candidate')
        if len(x)<cfg.block_samples:return empty
        if np.sqrt(np.mean(x*x))<cfg.silence_rms:return empty
        fs=audio.sample_rate_hz
        start_s=audio.capture.start_ns/1e9
        indices=list(range(0,len(x)-cfg.block_samples+1,cfg.hop_samples))
        if not indices:return empty
        angles=np.deg2rad(np.arange(-180.,180.,cfg.angular_resolution_deg))
        rays=np.column_stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))])
        offsets=array.positions_m
        pairs=list(combinations(range(array.channels),2))
        # Validate entire required pose span *before* any signal calculations.
        block_rot=[]
        for begin in indices:
            t=start_s+(begin+(cfg.block_samples-1)/2)/fs
            if len(poses.times_s)>1:
                if t<poses.times_s[0] or t>poses.times_s[-1]:raise ValueError('pose extrapolation refused')
                near=float(np.min(np.abs(poses.times_s-t)))
                if near>cfg.max_pose_gap_s+1e-12:raise ValueError('stale pose sample')
            _,r=poses.at(t)
            block_rot.append(r)
        global_score=np.zeros(len(angles),dtype=float)
        global_support=np.zeros(len(angles),dtype=float)
        used_blocks=0
        nfft=1<<(2*cfg.block_samples-2).bit_length()
        window=np.hanning(cfg.block_samples)[:,None]
        for begin,rot in zip(indices,block_rot):
            block=x[begin:begin+cfg.block_samples]
            if np.sqrt(np.mean(block*block))<cfg.silence_rms:continue
            spectra=np.fft.rfft((block-block.mean(axis=0))*window,nfft,axis=0)
            mics=offsets@rot.T
            scores=np.zeros(len(angles));supports=np.zeros(len(angles));active_pairs=0
            for i,j in pairs:
                cross=spectra[:,i]*spectra[:,j].conj()
                mag=np.abs(cross)
                if np.max(mag)<1e-13:continue
                cross=cross/np.maximum(mag,1e-10)
                cc=np.fft.irfft(cross,nfft)
                lags=-(rays@(mics[i]-mics[j]))*fs/cfg.speed_of_sound_mps
                ix=np.mod(lags,nfft)
                response=np.interp(ix,np.arange(nfft+1),np.r_[cc,cc[0]])
                scores+=response
                supports+=(response>0)
                active_pairs+=1
            if active_pairs<2:continue
            global_score+=scores/active_pairs
            global_support+=supports/active_pairs
            used_blocks+=1
        if used_blocks<1:return empty
        score=global_score/used_blocks;support=global_support/used_blocks
        maxima=np.flatnonzero((score>np.roll(score,1))&(score>=np.roll(score,-1)))
        if not len(maxima):return empty
        ordered=sorted(maxima,key=lambda k:(-float(score[k]),int(k)))
        best=float(score[ordered[0]]);median=float(np.median(score))
        if best<=0:return empty
        directions=[];evidence=[]
        for k in ordered:
            strength=float(score[k]);prominence=strength-median
            if strength<best*cfg.min_relative_score or prominence<cfg.min_prominence or support[k]<cfg.min_pair_support_fraction:continue
            theta=float(angles[k])
            if any(abs((theta-d.azimuth_rad+np.pi)%(2*np.pi)-np.pi)<math.radians(cfg.min_separation_deg) for d in directions):continue
            directions.append(DirectionEstimate(theta,world_frame_id,max(0.,strength)))
            evidence.append(SourceEvidence(theta,strength,prominence,strength/best,int(round(support[k]*len(pairs))),len(pairs)))
            if len(directions)>=cfg.max_candidates:break
        return LocalizationFrame(SpatialResult(audio.capture,world_frame_id,tuple(directions),'motion_srp_v08'),tuple(evidence),'motion_srp_candidate')
