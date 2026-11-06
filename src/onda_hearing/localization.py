"""Conservative planar SRP-PHAT bearing estimation.

Candidates are spatial peaks, never proof of distinct sources. This backend is
intentionally single-source by default, even when asked to return more peaks.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
from .core import AudioBlock, Direction, MicrophoneArray


def gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, sr: int, max_tau_s: float) -> tuple[float, float]:
    """GCC-PHAT delay (a relative to b), with uncalibrated peak magnitude."""
    a, b = np.asarray(sig_a, dtype=np.float64), np.asarray(sig_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or len(a) != len(b) or sr <= 0 or not np.isfinite(max_tau_s) or max_tau_s <= 0:
        raise ValueError('Invalid GCC inputs')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite signal')
    nfft = 1 << (2 * len(a) - 2).bit_length()
    A = np.fft.rfft(a - a.mean(), nfft)
    B = np.fft.rfft(b - b.mean(), nfft)
    cross = A * B.conj()
    cross /= np.maximum(abs(cross), 1e-12)
    cc = np.fft.irfft(cross, nfft)
    extent = min(int(np.ceil(sr * max_tau_s)), nfft // 2 - 1)
    if extent < 1:
        raise ValueError('Insufficient sample rate for specified delay')
    window = np.r_[cc[-extent:], cc[:extent + 1]]
    idx = int(np.argmax(np.abs(window)))
    lag = float(idx - extent)
    if 0 < idx < len(window) - 1:
        z = np.abs(window[idx-1:idx+2])
        denominator = z[0] - 2*z[1] + z[2]
        if abs(denominator) > 1e-12:
            lag += float(np.clip(.5 * (z[0] - z[2]) / denominator, -.5, .5))
    return lag / sr, float(abs(window[idx]))


def _circular_peaks(scores: np.ndarray) -> np.ndarray:
    """Only genuine local maxima; flat plateaus do not create fake peaks."""
    return np.flatnonzero((scores > np.roll(scores, 1)) & (scores >= np.roll(scores, -1)))


def estimate_azimuth(
    block: AudioBlock, array: MicrophoneArray, *, c: float = 343.,
    angle_step_deg: float = 2., max_sources: int = 1,
    min_separation_deg: float = 20., min_peak_relative: float = .65,
    min_peak_prominence: float = .015, min_pair_coherence: float = .05,
) -> tuple[Direction, ...]:
    """Estimate planar far-field SRP-PHAT spatial hypotheses.

    Conservative rejection removes weak secondary maxima, but cannot certify
    that multiple maxima correspond to independent sound sources. Explicitly
    enable multiple candidate output using max_sources>1. Scores are not probabilities.
    """
    array.validate_azimuth()
    if block.pcm.shape[1] != array.channels or block.frame_id != array.frame_id:
        raise ValueError('PCM channels/frame do not match array')
    if (not np.isfinite(c) or c <= 0 or not 0 < angle_step_deg <= 30 or
        not 1 <= max_sources <= 16 or not 0 < min_separation_deg <= 180 or
        not 0 <= min_peak_relative <= 1 or min_peak_prominence < 0 or
        not 0 <= min_pair_coherence <= 1):
        raise ValueError('Invalid DOA configuration')
    pcm = block.pcm.astype(np.float64)
    if len(pcm) < 32 or np.max(np.abs(pcm)) < 1e-7:
        return ()
    angles = np.deg2rad(np.arange(-180., 180., angle_step_deg))
    dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    nfft = 1 << (2 * len(pcm) - 2).bit_length()
    score = np.zeros(len(angles))
    pair_support = np.zeros(len(angles))
    used = 0
    for i, j in itertools.combinations(range(array.channels), 2):
        delta = array.positions_m[i] - array.positions_m[j]
        if np.linalg.norm(delta) < 1e-8:
            continue
        a, b = pcm[:, i] - pcm[:, i].mean(), pcm[:, j] - pcm[:, j].mean()
        if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9:
            continue
        A, B = np.fft.rfft(a, nfft), np.fft.rfft(b, nfft)
        cross = A * B.conj()
        cross /= np.maximum(np.abs(cross), 1e-12)
        cc = np.fft.irfft(cross, nfft)
        lags = -(dirs @ delta) * block.sample_rate_hz / c
        periodic = np.mod(lags, nfft)
        values = np.interp(periodic, np.arange(nfft+1), np.r_[cc, cc[0]])
        score += values
        pair_support += (values > 0).astype(float)
        used += 1
    if used < 2:
        return ()
    score /= used
    pair_support /= used
    peaks = _circular_peaks(score)
    if not len(peaks):
        return ()
    ordered = sorted(peaks, key=lambda k: (-score[k], k))
    best = float(score[ordered[0]])
    if best <= 0:
        return ()
    baseline = float(np.median(score))
    selected: list[Direction] = []
    for idx in ordered:
        strength = float(score[idx])
        if strength < best * min_peak_relative:
            continue
        if strength - baseline < min_peak_prominence:
            continue
        if pair_support[idx] < min_pair_coherence:
            continue
        if any(abs((angles[idx] - s.azimuth_rad + np.pi) % (2*np.pi) - np.pi) < math.radians(min_separation_deg) for s in selected):
            continue
        selected.append(Direction(float(angles[idx]), strength, block.frame_id))
        if len(selected) >= max_sources:
            break
    return tuple(selected)
