"""Deterministic static free-field waveform simulator, with overlapping sources.

Engineering scope: direct paths, constant source/array pose per rendering call.
No room, diffraction, occlusion, moving-source Doppler, or hardware calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import wave
import numpy as np
from .core import AudioBlock, MicrophoneArray, vec3


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read mono/stereo PCM WAV, return float32 [samples, input channels]."""
    with wave.open(str(path), "rb") as w:
        channels, sr, width, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        if w.getcomptype() != "NONE" or channels < 1 or width not in (1, 2, 3, 4) or sr <= 0:
            raise ValueError("Only uncompressed 8/16/24/32-bit PCM WAV is supported")
        if n > 25_000_000:
            raise ValueError("WAV exceeds conservative reference loader size limit")
        raw = w.readframes(n)
    if width == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) / 128
    elif width == 2:
        a = np.frombuffer(raw, '<i2').astype(np.float64) / 32768
    elif width == 4:
        a = np.frombuffer(raw, '<i4').astype(np.float64) / 2147483648
    else:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float64) / 8388608
    if a.size % channels:
        raise ValueError("Malformed WAV")
    return a.reshape(-1, channels).astype(np.float32), sr


def save_wav(path: str | Path, pcm: np.ndarray, sr: int) -> None:
    a = np.asarray(pcm, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1] < 1:
        raise ValueError("Expected [samples,channels] PCM")
    raw = np.round(np.clip(a, -1., 1. - 1 / 32768) * 32768).astype('<i2')
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(raw.tobytes())


@dataclass(frozen=True)
class SoundSource:
    """One source at a static world position; mono waveform at output sample rate."""
    name: str
    samples: np.ndarray
    position_world_m: np.ndarray
    start_s: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        x = np.asarray(self.samples, dtype=np.float64)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError("SoundSource samples must be finite mono waveform")
        if not np.isfinite([self.start_s, self.gain]).all() or self.gain < 0:
            raise ValueError("Invalid source timing/gain")
        object.__setattr__(self, 'samples', x.copy())
        object.__setattr__(self, 'position_world_m', vec3(self.position_world_m))


class FreeFieldSimulator:
    def __init__(self, sample_rate_hz: int, speed_of_sound_mps: float = 343.0) -> None:
        if sample_rate_hz <= 0 or not 200 < speed_of_sound_mps < 400:
            raise ValueError("Invalid sample rate or sound speed")
        self.sample_rate_hz = sample_rate_hz
        self.c = speed_of_sound_mps

    def render(self, array: MicrophoneArray, array_origin_world_m: np.ndarray,
               sources: list[SoundSource], duration_s: float, *, t0_s: float = 0.0,
               noise_std: float = 0., seed: int = 0) -> AudioBlock:
        if not np.isfinite([duration_s, t0_s, noise_std]).all() or duration_s <= 0 or noise_std < 0:
            raise ValueError("Invalid duration/start/noise")
        n = int(round(duration_s * self.sample_rate_hz))
        if n < 1 or n > 30_000_000:
            raise ValueError("Invalid or oversized requested render")
        origin = vec3(array_origin_world_m, 'array_origin_world_m')
        mic_positions = array.positions_m + origin[None, :]
        out = np.zeros((n, array.channels), dtype=np.float64)
        timeline = t0_s + np.arange(n, dtype=np.float64) / self.sample_rate_hz
        for source in sources:
            for ch, mic in enumerate(mic_positions):
                distance = float(np.linalg.norm(source.position_world_m - mic))
                distance = max(distance, 0.01)  # avoid singularity; documented near-field floor
                wave_time = timeline - source.start_s - distance / self.c
                # Linear fractional sample delay; no interpolation across source boundaries.
                index = wave_time * self.sample_rate_hz
                values = np.interp(index, np.arange(len(source.samples)), source.samples,
                                   left=0.0, right=0.0)
                out[:, ch] += source.gain * values / distance
        if noise_std:
            out += np.random.default_rng(seed).normal(0, noise_std, size=out.shape)
        return AudioBlock(out.astype(np.float32), self.sample_rate_hz, t0_s, array.frame_id)
