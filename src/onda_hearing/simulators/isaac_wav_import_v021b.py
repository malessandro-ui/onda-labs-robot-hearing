"""Bounded WAV importer. Explicit stereo-to-mono and polyphase resampling.

Input is user-supplied audio; no host paths are accepted from HTTP. The caller
passes a private temporary filename and a generated destination inside a single
controlled audio root. Writes validated 16 kHz mono PCM16 for Onda v0.20.
"""
from __future__ import annotations

from pathlib import Path
import math
import os
import tempfile
import wave
import numpy as np


def import_wav_to_pcm16(source, target, *, sample_rate=16000, max_seconds=600) -> dict:
    try:
        from scipy.signal import resample_poly
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError('WAV upload/conversion requires scipy and soundfile in Isaac environment') from exc
    source=Path(source)
    target=Path(target)
    if source.stat().st_size>64*1024*1024:
        raise ValueError('input WAV exceeds 64 MiB')
    try:
        info=sf.info(str(source))
    except RuntimeError as exc:
        raise ValueError(f'WAV decoder error: {exc}') from exc
    if info.format!='WAV' or not (1<=info.channels<=8):
        raise ValueError('requires a WAV with 1..8 channels')
    if not (8000<=info.samplerate<=192000):
        raise ValueError('input sampling rate must be 8..192 kHz')
    if info.frames<=0 or info.frames/info.samplerate>max_seconds:
        raise ValueError('input WAV duration exceeds configured limit or is empty')
    frames,sr=sf.read(str(source),dtype='float32',always_2d=True)
    if not np.isfinite(frames).all():raise ValueError('WAV contains nonfinite audio')
    mono=np.mean(frames.astype(np.float64),axis=1)
    if sr!=sample_rate:
        g=math.gcd(int(sr),int(sample_rate))
        mono=resample_poly(mono,sample_rate//g,sr//g)
    mono=np.asarray(mono,dtype=np.float64)
    if not len(mono) or not np.isfinite(mono).all():raise ValueError('WAV processing produced invalid samples')
    over=int(np.count_nonzero(np.abs(mono)>1))
    pcm=np.round(np.clip(mono,-1.,1.)*32767.).astype('<i2')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.onda_converted_',suffix='.wav',dir=str(target.parent))
    os.close(fd)
    try:
        with wave.open(tmp,'wb') as f:
            f.setframerate(sample_rate);f.setnchannels(1);f.setsampwidth(2)
            f.writeframes(pcm.tobytes())
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return {'original_channels':info.channels,'original_sample_rate':sr,
            'sample_rate':sample_rate,'output_channels':1,'output_frames':len(pcm),
            'duration_s':len(pcm)/sample_rate,'overrange_samples_before_conversion':over}
