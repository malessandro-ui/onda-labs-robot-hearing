"""Validated mono assets. No implicit sample-rate conversion or hidden mixing."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import wave
import numpy as np

@dataclass(frozen=True)
class AudioAsset:
    asset_id: str
    samples: np.ndarray
    sample_rate: int

    def __post_init__(self):
        if not self.asset_id or not 1000 <= self.sample_rate <= 192000:
            raise ValueError('invalid audio asset metadata')
        a = np.asarray(self.samples)
        if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():
            raise ValueError('audio asset must contain finite nonempty mono samples')
        a = np.array(a, dtype=np.float32,copy=True)
        if np.max(np.abs(a)) > 1.00001:
            raise ValueError('audio asset samples must be normalized to [-1, 1]')
        a.setflags(write=False)
        object.__setattr__(self,'samples',a)


def load_wav(path: str | Path, asset_id: str | None = None) -> AudioAsset:
    path = Path(path).expanduser()
    with wave.open(str(path),'rb') as f:
        rate=f.getframerate(); channels=f.getnchannels(); width=f.getsampwidth()
        if f.getcomptype() != 'NONE' or channels != 1 or width not in (1,2,3,4):
            raise ValueError('use uncompressed mono PCM WAV (8/16/24/32-bit)')
        raw=f.readframes(f.getnframes())
    if width == 1:
        a=(np.frombuffer(raw,dtype=np.uint8).astype(np.float64)-128)/128
    elif width == 2:
        a=np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
    elif width == 4:
        a=np.frombuffer(raw,dtype='<i4').astype(np.float64)/2147483648
    else:
        b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(np.int32)
        n=(b[:,0] | b[:,1]<<8 | b[:,2]<<16)
        a=((n ^ (1<<23))-(1<<23)).astype(np.float64)/8388608
    return AudioAsset(asset_id or path.stem,a,rate)
