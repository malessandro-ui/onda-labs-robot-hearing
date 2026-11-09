#!/usr/bin/env python3
"""Generate a synthetic *hybrid* phone-like test WAV for spatial diagnostics.

Not a realistic telephone benchmark. Includes seeded broadband content so a
GCC-PHAT localizer has timing information, alongside two dial-tone frequencies.
"""
from __future__ import annotations
import argparse
import wave
import numpy as np
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('/tmp/onda_v017_hybrid_ring.wav'))
    p.add_argument('--duration',type=float,default=4.)
    args=p.parse_args()
    if not 1<=args.duration<=30:p.error('duration must be in [1,30] seconds')
    sr=16000;n=round(sr*args.duration)
    t=np.arange(n,dtype=np.float64)/sr
    rng=np.random.default_rng(1742)
    noise=rng.standard_normal(n)
    # Smooth bounded broadband component with deterministic spectrum.
    noise=(noise+np.roll(noise,1)+np.roll(noise,2))/3
    tone=.13*np.sin(2*np.pi*350*t)+.11*np.sin(2*np.pi*440*t)
    env=((t%1.0)<.7).astype(np.float64)
    pcm=np.clip(env*(tone+.095*noise),-.95,.95)
    encoded=np.rint(pcm*32767).astype('<i2')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(args.output),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        f.writeframes(encoded.tobytes())
    print('Created',args.output,'16 kHz mono',n,'samples')
if __name__=='__main__':main()
