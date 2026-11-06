"""Two overlapping synthetic signals -> real PCM mixture -> DOA candidates + tracks.

This example does NOT guarantee resolving both sources (no waveform separation).
"""
from pathlib import Path
import numpy as np
from onda_hearing import (MicrophoneArray, FreeFieldSimulator, SoundSource,
                          HearingPipeline, save_wav)

SR = 24000
array = MicrophoneArray(np.array([[-.06,-.06,0],[.06,-.06,0],[.06,.06,0],[-.06,.06,0]]))
t = np.arange(int(SR*0.5)) / SR
# Broadband deterministic events at distinct positions; overlapping at receiver.
rng = np.random.default_rng(5)
a = rng.normal(0, .07, size=len(t)) * np.exp(-7*t)
b = rng.normal(0, .055, size=len(t)) * np.exp(-5*t)
sources = [
    SoundSource('phone', a, np.array([2., 1., 0.]), start_s=.1),
    SoundSource('door', b, np.array([1., -2., 0.]), start_s=.25),
]
sim = FreeFieldSimulator(SR)
block = sim.render(array, np.zeros(3), sources, 1.0, noise_std=1e-5, seed=3)
path = Path('overlapping_mics.wav')
save_wav(path, block.pcm, SR)
print('Saved', path, 'shape', block.pcm.shape)
pipe = HearingPipeline(array, max_candidates=2)
window = int(.25*SR)
for i in range(0, len(block.pcm), window):
    from onda_hearing import AudioBlock
    chunk = AudioBlock(block.pcm[i:i+window], SR, i/SR, array.frame_id)
    obs = pipe.process(chunk)
    print(f't={obs.timestamp_s:.2f}s candidates:',
          [round(np.degrees(d.azimuth_rad),1) for d in obs.directions],
          'tracks:', [(tr.track_id,round(np.degrees(tr.azimuth_rad),1),tr.confirmed) for tr in obs.tracks])
