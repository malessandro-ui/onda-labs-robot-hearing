"""Standalone v0.4 physically motivated two-source example; no simulator dependency."""
from pathlib import Path
import numpy as np
from onda_hearing.physics import AcousticConfig,PoseTrajectory,PhysicalSource,DirectPathRenderer
from onda_hearing.acoustics import save_wav

fs=24000
cfg=AcousticConfig(sample_rate_hz=fs)
array=PoseTrajectory.stationary([0,0,0])
mic_offsets=[[-.08,-.08,0],[.08,-.08,0],[.08,.08,0],[-.08,.08,0]]
t=np.arange(fs)/fs
ring=.08*np.sin(2*np.pi*750*t)*(np.sin(2*np.pi*2*t)>0)
knock=np.zeros(fs);knock[0]=.08
sources=[
    PhysicalSource('telephone',ring,PoseTrajectory.stationary([3,1.5,0]),start_s=0),
    PhysicalSource('knock',knock,PoseTrajectory.stationary([2,-2,0]),start_s=.3),
]
y=DirectPathRenderer(cfg).render(mic_offsets,array,sources,1.,sensor_noise_std=1e-5,seed=17)
path=Path('physics_validation_mics.wav')
save_wav(path,y,fs)
print(f'Created {path} shape={y.shape} peak={np.max(abs(y)):.4f} channels={len(mic_offsets)}')
