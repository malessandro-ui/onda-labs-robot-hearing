"""Reference analytic point-source microphone renderer for SIMULATION ONLY.

This is not a microphone driver or a general acoustic simulator. It uses scene
geometry solely to create multi-channel PCM; the streaming estimator never sees
source positions. Moving sources are approximated constant within each chunk.
"""
import numpy as np
from .core import rotate_xyzw

DEFAULT_ARRAY_METERS=np.array([[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],dtype=float)


def render_point_source(mono, start_sample, frames, sample_rate, array_position_xyz,
                        array_quat_xyzw, source_position_xyz, mic_positions=DEFAULT_ARRAY_METERS,
                        sound_speed=343.0, gain=0.2):
    mono=np.asarray(mono,dtype=float)
    mics=np.asarray(mic_positions,dtype=float)
    center=np.asarray(array_position_xyz,dtype=float)
    source=np.asarray(source_position_xyz,dtype=float)
    if mono.ndim!=1 or frames<1 or start_sample<0 or mics.ndim!=2 or mics.shape[1]!=3:
        raise ValueError('invalid renderer input')
    if not np.isfinite(mono).all() or not np.isfinite(mics).all():
        raise ValueError('nonfinite renderer input')
    tick=np.arange(start_sample,start_sample+frames,dtype=float)
    outputs=[]
    for local in mics:
        world=center+rotate_xyzw(array_quat_xyzw,local)
        distance=float(np.linalg.norm(source-world))
        delay_samples=distance*sample_rate/sound_speed
        # Source timestamp = receive timestamp - acoustic time of flight.
        values=np.interp(tick-delay_samples,np.arange(len(mono)),mono,left=0.,right=0.)
        outputs.append(values * gain / max(1.,distance))
    return np.ascontiguousarray(np.stack(outputs,axis=1),dtype=np.float32)
