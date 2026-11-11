import json
import wave

import numpy as np
import pytest

from onda_hearing.acoustic_scene.assets import AudioAsset
from onda_hearing.acoustic_scene.physical import AcousticSurface, TransmissionMaterial
from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry
from onda_hearing.acoustic_scene.types import Binding, Emitter, Pose3, Schedule
from onda_hearing.acoustic_scene.unified_physics import ReflectionMaterial
from onda_hearing.simulators.isaac_audio_snapshot import render_static_snapshot


def test_static_v020_render_four_channel_wav(tmp_path):
    freqs=(125.,500.,1000.,4000.)
    tl=TransmissionMaterial(freqs,(5.,10.,15.,20.),'synthetic_test')
    reflection=ReflectionMaterial(freqs,(0.1,0.15,0.2,0.3),
                                  (0.05,0.08,0.1,0.15),'synthetic_test')
    floor=AcousticSurface.quad('floor', [(-3,-3,0),(3,-3,0),(3,3,0),(-3,3,0)],tl)
    geom=FastAcousticGeometry((floor,))
    n=8000
    t=np.arange(n)/16000
    audio=AudioAsset('beep',0.2*np.sin(2*np.pi*440*t).astype(np.float32),16000)
    emitter=Emitter('beep','beep',Binding(object_path='/Root/source'),
                    Schedule(onset_s=0.,loop=True),gain_db=0.)
    outfile=tmp_path/'render.wav'
    result=render_static_snapshot(geometry=geom,reflectors={'floor':reflection},
                                  emitters=[emitter],assets={'beep':audio},
                                  source_poses={'/Root/source':Pose3((1.,0.,1.))},
                                  microphone_center_m=(0.,0.,1.),output_file=outfile,
                                  duration_s=0.2,specular_order=1,reflector_limit=1)
    assert result['result']=='PASS'
    assert result['channels']==4
    assert result['rms']>0
    assert result['overrange_samples']==0
    assert result['reflectors_used']==1
    with wave.open(str(outfile),'rb') as w:
        assert w.getnchannels()==4
        assert w.getframerate()==16000
        assert w.getnframes()==3200


def test_reject_unsupported_snapshot_inputs(tmp_path):
    with pytest.raises(ValueError, match='duration'):
        render_static_snapshot(geometry=None,reflectors={},emitters=[],assets={},
                               source_poses={},microphone_center_m=(0,0,0),
                               output_file=tmp_path/'a.wav',duration_s=10)
