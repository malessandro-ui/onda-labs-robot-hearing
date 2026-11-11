import wave
import numpy as np
import pytest
from onda_hearing.acoustic_scene.assets import AudioAsset
from onda_hearing.acoustic_scene.physical import AcousticSurface,TransmissionMaterial
from onda_hearing.acoustic_scene.fast_geometry_v020 import FastAcousticGeometry
from onda_hearing.acoustic_scene.types import Binding,Emitter,Pose3,Schedule
from onda_hearing.acoustic_scene.unified_physics import ReflectionMaterial
from onda_hearing.simulators.isaac_microphone_array_v021b import MicrophoneRig,PRESETS
from onda_hearing.simulators.isaac_audio_snapshot import render_static_snapshot

@pytest.mark.parametrize('layout',['triangle3','hexagon6'])
def test_real_multichannel_reflection_wav(tmp_path,layout):
    freqs=(125.,500.,1000.,4000.)
    tl=TransmissionMaterial(freqs,(3.,7.,11.,20.),'synthetic_test')
    refl=ReflectionMaterial(freqs,(0.1,0.12,0.16,0.2),(.04,.05,.06,.08),'synthetic_test')
    geom=FastAcousticGeometry((AcousticSurface.quad('floor',
            [(-3,-3,0),(3,-3,0),(3,3,0),(-3,3,0)],tl),))
    t=np.arange(4800)/16000
    tone=AudioAsset('tone',(.25*np.sin(2*np.pi*440*t)).astype(np.float32),16000)
    e=Emitter('tone','tone',Binding(object_path='/Root/Beeper'),Schedule(onset_s=0.,loop=True))
    rig=MicrophoneRig(center_m=(0.,0.,1.),local_positions_m=PRESETS[layout])
    mics,positions,pose=rig.capture()
    output=tmp_path/'out.wav'
    result=render_static_snapshot(geometry=geom,reflectors={'floor':refl},
        emitters=[e],assets={'tone':tone},source_poses={'/Root/Beeper':Pose3((1.,0.,1.))},
        microphone_center_m=(0,0,1),microphone_array=mics,output_file=output,
        duration_s=.12,specular_order=1,reflector_limit=1)
    n=len(PRESETS[layout]);assert result['channels']==n
    assert len(result['channel_rms'])==n
    assert len(result['microphone_world_positions_m'])==n
    assert len(result['waveform_envelope'])==n
    assert result['rms']>0 and result['overrange_samples']==0
    assert result['active_reflection_paths_total']>=1
    for path in result['active_reflection_paths']:
        assert path['surface_ids']==['floor']
        assert path['arrival_delay_ms']>0
        assert len(path['bounce_points_m'])==1
    with wave.open(str(output),'rb') as f:
        assert f.getnchannels()==n
        assert f.getnframes()==1920


def test_reject_unattached_mic_array():
    from onda_hearing.acoustic_scene.types import MicArray
    with pytest.raises(ValueError,match='captured world'):
        render_static_snapshot(geometry=None,reflectors={},
             emitters=[Emitter("source","source",Binding(world=Pose3((1.,0.,1.))),Schedule(onset_s=0.))],assets={},
             source_poses={},microphone_center_m=(0.,0.,1.),output_file='x.wav',
             microphone_array=MicArray(Binding(object_path='/Root/rig'),PRESETS['square4']))
