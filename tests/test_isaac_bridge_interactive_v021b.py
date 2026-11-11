from threading import Thread
import time
from pathlib import Path
import wave
import numpy as np
from fastapi.testclient import TestClient
from onda_hearing.simulators.isaac_gui_bridge import IsaacBridgeState,create_app
from onda_hearing.simulators.isaac_scene_inspector_v021b import (IndexedObject,page_catalog)
from onda_hearing.simulators.isaac_wav_import_v021b import import_wav_to_pcm16
import pytest


def responder(state):
    deadline=time.monotonic()+6.
    while time.monotonic()<deadline:
        if state.drain(lambda name,payload:{'command':name,'payload':payload},max_count=2):return
        time.sleep(.003)

@pytest.mark.parametrize('method,path,payload,command',[
    ('POST','/api/isaac/selection',{'prim_path':'/Root/forklift'},'select_prim'),
    ('PUT','/api/isaac/microphones',{'preset':'hexagon6'},'set_microphones'),
    ('POST','/api/isaac/microphones/marker',{},'ensure_microphone_marker'),
    ('PATCH','/api/isaac/emitters/one',{'gain_db':3.},'edit_emitter'),
    ('POST','/api/isaac/open-warehouse',{},'open_warehouse'),
])
def test_new_mutation_routes(method,path,payload,command):
    state=IsaacBridgeState();state.set_status(connected_to_isaac=True)
    client=TestClient(create_app(state))
    assert client.request(method,path,json=payload).status_code==403
    worker=Thread(target=responder,args=(state,));worker.start()
    r=client.request(method,path,json=payload,headers={'X-Onda-Client':'OndaGUI'})
    worker.join(6)
    assert r.status_code in (200,202)
    assert r.json()['command']==command


def test_mic_snapshots_and_paged_objects():
    state=IsaacBridgeState()
    state.set_microphones({'count':6})
    assert TestClient(create_app(state)).get('/api/isaac/microphones').json()['data']['count']==6
    catalog=[IndexedObject('/Root/B','Xform','other','B'),
             IndexedObject('/Root/forklift','Mesh','forklift','forklift'),
             IndexedObject('/Root/floor','Mesh','floor','floor')]
    assert page_catalog(catalog,q='forklift')['total']==1
    assert len(page_catalog(catalog,limit=2)['items'])==2
    with pytest.raises(ValueError):page_catalog(catalog,limit=10000)


def _wav_bytes(path,sr=48000,channels=2):
    samples=(0.2*np.sin(np.linspace(0,6*np.pi,4800))).astype(np.float32)
    data=np.repeat(samples[:,None],channels,axis=1)
    import soundfile as sf
    sf.write(path,data,sr,subtype='PCM_16')


def test_audio_upload_converts_48khz_stereo_to_validated_pcm16(tmp_path):
    root=tmp_path/'audio';root.mkdir()
    source=tmp_path/'stereo.wav';_wav_bytes(source)
    client=TestClient(create_app(IsaacBridgeState(),audio_root=root))
    r=client.post('/api/isaac/audio-assets',data=source.read_bytes(),
                 headers={'X-Onda-Client':'OndaGUI','Content-Type':'application/octet-stream'})
    assert r.status_code==201,r.text
    assert r.json()['original_sample_rate']==48000
    filename=r.json()['asset']
    assert '/' not in filename
    with wave.open(str(root/filename),'rb') as w:
        assert w.getframerate()==16000 and w.getnchannels()==1
    assert any(x['asset']==filename for x in client.get('/api/isaac/audio-assets').json()['data'])
    bad=client.post('/api/isaac/audio-assets',data=b'not WAV'*20,
                headers={'X-Onda-Client':'OndaGUI','Content-Type':'application/octet-stream'})
    assert bad.status_code==400
    assert client.post('/api/isaac/audio-assets',data=source.read_bytes()).status_code==403


def test_wav_converter_mono_roundtrip(tmp_path):
    source=tmp_path/'mono.wav';_wav_bytes(source,sr=48000,channels=1)
    target=tmp_path/'out.wav'
    report=import_wav_to_pcm16(source,target)
    assert target.is_file() and report['output_channels']==1
    assert report['output_frames']==1600


def test_emitter_registry_update_preserves_id_and_rejects_invalid(tmp_path):
    from onda_hearing.simulators.isaac_emitter_registry import IsaacEmitterRegistry,RegistryError
    class FakePrim:
        def IsValid(self):return True
        def IsActive(self):return True
        def __bool__(self):return True
    class FakeStage:
        def GetPrimAtPath(self,p):return FakePrim()
    root=tmp_path/'audio';root.mkdir()
    from pathlib import Path
    with wave.open(str(root/'a.wav'),'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000)
        out.writeframes(b'\x00\x00'*1600)
    reg=IsaacEmitterRegistry(stage=FakeStage(),audio_root=root,
         manifest=tmp_path/'emitters.json',scene_id='scene',sample_rate=16000)
    before=reg.add({'emitter_id':'alarm','prim_path':'/Root/forklift','audio_asset':'a.wav'})
    changed=reg.update('alarm',{'gain_db':-8.,'loop':False,'enabled':False})
    assert changed.emitter_id=='alarm' and changed.gain_db==-8
    assert changed.loop is False and changed.enabled is False
    assert len(reg.list_records())==1
    with pytest.raises(RegistryError):reg.update('alarm',{'gain_db':float('inf')})
    with pytest.raises(RegistryError):reg.update('alarm',{'emitter_id':'overwrite'})
    assert reg.list_records()[0]['gain_db']==-8
