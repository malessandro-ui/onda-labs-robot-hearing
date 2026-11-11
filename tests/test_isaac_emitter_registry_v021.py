import json
from pathlib import Path
import wave

import pytest

from onda_hearing.simulators.isaac_emitter_registry import (
    IsaacEmitterRegistry, RegistryError, RegistryConflict, validate_prim_path,
)


class FakePrim:
    def __init__(self, valid=True): self.valid=valid
    def IsValid(self): return self.valid
    def IsActive(self): return self.valid
    def __bool__(self): return self.valid


class FakeStage:
    def GetPrimAtPath(self, path): return FakePrim(path in ('/Root/forklift', '/Root/forklift/S_ForkliftBody'))


def make_pcm(path, sample_rate=16000):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b'\0\0' * 1600)


@pytest.fixture
def reg(tmp_path):
    audio=tmp_path/'audio'
    audio.mkdir()
    make_pcm(audio/'alarm.wav')
    return IsaacEmitterRegistry(stage=FakeStage(), audio_root=audio,
                                manifest=tmp_path/'runs'/'emitters.json',
                                scene_id='stage123', sample_rate=16000)


def test_create_persist_load_and_convert(reg):
    obj=reg.add({'emitter_id':'alarm','prim_path':'/Root/forklift',
                 'audio_asset':'alarm.wav','loop':True,'gain_db':-3.})
    assert obj.prim_path == '/Root/forklift'
    assert reg.manifest.is_file()
    em, assets=reg.for_renderer()
    assert em[0].binding.object_path=='/Root/forklift'
    assert assets['alarm'].sample_rate==16000
    reg2=IsaacEmitterRegistry(stage=FakeStage(), audio_root=reg.audio_root,
                              manifest=reg.manifest, scene_id='stage123')
    assert reg2.load()['loaded']==1
    assert reg2.list_records()[0]['audio_asset']=='alarm.wav'
    assert json.loads(reg.manifest.read_text())['emitters'][0]['audio_asset'] == 'alarm.wav'
    assert reg2.remove('alarm')['deleted']=='alarm'
    assert reg2.list_records()==[]


def test_reject_wrong_scene_manifest(reg):
    reg.add({'prim_path':'/Root/forklift','audio_asset':'alarm.wav'})
    different=IsaacEmitterRegistry(stage=FakeStage(), audio_root=reg.audio_root,
                                   manifest=reg.manifest, scene_id='different')
    assert different.load()['reason']=='scene_id_mismatch'
    assert not different.list_records()


def test_reject_unexpected_fields(reg):
    with pytest.raises(RegistryError, match='unknown emitter fields'):
        reg.add({'prim_path':'/Root/forklift','audio_asset':'alarm.wav','dangerous_command':'foo'})
    assert not reg.list_records()


def test_duplicate_id_rejected(reg):
    p={'emitter_id':'foo','prim_path':'/Root/forklift','audio_asset':'alarm.wav'}
    reg.add(p)
    with pytest.raises(RegistryConflict): reg.add(p)


def test_missing_and_invalid_prim(reg):
    for bad in ('/Root/missing','/Root/../secret','', 'Root/forklift'):
        with pytest.raises(RegistryError):
            reg.add({'prim_path':bad,'audio_asset':'alarm.wav'})
    with pytest.raises(RegistryError): validate_prim_path('/Root/forklift.someAttr')


def test_audio_path_escape_and_wrong_rate(reg,tmp_path):
    outside=tmp_path/'secrets.wav';make_pcm(outside)
    with pytest.raises(RegistryError):
        reg.add({'prim_path':'/Root/forklift','audio_asset':'../secrets.wav'})
    make_pcm(reg.audio_root/'wrong.wav',sample_rate=48000)
    with pytest.raises(RegistryError, match='sample rate'):
        reg.add({'prim_path':'/Root/forklift','audio_asset':'wrong.wav'})


def test_invalid_gain_and_boolean(reg):
    for patch in ({'gain_db':1e99},{'loop':'yes'},{'local_offset_m':[float('nan'),0,0]}):
        with pytest.raises((RegistryError, ValueError)):
            reg.add({'prim_path':'/Root/forklift','audio_asset':'alarm.wav',**patch})
