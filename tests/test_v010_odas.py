from pathlib import Path
import hashlib
import numpy as np
import pytest
from onda_hearing.contracts import AudioPacket,MicrophoneArray,UnsupportedConfigurationError
from onda_hearing.spatial.odas_adapter import (ODASBinding,ODASOfflineBackend,ODASUnavailableError,ODASProtocolError,
    encode_pcm_s16le,parse_potential_json,geometry_digest,validate_binding)
from onda_hearing.spatial.music import MUSICConfig,CovarianceMUSICLocalizer

def sample():
    array=MicrophoneArray.from_positions([[-.04,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0]])
    return array,AudioPacket.from_pcm(np.zeros((1024,4),np.float32),16000,start_ns=0,clock_id='sim',array=array)

def binding(tmp_path,a):
    cfg=tmp_path/'odas.cfg';cfg.write_text('trusted external ODAS config')
    return ODASBinding(cfg,hashlib.sha256(cfg.read_bytes()).hexdigest(),geometry_digest(a),16000,4,
                       tmp_path/'audio.raw',tmp_path/'potential.json')

def test_digest_channel_order_and_position():
    a,_=sample();b=MicrophoneArray.from_positions([[-.041,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0]])
    assert geometry_digest(a)!=geometry_digest(b)

def test_s16le_exact():
    a,p=sample();payload=encode_pcm_s16le(p)
    assert len(payload)==1024*4*2
    assert np.max(np.frombuffer(payload,dtype='<i2'))==0

def test_parser_multi_frame_last_wins():
    s='{"timeStamp":1,"src":[{"x":1,"y":0,"z":0,"E":0.8}]}\n{"timeStamp":2,"src":[{"x":0,"y":1,"z":0,"E":0.5}]}'
    out=parse_potential_json(s,frame_id='body')
    assert len(out)==1 and abs(out[0].azimuth_rad-np.pi/2)<1e-8

def test_parser_empty_last_frame():
    assert parse_potential_json('{"src":[{"x":1,"y":0,"z":0,"E":1}]}\n{"src":[]}',frame_id='body')==()

@pytest.mark.parametrize('value',["",'{"src":[{"x":0,"y":0,"z":1}]}','{"src":[{"id":1,"x":1,"y":0,"z":0,"E":1}]}',
  '{"src":[{"x":2,"y":0,"z":0,"E":1}]}', '{"src":{}}'])
def test_invalid_potential_fails(value):
    with pytest.raises(ODASProtocolError):parse_potential_json(value,frame_id='body')

def test_geometry_binding_rejection(tmp_path):
    a,p=sample();b=binding(tmp_path,a)
    validate_binding(b,p,a)
    wrong=MicrophoneArray.from_positions([[-.041,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0]])
    with pytest.raises(UnsupportedConfigurationError):validate_binding(b,p,wrong)

def test_mutated_config_rejection(tmp_path):
    a,p=sample();b=binding(tmp_path,a)
    b.config_path.write_text('changed')
    with pytest.raises(UnsupportedConfigurationError):validate_binding(b,p,a)

def test_unavailable_is_not_fabricated(tmp_path):
    a,p=sample();b=binding(tmp_path,a)
    backend=ODASOfflineBackend(b,executable='odas-core-this-does-not-exist')
    with pytest.raises(ODASUnavailableError):backend.process(p,a)
    assert not b.raw_input_path.exists()

def test_music_silence():
    a,p=sample();result=CovarianceMUSICLocalizer().estimate(p,a)
    assert result.directions==()

def test_music_rejects_unsupported_geometry():
    a=MicrophoneArray.from_positions([[0,0,0],[.03,0,0]])
    p=AudioPacket.from_pcm(np.zeros((2048,2),np.float32),16000,start_ns=0,clock_id='sim',array=a)
    with pytest.raises(ValueError):CovarianceMUSICLocalizer().estimate(p,a)

def test_music_config_validation():
    with pytest.raises(ValueError):MUSICConfig(fft_samples=32)
