import json
import numpy as np
import pytest
from onda_hearing.contracts import AudioPacket,MicrophoneArray
from onda_hearing.spatial.odas_docker import (parse_potential_frames,build_offline_cfg,DockerODASRunner,DockerODASOptions)
from onda_hearing.spatial.odas_adapter import ODASProtocolError,ODASUnavailableError


def fixture():
    array=MicrophoneArray.from_positions([[0,0,0],[.03,0,0],[0,.03,0],[0,0,.03]])
    return array,AudioPacket.from_pcm(np.zeros((1024,4),dtype=np.float32),16000,start_ns=0,clock_id='sim',array=array)

def test_cfg_matches_geometry_and_file_interface():
    array,_=fixture(); cfg=build_offline_cfg(array)
    assert 'nChannels = 4' in cfg
    assert 'map: (1,2,3,4)' in cfg
    assert '/work/audio.raw' in cfg and '/work/potentials.json' in cfg
    assert '0.030000000' in cfg
    assert 'type = "socket"' not in cfg

def test_every_frame_preserved():
    payload='{"timeStamp":0,"src":[{"x":1,"y":0,"z":0,"E":0.8}]}\n{"timeStamp":2,"src":[]}\n'
    frames=parse_potential_frames(payload,sample_rate_hz=16000,hop_size=128)
    assert len(frames)==2 and frames[0].directions_deg==(0.0,)
    assert frames[1].directions_deg==() and frames[1].sample_offset==256

@pytest.mark.parametrize('bad',["",'{"src":{}}', '{"timeStamp":1,"src":[]}\n{"timeStamp":1,"src":[]}',
    '{"src":[{"id":1,"x":1,"y":0,"z":0,"E":.5}]}',
    '{"src":[{"x":9,"y":0,"z":0,"E":.5}]}'])
def test_bad_frames_fail_closed(bad):
    with pytest.raises(ODASProtocolError):parse_potential_frames(bad,sample_rate_hz=16000,hop_size=128)

def test_invalid_configurations():
    a,p=fixture()
    with pytest.raises(ValueError):build_offline_cfg(a,sample_rate_hz=12345)
    with pytest.raises(ValueError):DockerODASOptions(image='hello; rm -rf /')
    with pytest.raises(ValueError):DockerODASOptions(timeout_s=0)

def test_no_docker_does_not_fabricate(monkeypatch):
    a,p=fixture();monkeypatch.setattr('onda_hearing.spatial.odas_docker.shutil.which',lambda _:None)
    with pytest.raises(ODASUnavailableError):DockerODASRunner().run(p,a)

def test_benchmark_default_never_runs_docker(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
    from odas_docker_v011 import run
    summary=run(tmp_path/'report',limit=1)
    assert summary['methods']['odas_docker']['evaluated']==0
    assert summary['odas_executed'] is False
