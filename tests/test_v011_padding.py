"""ODAS offline hop-alignment contract and isolated-run regression tests."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.spatial.odas_docker import (
    DockerODASRunner, DockerODASResult, pad_interleaved_pcm_s16le,
)
from onda_hearing.spatial.odas_adapter import ODASProtocolError


def packet_with_samples(n: int):
    a = MicrophoneArray.from_positions(
        [[0, 0, 0], [.03, 0, 0], [0, .03, 0], [0, 0, .03]]
    )
    pcm = np.zeros((n, 4), dtype=np.float32)
    pcm[0, 0] = .5
    return a, AudioPacket.from_pcm(pcm, 16000, start_ns=0,
                                   clock_id='sim', array=a)


@pytest.mark.parametrize('samples,pad', [(1,127),(127,1),(128,0),(256,0),
                                        (3200,0),(3210,118)])
def test_padding_size_and_original_content(samples,pad):
    src = bytes((i % 251 for i in range(samples * 4 * 2)))
    data, added = pad_interleaved_pcm_s16le(src, samples=samples,
                                             channels=4, hop_size=128)
    assert added == pad
    assert data[:len(src)] == src
    assert data[len(src):] == bytes(pad * 4 * 2)
    assert len(data) == (samples + pad) * 4 * 2
    assert (samples + added) % 128 == 0


@pytest.mark.parametrize('raw,samples,channels,hop', [
    (b'', 0, 4, 128), (b'0', 1, 4, 128), (b'0' * 8, 1, 0, 128),
    (b'0' * 8, 1, 4, 0),
])
def test_invalid_padding_inputs(raw,samples,channels,hop):
    with pytest.raises(ValueError):
        pad_interleaved_pcm_s16le(raw, samples=samples,
                                   channels=channels, hop_size=hop)


def test_runner_pads_and_filters_padding_only_frames(monkeypatch):
    array, packet = packet_with_samples(3210)
    monkeypatch.setattr('onda_hearing.spatial.odas_docker.shutil.which',
                        lambda _name: '/usr/bin/docker')
    def pretend_docker_run(cmd, **kwargs):
        mount = next(x for x in cmd if x.startswith('type=bind,source='))
        work = Path(mount.split('source=',1)[1].split(',target=',1)[0])
        raw = (work / 'audio.raw').read_bytes()
        assert len(raw) == 3328 * 4 * 2
        assert raw[-118 * 4 * 2:] == bytes(118 * 4 * 2)
        (work / 'potentials.json').write_text(
            '{"timeStamp":24,"src":[{"x":1,"y":0,"z":0,"E":0.8}]}\n'
            '{"timeStamp":26,"src":[]}\n', encoding='utf-8'
        )
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
    monkeypatch.setattr('onda_hearing.spatial.odas_docker.subprocess.run',
                        pretend_docker_run)
    result = DockerODASRunner().run(packet, array)
    assert isinstance(result, DockerODASResult)
    assert (result.original_samples, result.padded_samples,
            result.padding_samples) == (3210, 3328, 118)
    assert len(result.frames) == 1
    assert result.frames[0].sample_offset == 3072


def test_runner_rejects_padding_only_output(monkeypatch):
    array, packet = packet_with_samples(3210)
    monkeypatch.setattr('onda_hearing.spatial.odas_docker.shutil.which',
                        lambda _name: '/usr/bin/docker')
    def pretend_docker_run(cmd, **kwargs):
        mount = next(x for x in cmd if x.startswith('type=bind,source='))
        work = Path(mount.split('source=',1)[1].split(',target=',1)[0])
        (work / 'potentials.json').write_text(
            '{"timeStamp":26,"src":[]}\n', encoding='utf-8'
        )
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
    monkeypatch.setattr('onda_hearing.spatial.odas_docker.subprocess.run',
                        pretend_docker_run)
    with pytest.raises(ODASProtocolError, match='original PCM duration'):
        DockerODASRunner().run(packet, array)
