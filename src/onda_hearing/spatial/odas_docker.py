"""Isolated ODAS offline Docker runner. Docker is an OPTIONAL executable dependency.

Strict boundaries: no shell, no network, read-only container root, fresh per-run
workspace, no access to running ODAS services. This is a benchmarking adapter,
NOT a production real-time driver or license for exact physical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from onda_hearing.contracts import AudioPacket, MicrophoneArray
from .odas_adapter import encode_pcm_s16le, ODASProtocolError, ODASUnavailableError


@dataclass(frozen=True)
class ODASFrame:
    frame_index: int
    sample_offset: int
    time_s: float
    directions_deg: tuple[float, ...]
    energies: tuple[float, ...]


def parse_potential_frames(text: str, *, sample_rate_hz: int, hop_size: int,
                           max_bytes: int = 16_000_000, max_frames: int = 100_000,
                           max_potentials: int = 32) -> tuple[ODASFrame, ...]:
    """Parse every concatenated JSON SSL message; never use SST track IDs.

    ODAS timeStamp is an integer hop counter (verified per configured version).
    If absent, monotonically count frames. Caller should calibrate actual ODAS
    acquisition/processing latency before comparing at millisecond precision.
    """
    if not isinstance(text, str) or len(text.encode('utf-8')) > max_bytes:
        raise ODASProtocolError('ODAS JSON missing or exceeds size limit')
    if type(sample_rate_hz) is not int or sample_rate_hz <= 0 or type(hop_size) is not int or hop_size <= 0:
        raise ValueError('invalid sample rate/hop')
    decoder = json.JSONDecoder(); offset = 0; out = []; prev_stamp = -1
    while offset < len(text):
        while offset < len(text) and text[offset].isspace(): offset += 1
        if offset == len(text): break
        try: record, end = decoder.raw_decode(text, offset)
        except ValueError as exc: raise ODASProtocolError(f'invalid ODAS potential JSON at character {offset}') from exc
        offset = end
        if not isinstance(record, dict) or not isinstance(record.get('src'), list):
            raise ODASProtocolError('ODAS SSL record requires a src list')
        if len(record['src']) > max_potentials:
            raise ODASProtocolError('too many ODAS potentials')
        stamp = record.get('timeStamp', len(out))
        if type(stamp) is not int or stamp < 0 or stamp <= prev_stamp:
            raise ODASProtocolError('ODAS timeStamp must increase monotonically')
        prev_stamp = stamp
        angles = []; energies = []
        for obj in record['src']:
            if not isinstance(obj,dict) or 'id' in obj:
                raise ODASProtocolError('SSL potentials only, not tracked source records')
            try: x,y,z,e = (float(obj[k]) for k in ('x','y','z','E'))
            except (TypeError, ValueError, KeyError) as exc:
                raise ODASProtocolError('malformed potential fields') from exc
            if not all(map(math.isfinite, (x,y,z,e))) or e < 0:
                raise ODASProtocolError('nonfinite or negative potential')
            norm = math.sqrt(x*x+y*y+z*z)
            if norm < 1e-9: continue
            if not .5 <= norm <= 1.5:
                raise ODASProtocolError('direction vector must have near-unit norm')
            angles.append(math.degrees(math.atan2(y,x))); energies.append(e)
        sample_offset = stamp*hop_size
        out.append(ODASFrame(len(out),sample_offset,sample_offset/sample_rate_hz,tuple(angles),tuple(energies)))
        if len(out) > max_frames: raise ODASProtocolError('too many output frames')
    if not out: raise ODASProtocolError('ODAS returned zero SSL frames')
    return tuple(out)


def _mic_entries(array: MicrophoneArray) -> str:
    result = []
    for mic in array.microphones:
        x,y,z = mic.position_m
        result.append('  { mu = (%.9f, %.9f, %.9f); sigma2 = (1E-6,0,0,0,1E-6,0,0,0,1E-6); direction = (0,0,1); angle = (0.0,180.0); }' % (x,y,z))
    return ',\n'.join(result)


def build_offline_cfg(array: MicrophoneArray, *, sample_rate_hz: int = 16000,
                      hop_size: int = 128, max_sources: int = 4) -> str:
    """Create offline libconfig configuration, matching all microphone positions.

    Keeps ODAS SST/SSS outputs on blackhole to avoid implicit track/separation
    comparisons. The general spatial filter is unrestricted.
    """
    if array.channels < 2 or array.channels > 32 or array.geometry_rank < 1:
        raise ValueError('ODAS requires 2..32 spatially distinct mics')
    if sample_rate_hz not in (16000, 24000, 44100, 48000):
        raise ValueError('unsupported benchmark sample rate')
    if hop_size not in (128,256,512) or type(max_sources) is not int or not 1 <= max_sources <= 4:
        raise ValueError('unsupported hop size or source count')
    mapping = ','.join(map(str, range(1,array.channels+1)))
    return f'''# Onda isolated ODAS benchmark; generated, never edit by hand.
version = "2.1";
raw: {{ fS = {sample_rate_hz}; hopSize = {hop_size}; nBits = 16; nChannels = {array.channels};
  interface: {{ type = "file"; path = "/work/audio.raw"; }}; }};
mapping: {{ map: ({mapping}); }};
general: {{
 epsilon = 1E-20;
 size: {{ hopSize = {hop_size}; frameSize = {hop_size*2}; }};
 samplerate: {{ mu = {sample_rate_hz}; sigma2 = 0.01; }};
 speedofsound: {{ mu = 343.0; sigma2 = 25.0; }};
 mics = (\n{_mic_entries(array)}\n );
 spatialfilters = ({{ direction = (0.0,0.0,1.0); angle = (0.0,180.0); }});
 nThetas = 181; gainMin = 0.25;
}};
sne: {{ b = 3; alphaS = 0.1; L = 150; delta = 3.0; alphaD = 0.1; }};
ssl: {{ nPots = {max_sources}; nMatches = 10; probMin = 0.5; nRefinedLevels = 1; interpRate = 4;
 scans = ({{ level = 2; delta = -1; }}, {{ level = 4; delta = -1; }});
 potential: {{ format = "json"; interface: {{ type = "file"; path = "/work/potentials.json"; }}; }};
}};
sst: {{ mode = "kalman"; add = "dynamic";
 active = ({{ weight=1.0; mu=0.18; sigma2=0.0025; }});
 inactive = ({{ weight=1.0; mu=0.10; sigma2=0.0025; }});
 sigmaR2_prob=0.0025; sigmaR2_active=0.0225; sigmaR2_target=0.0025;
 Pfalse=0.1; Pnew=0.1; Ptrack=0.8; theta_new=0.9; N_prob=5; theta_prob=0.8;
 N_inactive=({','.join(['250']*max_sources)}); theta_inactive=0.9;
 kalman: {{ sigmaQ=0.001; }};
 particle: {{ nParticles=1000; st_alpha=2.0; st_beta=0.04; st_ratio=0.5; ve_alpha=0.05;
 ve_beta=0.2; ve_ratio=0.3; ac_alpha=0.5; ac_beta=0.2; ac_ratio=0.2; Nmin=0.7; }};
 target=(); tracked: {{ format="undefined"; interface: {{ type="blackhole"; }}; }};
}};
sss: {{ mode_sep="dds"; mode_pf="ms"; gain_sep=1.0; gain_pf=10.0;
 dds: {{ }}; dgss: {{ mu=0.01; lambda=0.5; }}; dmvdr: {{ }};
 ms: {{ alphaPmin=0.07; eta=0.5; alphaZ=0.8; thetaWin=0.3; alphaWin=0.3;
 maxAbsenceProb=0.9; Gmin=0.01; winSizeLocal=3; winSizeGlobal=23; winSizeFrame=256; }};
 ss: {{ Gmin=0.01; Gmid=0.9; Gslope=10.0; }};
 separated: {{ fS={sample_rate_hz}; hopSize={hop_size}; nBits=16;
 interface: {{ type="blackhole"; }}; }};
 postfiltered: {{ fS={sample_rate_hz}; hopSize={hop_size}; nBits=16; gain=10.0;
 interface: {{ type="blackhole"; }}; }};
}};
classify: {{ frameSize=4096; winSize=3; tauMin=88; tauMax=551; deltaTauMax=20;
 alpha=0.3; gamma=0.05; phiMin=0.5; r0=0.2;
 category: {{ format="undefined"; interface: {{ type="blackhole"; }}; }};
}};
'''


@dataclass(frozen=True)
class DockerODASOptions:
    image: str = 'onda-odas:dev'
    executable: str = '/opt/odas/build-linux/bin/odaslive'
    docker_binary: str = 'docker'
    hop_size: int = 128
    timeout_s: float = 35.
    max_sources: int = 4
    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,200}',self.image):
            raise ValueError('invalid image name')
        if not self.executable.startswith('/') or '\x00' in self.executable:
            raise ValueError('absolute executable path required')
        if not math.isfinite(self.timeout_s) or not 1 <= self.timeout_s <= 600:
            raise ValueError('timeout_s must be between 1 and 600 seconds')
        if self.hop_size not in (128,256,512) or not 1 <= self.max_sources <= 4:
            raise ValueError('invalid hop size or max sources')


@dataclass(frozen=True)
class DockerODASResult:
    frames: tuple[ODASFrame,...]
    image: str
    configuration_sha256: str
    audio_sha256: str
    command_returncode: int
    original_samples: int = 0
    padded_samples: int = 0
    padding_samples: int = 0


def pad_interleaved_pcm_s16le(raw: bytes, *, samples: int, channels: int,
                              hop_size: int) -> tuple[bytes, int]:
    """Right-pad whole interleaved PCM frames to an ODAS hop boundary.

    Never truncate input or alter pre-existing audio samples. Returned padding
    is measured in *per-channel samples*, rather than bytes or sample values.
    """
    if any(type(x) is not int or x <= 0 for x in (samples, channels, hop_size)):
        raise ValueError('samples, channels, and hop_size must be positive integers')
    if not isinstance(raw, bytes) or len(raw) != samples * channels * 2:
        raise ValueError('PCM must contain exactly samples * channels * 2 bytes')
    padding = (-samples) % hop_size
    return raw + bytes(padding * channels * 2), padding


class DockerODASRunner:
    def __init__(self, options: DockerODASOptions | None = None):
        self.options = options or DockerODASOptions()

    def run(self, packet: AudioPacket, array: MicrophoneArray) -> DockerODASResult:
        packet.validate_array(array)
        opt = self.options
        if shutil.which(opt.docker_binary) is None:
            raise ODASUnavailableError('Docker CLI is not installed; optional backend unavailable')
        cfg = build_offline_cfg(array,sample_rate_hz=packet.sample_rate_hz,hop_size=opt.hop_size,max_sources=opt.max_sources)
        original_samples = len(packet.pcm)
        raw, padding_samples = pad_interleaved_pcm_s16le(
            encode_pcm_s16le(packet), samples=original_samples,
            channels=packet.channel_count, hop_size=opt.hop_size,
        )
        padded_samples = original_samples + padding_samples
        with tempfile.TemporaryDirectory(prefix='onda-odas-v011-') as directory:
            work = Path(directory)
            (work/'audio.raw').write_bytes(raw)
            (work/'odas.cfg').write_text(cfg,encoding='utf-8')
            command=[opt.docker_binary,'run','--rm','--network','none','--read-only',
                     '--cap-drop','ALL','--security-opt','no-new-privileges',
                     '--pids-limit','128','--memory','768m','--cpus','2',
                     '--mount',f'type=bind,source={work},target=/work',
                     '--entrypoint',opt.executable,opt.image,'-c','/work/odas.cfg']
            try:
                process=subprocess.run(command,capture_output=True,timeout=opt.timeout_s,check=False)
            except subprocess.TimeoutExpired as exc:
                raise ODASUnavailableError('isolated ODAS process timed out') from exc
            except OSError as exc:
                raise ODASUnavailableError('Docker invocation failed') from exc
            if process.returncode != 0:
                tail=(process.stderr+process.stdout).decode('utf-8','replace')[-1800:]
                raise ODASUnavailableError(f'ODAS container returned {process.returncode}: {tail}')
            out=work/'potentials.json'
            if not out.is_file():
                raise ODASProtocolError('ODAS did not create potentials.json; inspect compatibility of generated config')
            if out.stat().st_size > 16_000_000:
                raise ODASProtocolError('ODAS output exceeds 16MB limit')
            frames_all = parse_potential_frames(
                out.read_text(encoding='utf-8'),
                sample_rate_hz=packet.sample_rate_hz, hop_size=opt.hop_size,
            )
            # ODAS timeStamp is interpreted as a hop index, not a calibrated
            # capture timestamp. Exclude frames whose nominal timestamp falls
            # wholly outside the original (unpadded) recording. Do not shift
            # any retained timestamp; acquisition latency is still unknown.
            frames = tuple(f for f in frames_all if f.sample_offset < original_samples)
            if not frames:
                raise ODASProtocolError(
                    'ODAS produced no localization frames within original PCM duration'
                )
            return DockerODASResult(
                frames, opt.image,
                hashlib.sha256(cfg.encode()).hexdigest(),
                hashlib.sha256(raw).hexdigest(), process.returncode,
                original_samples, padded_samples, padding_samples,
            )
