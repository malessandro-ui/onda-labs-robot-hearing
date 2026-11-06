"""Optional offline ODAS SSL adapter.

Uses a separately installed odascore and a user-authored, calibrated .cfg.
No ODAS code, executable, or bundled configuration is redistributed here.
The binding manifest is a strict assertion of the externally maintained cfg.
For correctness, configure raw input as file, potential output as JSON file,
and disable/redirect other sinks as appropriate for the local ODAS build.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import numpy as np
from onda_hearing.contracts import (AudioPacket, MicrophoneArray, SpatialCapabilities,
                                    DirectionEstimate, SpatialResult, UnsupportedConfigurationError)

class ODASUnavailableError(RuntimeError):
    pass

class ODASProtocolError(RuntimeError):
    pass


def geometry_digest(array: MicrophoneArray) -> str:
    """Digest includes channel order, units (metres), identities, frame and positions."""
    payload = {'array_id': array.array_id, 'frame_id': array.frame_id,
               'geometry_id': array.geometry_id,
               'microphones': [{'channel': m.channel, 'name': m.name,
                                'position_m': list(m.position_m)} for m in array.microphones]}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class ODASBinding:
    """Explicit assertion about a trusted, separately configured ODAS instance.

    Config must use raw.interface=file and ssl.potential.interface=file, JSON.
    Input is 16-bit signed little-endian interleaved PCM, ODAS mapping is 1-based.
    Every listed channel must match exactly the SDK microphone channel ordering.
    Output values are sensor-frame hypotheses, not calibrated probabilities.
    """
    config_path: Path
    config_sha256: str
    geometry_sha256: str
    sample_rate_hz: int
    channels: int
    raw_input_path: Path
    potential_output_path: Path
    raw_hop_size: int = 128
    max_candidates: int = 4
    def __post_init__(self):
        for key in ('config_sha256', 'geometry_sha256'):
            digest = getattr(self, key)
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'{key} must be a lowercase SHA256 hex digest')
        if any(type(v) is not int or v<=0 for v in (self.sample_rate_hz,self.channels,self.raw_hop_size,self.max_candidates)):
            raise ValueError('sample rate, channels, raw hop and candidate count must be positive integers')
        if self.max_candidates>64: raise ValueError('maximum candidate count too high')
        for key in ('config_path','raw_input_path','potential_output_path'):
            val=Path(getattr(self,key)).expanduser().resolve()
            object.__setattr__(self,key,val)
        if self.raw_input_path==self.potential_output_path or self.config_path in (self.raw_input_path,self.potential_output_path):
            raise ValueError('config/input/output must be distinct files')


def validate_binding(binding: ODASBinding, audio: AudioPacket, array: MicrophoneArray) -> None:
    audio.validate_array(array)
    if geometry_digest(array)!=binding.geometry_sha256:
        raise UnsupportedConfigurationError('ODAS config geometry binding does not match input array')
    if array.channels!=binding.channels or audio.sample_rate_hz!=binding.sample_rate_hz:
        raise UnsupportedConfigurationError('ODAS config sample rate/channel count does not match input')
    if len(audio.pcm)%binding.raw_hop_size:
        raise UnsupportedConfigurationError('ODAS offline input requires a whole number of raw hops; pad explicitly')
    try:
        content=binding.config_path.read_bytes()
    except OSError as exc:
        raise ODASUnavailableError(f'ODAS configuration not readable: {binding.config_path}') from exc
    if hashlib.sha256(content).hexdigest()!=binding.config_sha256:
        raise UnsupportedConfigurationError('ODAS configuration changed since binding was approved')
    if binding.raw_input_path.parent!=binding.potential_output_path.parent:
        raise UnsupportedConfigurationError('ODAS raw and output files must be in same isolated directory')
    # Configuration binding checks prove only that metadata matches the recorded manifest.
    # The .cfg is still a trusted external artifact and must be independently audited.


def encode_pcm_s16le(audio: AudioPacket) -> bytes:
    """Interleaved signed 16-bit LE PCM, clip and round deterministically."""
    x=np.asarray(audio.pcm, dtype=np.float64)
    return np.rint(np.clip(x,-1.,1.)*32767.).astype('<i2').tobytes(order='C')


def parse_potential_json(data: str, *, frame_id: str, max_candidates: int = 4) -> tuple[DirectionEstimate,...]:
    """Parse concatenated ODAS JSON records; use last nonempty potential frame.

    ODAS uses {'src': [{'x':...,'y':...,'z':...,'E':...}]} for SSL potentials.
    Reject tracked JSON (which carries IDs), nulls, malformed/nonfinite vectors.
    """
    decoder=json.JSONDecoder()
    pos=0; latest=[]; seen=0
    if len(data)>20_000_000:raise ODASProtocolError('ODAS JSON exceeds maximum size')
    while pos<len(data):
        while pos<len(data) and data[pos].isspace():pos+=1
        if pos==len(data):break
        try: record,end=decoder.raw_decode(data,pos)
        except ValueError as exc:raise ODASProtocolError(f'invalid ODAS JSON near byte {pos}') from exc
        pos=end;seen+=1
        if not isinstance(record,dict) or not isinstance(record.get('src'),list):
            raise ODASProtocolError('expected ODAS SSL potential object with src list')
        current=[]
        for src in record['src']:
            if not isinstance(src,dict) or any(k not in src for k in ('x','y','z','E')) or 'id' in src:
                raise ODASProtocolError('invalid ODAS SSL potential entry (or SST output used)')
            try:
                x,y,z,e=(float(src[k]) for k in ('x','y','z','E'))
            except (TypeError,ValueError) as exc:raise ODASProtocolError('invalid ODAS numeric potential') from exc
            if not all(map(math.isfinite,(x,y,z,e))) or e<0:
                raise ODASProtocolError('nonfinite or negative potential energy')
            norm=math.hypot(math.hypot(x,y),z)
            if norm<1e-9:continue  # inactive/null candidate
            if not .5 <= norm <= 1.5:raise ODASProtocolError('ODAS direction not approximately unit length')
            current.append(DirectionEstimate(math.atan2(y,x),frame_id,e,
                                             elevation_rad=math.atan2(z,math.hypot(x,y))))
        # Preserve the most recent frame, including a valid empty frame (no phantom sources).
        latest=sorted(current,key=lambda d:-d.score)[:max_candidates]
    if not seen:raise ODASProtocolError('ODAS produced no potential JSON frames')
    return tuple(latest)


class ODASOfflineBackend:
    """Offline subprocess adapter. Binary/config are user-provisioned, never bundled.

    IMPORTANT: ODAS is not inherently world-frame motion-compensated. Potential
    bearings are in the microphone-array frame at the ODAS output frame time.
    For moving arrays, do not treat the last output frame as an entire-window DOA.
    This adapter explicitly refuses non-stationary inputs through caller policy.
    """
    def __init__(self,binding: ODASBinding, *, executable: str='odascore',timeout_s: float=20.):
        if not isinstance(binding,ODASBinding):raise TypeError('ODASBinding required')
        if not math.isfinite(timeout_s) or not 0<timeout_s<=600:raise ValueError('invalid timeout')
        self.binding=binding;self.executable=executable;self.timeout_s=float(timeout_s)
    @property
    def capabilities(self):
        b=self.binding
        return SpatialCapabilities('odas_offline_ssl',b.channels,b.channels,frozenset({b.sample_rate_hz}),
                                   multi_source=True,max_sources=b.max_candidates,stateful=True)
    def reset(self):pass
    def process(self,audio:AudioPacket,geometry:MicrophoneArray)->SpatialResult:
        b=self.binding
        self.capabilities.validate(audio,geometry)
        validate_binding(b,audio,geometry)
        exe=shutil.which(self.executable)
        if exe is None:raise ODASUnavailableError(f'ODAS binary unavailable: {self.executable}')
        if not b.raw_input_path.parent.is_dir():
            raise ODASUnavailableError('ODAS isolated working directory does not exist')
        if b.raw_input_path.exists() or b.potential_output_path.exists():
            raise ODASProtocolError('ODAS workspace contains stale files; use fresh isolated workspace')
        b.raw_input_path.write_bytes(encode_pcm_s16le(audio))
        try:
            try:
                run=subprocess.run([exe,str(b.config_path)],cwd=str(b.raw_input_path.parent),
                                   capture_output=True,timeout=self.timeout_s,check=False)
            except subprocess.TimeoutExpired as exc:
                raise ODASUnavailableError('ODAS process timed out') from exc
            if run.returncode!=0:
                err=run.stderr.decode('utf8','replace')[-1500:]
                raise ODASUnavailableError(f'ODAS returned {run.returncode}: {err}')
            if not b.potential_output_path.is_file():
                raise ODASProtocolError('ODAS did not create a potential JSON output file; verify SSL sink configuration')
            raw=b.potential_output_path.read_text(encoding='utf8')
            directions=parse_potential_json(raw,frame_id=geometry.frame_id,max_candidates=b.max_candidates)
            return SpatialResult(audio.capture,geometry.frame_id,directions,'odas_offline_ssl')
        finally:
            b.raw_input_path.unlink(missing_ok=True)
            b.potential_output_path.unlink(missing_ok=True)
