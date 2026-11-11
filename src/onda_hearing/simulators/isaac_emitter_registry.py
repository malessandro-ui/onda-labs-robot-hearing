"""Onda v0.21: validated USD prim -> WAV emitter bindings and manifest.

No Isaac dependency at import time; stage validation is injected, and all USD
calls occur on the Kit application thread. Read/write the manifest under the
user's chosen working directory, never into NVIDIA's source USD asset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
import json
import math
import os
import re
import tempfile
import uuid
import wave

from onda_hearing.acoustic_scene.types import Binding, Emitter, Schedule
from onda_hearing.acoustic_scene.assets import load_wav

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_PRIM = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_MAX_AUDIO_BYTES = 64 * 1024 * 1024


class RegistryError(ValueError):
    """Well-formed but invalid emitter configuration."""


class RegistryConflict(RegistryError):
    pass


def validate_prim_path(path: str) -> str:
    # USD names can contain Unicode and valid USD constructs not matched here;
    # this deliberately accepts a strict safe subset for HTTP requests.
    if not isinstance(path, str) or len(path) > 1024 or not _PRIM.fullmatch(path):
        raise RegistryError("prim_path must be an absolute, simple USD prim path")
    return path


def validate_wav_path(audio_root: Path, value: str, sample_rate: int) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 1024:
        raise RegistryError("audio_asset must be a nonempty WAV path")
    root = audio_root.expanduser().resolve(strict=True)
    path = Path(value).expanduser()
    try:
        path = (path if path.is_absolute() else root / path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise RegistryError("WAV asset not found inside configured Onda audio root") from exc
    if root not in path.parents:
        raise RegistryError("audio_asset must be inside configured ONDA audio root")
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise RegistryError("audio_asset must be an existing .wav file")
    if path.stat().st_size > _MAX_AUDIO_BYTES:
        raise RegistryError("audio_asset exceeds 64 MiB limit")
    try:
        with wave.open(str(path), "rb") as w:
            if w.getcomptype() != "NONE" or w.getnchannels() != 1 or w.getsampwidth() not in (1, 2, 3, 4):
                raise RegistryError("requires uncompressed mono PCM WAV, 8/16/24/32 bit")
            if w.getframerate() != sample_rate:
                raise RegistryError(f"WAV sample rate {w.getframerate()} != configured {sample_rate}; resample explicitly")
            if w.getnframes() <= 0:
                raise RegistryError("WAV contains no samples")
    except (wave.Error, EOFError) as exc:
        raise RegistryError(f"invalid PCM WAV: {exc}") from exc
    return path


@dataclass(frozen=True)
class EmitterRecord:
    emitter_id: str
    prim_path: str
    audio_asset: str  # normalized relative to the configured audio root
    gain_db: float = 0.
    loop: bool = True
    enabled: bool = True
    onset_s: float | None = 0.
    duration_s: float | None = None
    directivity: str = "omni"
    local_offset_m: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if not isinstance(self.emitter_id, str) or not _ID.fullmatch(self.emitter_id):
            raise RegistryError("emitter_id must begin with a letter, max 64 characters")
        validate_prim_path(self.prim_path)
        if not isinstance(self.audio_asset, str) or not self.audio_asset:
            raise RegistryError("audio_asset required")
        if not isinstance(self.gain_db, (float, int)) or not math.isfinite(self.gain_db) or not -120 <= self.gain_db <= 40:
            raise RegistryError("gain_db must be between -120 and 40")
        if type(self.loop) is not bool or type(self.enabled) is not bool:
            raise RegistryError("loop/enabled must be booleans")
        if self.directivity not in ("omni", "cardioid"):
            raise RegistryError("unsupported emitter directivity")
        off = tuple(float(v) for v in self.local_offset_m)
        if len(off) != 3 or not all(math.isfinite(v) and abs(v) <= 1000 for v in off):
            raise RegistryError("local_offset_m must be finite XYZ meters")
        object.__setattr__(self, "local_offset_m", off)
        Schedule(onset_s=self.onset_s, duration_s=self.duration_s, loop=self.loop)

    def to_onda(self) -> Emitter:
        return Emitter(
            emitter_id=self.emitter_id,
            asset_id=self.emitter_id,
            binding=Binding(object_path=self.prim_path, local_offset_m=self.local_offset_m),
            schedule=Schedule(onset_s=self.onset_s, duration_s=self.duration_s, loop=self.loop),
            gain_db=float(self.gain_db),
            directivity=self.directivity,
            enabled=self.enabled,
        )


class IsaacEmitterRegistry:
    """Synchronous stage-owned registry. All mutations on Kit app thread."""

    MAX_EMITTERS = 32

    def __init__(self, *, stage, audio_root: str | Path, manifest: str | Path,
                 scene_id: str, sample_rate: int = 16000):
        if stage is None:
            raise ValueError("stage required")
        if not isinstance(sample_rate, int) or not 1000 <= sample_rate <= 192000:
            raise ValueError("invalid sample_rate")
        self.stage = stage
        self.audio_root = Path(audio_root).expanduser().resolve(strict=True)
        self.manifest = Path(manifest).expanduser().resolve(strict=False)
        self.scene_id = scene_id
        self.sample_rate = sample_rate
        self._records: dict[str, EmitterRecord] = {}
        self.revision = 0

    def _check_prim(self, path: str):
        path = validate_prim_path(path)
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsActive():
            raise RegistryError(f"USD prim does not exist or is inactive: {path}")
        try:
            from pxr import UsdGeom
        except ImportError:
            pass  # unit-test fake stages can validate without USD installed
        else:
            if not prim.IsA(UsdGeom.Xformable):
                raise RegistryError("sound emitter must attach to an Xformable USD object, not a shader/material")
        return prim

    def add(self, payload: Mapping) -> EmitterRecord:
        if not isinstance(payload, Mapping):
            raise RegistryError("emitter body must be an object")
        allowed = {"emitter_id", "prim_path", "audio_asset", "gain_db", "loop",
                   "enabled", "onset_s", "duration_s", "directivity", "local_offset_m"}
        unexpected = set(payload) - allowed
        if unexpected:
            raise RegistryError(f"unknown emitter fields: {', '.join(sorted(unexpected))}")
        if len(self._records) >= self.MAX_EMITTERS:
            raise RegistryError(f"maximum {self.MAX_EMITTERS} emitters")
        path = payload.get("prim_path")
        if not path:
            raise RegistryError("prim_path required (GUI may substitute selected prim)")
        self._check_prim(path)
        if "audio_asset" not in payload:
            raise RegistryError("audio_asset required")
        wav_path = validate_wav_path(self.audio_root, payload["audio_asset"], self.sample_rate)
        eid = payload.get("emitter_id") or ("onda_" + uuid.uuid4().hex[:12])
        if eid in self._records:
            raise RegistryConflict(f"duplicate emitter_id {eid}")
        rec = EmitterRecord(
            emitter_id=eid,
            prim_path=path,
            audio_asset=str(wav_path.relative_to(self.audio_root)),
            gain_db=payload.get("gain_db", 0.),
            loop=payload.get("loop", True),
            enabled=payload.get("enabled", True),
            onset_s=payload.get("onset_s", 0.),
            duration_s=payload.get("duration_s"),
            directivity=payload.get("directivity", "omni"),
            local_offset_m=tuple(payload.get("local_offset_m", (0., 0., 0.))),
        )
        self._records[eid] = rec
        self.revision += 1
        try:
            self.save()
        except Exception:
            del self._records[eid]
            self.revision -= 1
            raise
        return rec

    def remove(self, emitter_id: str) -> dict:
        if not isinstance(emitter_id, str) or emitter_id not in self._records:
            raise RegistryError("emitter_id does not exist")
        previous = self._records.pop(emitter_id)
        self.revision += 1
        try:
            self.save()
        except Exception:
            self._records[emitter_id] = previous
            self.revision -= 1
            raise
        return {"deleted": emitter_id}

    def update(self, emitter_id: str, payload: Mapping) -> EmitterRecord:
        """Validated atomic edit. Never partially mutate manifest or live renderer."""
        if emitter_id not in self._records:
            raise RegistryError("emitter_id does not exist")
        if not isinstance(payload, Mapping):
            raise RegistryError("emitter edit body must be an object")
        allowed = {"audio_asset", "gain_db", "loop", "enabled", "onset_s",
                   "duration_s", "directivity", "local_offset_m", "prim_path"}
        if set(payload) - allowed:
            raise RegistryError(f"unknown emitter edit fields: {sorted(set(payload)-allowed)}")
        prev = self._records[emitter_id]
        fields = asdict(prev)
        fields.update(payload)
        if fields["prim_path"] != prev.prim_path:
            self._check_prim(fields["prim_path"])
        wav = validate_wav_path(self.audio_root, fields["audio_asset"], self.sample_rate)
        fields["audio_asset"] = str(wav.relative_to(self.audio_root))
        updated = EmitterRecord(**fields)
        self._records[emitter_id] = updated
        self.revision += 1
        try:
            self.save()
        except Exception:
            self._records[emitter_id] = prev
            self.revision -= 1
            raise
        return updated

    def list_records(self) -> list[dict]:
        return [asdict(rec) for rec in self._records.values()]

    def save(self):
        """Crash-safe atomic JSON replace; no edits to original USD layers."""
        parent = self.manifest.parent
        parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1, "scene_id": self.scene_id,
            "sample_rate": self.sample_rate,
            "emitters": self.list_records(),
        }
        fd, name = tempfile.mkstemp(prefix=".onda_emitters_", suffix=".tmp", dir=str(parent))
        try:
            os.chmod(name, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.manifest)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self) -> dict:
        """Never import stale emitter bindings from a different USD stage."""
        if not self.manifest.exists():
            return {"loaded": 0, "reason": "no_manifest"}
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise RegistryError("unsupported manifest schema")
        if data.get("scene_id") != self.scene_id:
            return {"loaded": 0, "reason": "scene_id_mismatch"}
        if data.get("sample_rate") != self.sample_rate:
            return {"loaded": 0, "reason": "sample_rate_mismatch"}
        emitters = data.get("emitters")
        if not isinstance(emitters, list) or len(emitters) > self.MAX_EMITTERS:
            raise RegistryError("invalid emitter manifest list")
        restored = {}
        rejected = []
        for payload in emitters:
            try:
                rec = EmitterRecord(**payload)
                self._check_prim(rec.prim_path)
                validate_wav_path(self.audio_root, rec.audio_asset, self.sample_rate)
                if rec.emitter_id in restored:
                    raise RegistryError("duplicate id")
                restored[rec.emitter_id] = rec
            except (RegistryError, ValueError, TypeError, FileNotFoundError) as exc:
                rejected.append(str(exc))
        self._records = restored
        self.revision += 1
        return {"loaded": len(restored), "rejected": rejected[:32]}

    def for_renderer(self):
        """Return v0.20-compatible emitter and asset definitions.

        Note: building a renderer is explicit. A registry edit does not mutate
        a live renderer's emitter list; the app must rebuild between audio runs.
        """
        emitters = [r.to_onda() for r in self._records.values()]
        assets = {r.emitter_id: load_wav(self.audio_root / r.audio_asset, r.emitter_id)
                  for r in self._records.values()}
        return emitters, assets
