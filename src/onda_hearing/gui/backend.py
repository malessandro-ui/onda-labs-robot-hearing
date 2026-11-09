"""Validated GUI adapter over the v0.17 acoustic-scene renderer."""
from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from io import BytesIO
import importlib.util
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
import wave

import numpy as np

from onda_hearing.acoustic_scene import (
    AcousticSceneRenderer,
    AudioAsset,
    Binding,
    Emitter,
    MicArray,
    Pose3,
    RendererConfig,
    Schedule,
    StaticPoseResolver,
)
from onda_hearing.acoustic_scene.assets import decode_wav
from onda_hearing.policy import AcousticEvidence, PolicyHearingConfig, PolicyHearingEncoder
from onda_hearing.streaming.core import GCCPHATSRP


SAMPLE_RATE = 16_000
MAX_EMITTERS = 8
MAX_DURATION_S = 10.0
MAX_WAV_BYTES = 330_000
_brain_lock = Lock()

MIC_PRESETS = {
    "onda_reference_4": np.array(
        [[.050, .030, 0.], [-.045, .035, 0.], [-.050, -.038, 0.], [.042, -.047, 0.]],
        dtype=np.float64,
    ),
    "square_4": np.array(
        [[-.035, -.035, 0.], [.035, -.035, 0.], [.035, .035, 0.], [-.035, .035, 0.]],
        dtype=np.float64,
    ),
    "linear_4": np.array(
        [[-.045, 0., 0.], [-.015, 0., 0.], [.015, 0., 0.], [.045, 0., 0.]],
        dtype=np.float64,
    ),
}

ROOM_PRESETS = {
    "direct": None,
    "small_lab": ((123, .18), (317, .09)),
    "office": ((149, .22), (401, .13), (689, .06)),
}

SOUND_TYPES = ("broadband", "telephone", "alarm", "speech_like")


class GuiInputError(ValueError):
    """Safe validation error suitable for returning to a local GUI client."""


def capabilities() -> dict:
    semantic = _beats_capability()
    return {
        "sample_rate_hz": SAMPLE_RATE,
        "max_emitters": MAX_EMITTERS,
        "max_duration_s": MAX_DURATION_S,
        "microphone_presets": [
            {"id": key, "channels": len(value), "positions_m": value.tolist()}
            for key, value in MIC_PRESETS.items()
        ],
        "room_presets": list(ROOM_PRESETS),
        "sound_types": list(SOUND_TYPES),
        "enhancement_backends": [
            {"id": "raw", "label": "Raw acoustic scene", "available": True},
            {
                "id": "waveformer",
                "label": "Waveformer",
                "available": False,
                "reason": "Waveformer runtime adapter is not part of v0.17 yet",
            },
        ],
        "semantic_backend": semantic,
        "renderer": "Onda v0.17 deterministic CPU reference",
    }


def _beats_capability() -> dict:
    checkpoint = Path(os.path.expandvars(os.getenv("ONDA_BEATS_CHECKPOINT", ""))).expanduser()
    labels = Path(os.path.expandvars(os.getenv("ONDA_BEATS_LABELS", ""))).expanduser()
    module_dir = Path(os.path.expandvars(os.getenv("ONDA_BEATS_MODULE_DIR", ""))).expanduser()
    trusted = os.getenv("ONDA_GUI_ALLOW_TRUSTED_BEATS") == "1"
    checks = {
        "checkpoint": bool(str(checkpoint)) and checkpoint.is_file(),
        "labels": bool(str(labels)) and labels.is_file(),
        "module": bool(str(module_dir)) and (module_dir / "BEATs.py").is_file(),
        "torch": importlib.util.find_spec("torch") is not None,
        "trusted_opt_in": trusted,
    }
    available = all(checks.values())
    missing = [name for name, ready in checks.items() if not ready]
    return {
        "id": "onda_beats",
        "label": "Onda Sound Brain · BEATs",
        "available": available,
        "checks": checks,
        "reason": "ready" if available else "Missing configuration: " + ", ".join(missing),
        "window_scope": "whole_audio_window",
        "association_valid": False,
    }


@lru_cache(maxsize=1)
def _beats_brain():
    from onda_hearing.semantics import OndaSoundBrain

    return OndaSoundBrain(
        checkpoint_path=os.environ["ONDA_BEATS_CHECKPOINT"],
        labels_path=os.environ["ONDA_BEATS_LABELS"],
        beats_module_dir=os.environ["ONDA_BEATS_MODULE_DIR"],
        device=os.getenv("ONDA_BEATS_DEVICE"),
        allow_unsafe_checkpoint=True,
    )


def _classify_semantics(pcm: np.ndarray) -> dict:
    capability = _beats_capability()
    base = {
        "brain_name": capability["label"],
        "window_scope": "whole_audio_window",
        "association_valid": False,
        "events": [],
    }
    if not capability["available"]:
        return {**base, "status": "not_configured", "valid": False, "reason": capability["reason"]}
    if len(pcm) < SAMPLE_RATE:
        return {**base, "status": "insufficient_window", "valid": False, "reason": "BEATs requires at least 1 second of PCM"}
    try:
        with _brain_lock:
            events = _beats_brain().classify(np.ascontiguousarray(pcm[-SAMPLE_RATE * 2:, 0]), SAMPLE_RATE)
        return {
            **base,
            "status": "ok" if events else "no_event",
            "valid": bool(events),
            "reason": "ok" if events else "no event exceeded the configured threshold",
            "events": [{"label": event.label, "score": event.score} for event in events],
        }
    except Exception as exc:
        return {
            **base,
            "status": "brain_error",
            "valid": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _number(value, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise GuiInputError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GuiInputError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not lower <= result <= upper:
        raise GuiInputError(f"{name} must be within [{lower}, {upper}]")
    return result


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise GuiInputError(f"{name} must be boolean")
    return value


def _text(value, name: str, allowed: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise GuiInputError(f"{name} must be a short nonempty string")
    result = value.strip()
    if allowed is not None and result not in allowed:
        raise GuiInputError(f"unsupported {name}: {result}")
    return result


def _asset(kind: str, seed: int) -> AudioAsset:
    duration = 2.0
    count = int(duration * SAMPLE_RATE)
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    if kind == "broadband":
        samples = rng.normal(0., .22, count)
        samples = np.convolve(samples, np.array([.18, .64, .18]), mode="same")
    elif kind == "telephone":
        carrier = np.sin(2 * np.pi * 440 * t) + .55 * np.sin(2 * np.pi * 480 * t)
        gate = (np.mod(t, .72) < .32).astype(float)
        samples = .16 * carrier * gate
    elif kind == "alarm":
        carrier = np.sin(2 * np.pi * (760 + 180 * np.sin(2 * np.pi * 1.5 * t)) * t)
        samples = .20 * carrier
    else:
        excitation = rng.normal(0., 1., count)
        envelope = .15 + .85 * np.square(np.sin(2 * np.pi * 2.7 * t))
        samples = .12 * envelope * (
            np.convolve(excitation, np.ones(5) / 5, mode="same")
            + .5 * np.sin(2 * np.pi * 180 * t)
        )
    peak = max(float(np.max(np.abs(samples))), 1e-9)
    samples = np.asarray(samples * min(1., .32 / peak), dtype=np.float32)
    return AudioAsset(kind, samples, SAMPLE_RATE)


def _rir(room: str, channels: int) -> np.ndarray | None:
    echoes = ROOM_PRESETS[room]
    if echoes is None:
        return None
    taps = np.zeros((max(delay for delay, _ in echoes) + 1, channels), dtype=np.float32)
    taps[0] = 1.
    for index, (delay, gain) in enumerate(echoes):
        # Small deterministic channel differences avoid pretending this is a
        # geometrically correct room simulation.
        taps[delay] = gain * (1. - .025 * np.arange(channels)) * (1. if index % 2 == 0 else -.65)
    return taps


def _quaternion_facing_robot(bearing_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(bearing_deg + 180.)
    return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))


def _normalize(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise GuiInputError("scenario must be a JSON object")
    duration_s = _number(payload.get("duration_s", 2.4), "duration_s", .25, MAX_DURATION_S)
    noise_std = _number(payload.get("noise_std", .002), "noise_std", 0., .1)
    mic_preset = _text(payload.get("mic_preset", "onda_reference_4"), "mic_preset")
    if mic_preset not in MIC_PRESETS:
        raise GuiInputError(f"unknown microphone preset: {mic_preset}")
    room = _text(payload.get("room", "direct"), "room")
    if room not in ROOM_PRESETS:
        raise GuiInputError(f"unknown room preset: {room}")
    processing = _text(payload.get("processing", "raw"), "processing")
    if processing != "raw":
        raise GuiInputError("Waveformer is visible as a planned adapter but is not available in v0.17")
    raw_emitters = payload.get("emitters", [])
    if not isinstance(raw_emitters, list) or len(raw_emitters) > MAX_EMITTERS:
        raise GuiInputError(f"emitters must be a list with at most {MAX_EMITTERS} entries")
    emitters = []
    seen = set()
    for index, raw in enumerate(raw_emitters):
        if not isinstance(raw, dict):
            raise GuiInputError(f"emitter {index + 1} must be an object")
        emitter_id = _text(raw.get("id", f"source_{index + 1}"), "emitter id")
        if emitter_id in seen:
            raise GuiInputError(f"duplicate emitter id: {emitter_id}")
        seen.add(emitter_id)
        sound = _text(raw.get("sound", "broadband"), "sound", (*SOUND_TYPES, "wav"))
        bearing_deg = _number(raw.get("bearing_deg", 38.), "bearing_deg", -180., 180.)
        distance_m = _number(raw.get("distance_m", 2.), "distance_m", .15, 25.)
        gain_db = _number(raw.get("gain_db", -3.), "gain_db", -120., 20.)
        onset_s = _number(raw.get("onset_s", .1), "onset_s", 0., duration_s)
        source_duration = raw.get("duration_s")
        if source_duration is not None:
            source_duration = _number(source_duration, "emitter duration_s", .01, duration_s)
        directivity = _text(raw.get("directivity", "omni"), "directivity", ("omni", "cardioid"))
        enabled = _boolean(raw.get("enabled", True), "enabled")
        loop = _boolean(raw.get("loop", True), "loop")
        emitters.append(
            {
                "id": emitter_id,
                "sound": sound,
                "bearing_deg": bearing_deg,
                "distance_m": distance_m,
                "gain_db": gain_db,
                "onset_s": onset_s,
                "duration_s": source_duration,
                "directivity": directivity,
                "enabled": enabled,
                "loop": loop,
            }
        )
        if sound == "wav":
            emitters[-1]["wav_name"] = _text(raw.get("wav_name", "Custom WAV"), "WAV name")
    return {
        "duration_s": duration_s,
        "noise_std": noise_std,
        "mic_preset": mic_preset,
        "room": room,
        "processing": processing,
        "emitters": emitters,
    }


def _monitor_pcm16(pcm: np.ndarray) -> bytes:
    mono = np.mean(pcm, axis=1)
    return np.asarray(np.clip(mono, -1., 1.) * 32767., dtype="<i2").tobytes()


def _wav_base64(pcm: np.ndarray) -> str:
    peak = max(1., float(np.max(np.abs(np.mean(pcm, axis=1)))))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(_monitor_pcm16(pcm / peak))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _waveform(pcm: np.ndarray, points: int = 480) -> list[list[float]]:
    bins = min(points, len(pcm))
    boundaries = np.linspace(0, len(pcm), bins + 1, dtype=int)
    channels = []
    for channel in range(pcm.shape[1]):
        values = [float(np.max(np.abs(pcm[boundaries[i]:boundaries[i + 1], channel]))) for i in range(bins)]
        channels.append(values)
    return channels


def _best_window(pcm: np.ndarray, size: int = 2048) -> np.ndarray:
    if len(pcm) <= size:
        return pcm
    hop = size // 2
    starts = range(0, len(pcm) - size + 1, hop)
    start = max(starts, key=lambda item: float(np.mean(np.square(pcm[item:item + size]))))
    return pcm[start:start + size]


def _angle_error(estimate: float, truth: float) -> float:
    return abs((estimate - truth + 180.) % 360. - 180.)


def _uploaded_asset(encoded, asset_id: str) -> AudioAsset:
    if not isinstance(encoded, str) or len(encoded) > (MAX_WAV_BYTES + 2) // 3 * 4:
        raise GuiInputError("Choose a WAV file up to 330 KB")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_WAV_BYTES:
            raise ValueError("file is larger than 330 KB")
        asset = decode_wav(BytesIO(raw), asset_id)
        if asset.sample_rate != SAMPLE_RATE or len(asset.samples) > 10 * SAMPLE_RATE:
            raise ValueError("use 16 kHz audio up to 10 seconds")
    except (ValueError, EOFError, wave.Error, binascii.Error) as exc:
        raise GuiInputError(f"Invalid WAV: {exc}") from exc
    return asset


def build_scene(payload: object) -> tuple[dict, AcousticSceneRenderer, np.ndarray]:
    """Bind each GUI object and its audio asset to a fixed scene pose."""
    scenario = _normalize(payload)
    positions = MIC_PRESETS[scenario["mic_preset"]]
    assets: dict[str, AudioAsset] = {}
    poses = {}
    emitters = []
    for index, item in enumerate(scenario["emitters"]):
        object_path = f"/scene/objects/{index}"
        asset_id = item["sound"]
        if asset_id == "wav":
            asset_id = f"{object_path}/audio"
            assets[asset_id] = _uploaded_asset(payload["emitters"][index].get("wav_base64"), asset_id)
        elif asset_id not in assets:
            assets[asset_id] = _asset(asset_id, 700 + index)
        bearing = math.radians(item["bearing_deg"])
        xyz = (
            item["distance_m"] * math.cos(bearing),
            item["distance_m"] * math.sin(bearing),
            0.,
        )
        poses[object_path] = Pose3(xyz, _quaternion_facing_robot(item["bearing_deg"]))
        emitters.append(
            Emitter(
                item["id"],
                asset_id,
                Binding(object_path=object_path),
                Schedule(item["onset_s"], item["duration_s"], item["loop"]),
                item["gain_db"],
                item["directivity"],
                item["enabled"],
            )
        )
    microphones = MicArray(Binding(world=Pose3((0., 0., 0.))), tuple(map(tuple, positions)))
    renderer = AcousticSceneRenderer(
        RendererConfig(
            sample_rate=SAMPLE_RATE,
            noise_std=scenario["noise_std"],
            seed=73,
            rir_taps=_rir(scenario["room"], len(positions)),
        ),
        microphones,
        emitters,
        assets,
        resolver=StaticPoseResolver(poses),
    )
    return scenario, renderer, positions


def simulate_scenario(payload: object) -> dict:
    """Render, localize and summarize one validated scene."""
    started = perf_counter()
    scenario, renderer, positions = build_scene(payload)
    total = int(round(scenario["duration_s"] * SAMPLE_RATE))
    chunks = []
    cursor = 0
    while cursor < total:
        count = min(SAMPLE_RATE * 2, total - cursor)
        chunks.append(renderer.render(cursor, count))
        cursor += count
    pcm = np.concatenate(chunks, axis=0)
    estimate = GCCPHATSRP(positions, SAMPLE_RATE).estimate(_best_window(pcm))
    encoder = PolicyHearingEncoder(PolicyHearingConfig(frame_id="array", evidence_scale=1.))
    encoder.ingest(
        AcousticEvidence(
            time_ns=total * 1_000_000_000 // SAMPLE_RATE,
            azimuth_rad=math.radians(estimate.local_bearing_deg) if estimate.valid else None,
            score=estimate.confidence,
            frame_id="array",
            accepted=estimate.valid,
        )
    )
    policy = encoder.read(total * 1_000_000_000 // SAMPLE_RATE)
    active = [item for item in scenario["emitters"] if item["enabled"]]
    truth_bearing = active[0]["bearing_deg"] if len(active) == 1 else None
    angular_error = (
        _angle_error(estimate.local_bearing_deg, truth_bearing)
        if estimate.valid and truth_bearing is not None
        else None
    )
    rms = np.sqrt(np.mean(np.square(pcm), axis=0))
    semantic = _classify_semantics(pcm)
    return {
        "scenario": scenario,
        "renderer": "Onda v0.17 AcousticSceneRenderer",
        "sample_rate_hz": SAMPLE_RATE,
        "samples_per_channel": len(pcm),
        "channels": pcm.shape[1],
        "render_ms": (perf_counter() - started) * 1000.,
        "telemetry": renderer.telemetry(),
        "spatial": {
            "valid": estimate.valid,
            "bearing_deg": estimate.local_bearing_deg if estimate.valid else None,
            "confidence": estimate.confidence,
            "evidence": estimate.evidence,
            "evaluation_truth_deg": truth_bearing,
            "evaluation_error_deg": angular_error,
            "association_valid": len(active) == 1,
        },
        "policy": {
            "names": ["bearing_sin", "bearing_cos", "evidence_scaled", "valid", "age_s"],
            "values": policy.tolist(),
        },
        "semantic": semantic,
        "audio": {
            "peak": float(np.max(np.abs(pcm))),
            "rms_per_channel": rms.tolist(),
            "waveform": _waveform(pcm),
            "wav_base64": _wav_base64(pcm),
        },
        "limitations": [
            "Direct-path reference renderer; room presets are stationary FIR approximations",
            "Evaluation truth is never passed into localization or policy encoding",
            "Multiple-source estimates are not associated with a specific emitter",
            "Waveformer and physical Infineon capture require separate runtime adapters",
        ],
    }
