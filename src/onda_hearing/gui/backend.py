"""Validated GUI adapter over the real v0.17 acoustic-scene renderer.

The workbench is intentionally an offline reference renderer. Source truth is
used only for evaluation fields; localization and policy features consume PCM.
"""
from __future__ import annotations

import base64
from io import BytesIO
import math
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
)
from onda_hearing.policy import AcousticEvidence, PolicyHearingConfig, PolicyHearingEncoder
from onda_hearing.streaming.core import GCCPHATSRP


SAMPLE_RATE = 16_000
MAX_EMITTERS = 8
MAX_DURATION_S = 10.0

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
        "renderer": "Onda v0.17 deterministic CPU reference",
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
        sound = _text(raw.get("sound", "broadband"), "sound", SOUND_TYPES)
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
    return {
        "duration_s": duration_s,
        "noise_std": noise_std,
        "mic_preset": mic_preset,
        "room": room,
        "processing": processing,
        "emitters": emitters,
    }


def _wav_base64(pcm: np.ndarray) -> str:
    mono = np.mean(pcm, axis=1)
    peak = max(1., float(np.max(np.abs(mono))))
    audio = np.asarray(np.clip(mono / peak, -1., 1.) * 32767., dtype="<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(audio.tobytes())
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


def simulate_scenario(payload: object) -> dict:
    """Render, localize and summarize one validated fixed-world scene."""
    scenario = _normalize(payload)
    started = perf_counter()
    positions = MIC_PRESETS[scenario["mic_preset"]]
    assets: dict[str, AudioAsset] = {}
    emitters = []
    for index, item in enumerate(scenario["emitters"]):
        if item["sound"] not in assets:
            assets[item["sound"]] = _asset(item["sound"], 700 + index)
        bearing = math.radians(item["bearing_deg"])
        xyz = (
            item["distance_m"] * math.cos(bearing),
            item["distance_m"] * math.sin(bearing),
            0.,
        )
        emitters.append(
            Emitter(
                item["id"],
                item["sound"],
                Binding(world=Pose3(xyz, _quaternion_facing_robot(item["bearing_deg"]))),
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
    )
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
