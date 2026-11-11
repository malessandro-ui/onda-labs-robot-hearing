"""Optional Pyroomacoustics 0.10.1 static shoebox room backend.

This is a separate alternative to geometry direct-path transmission:
Pyroomacoustics provides actual image-source reflection paths and RIRs for
an idealized shoebox, but does not model dynamic door opening, complex USD
occlusion, or validated material transmission through room partitions.

Not mixed with Onda's existing direct paths: doing so would double-count
the direct sound in each Pyroomacoustics room impulse response.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RoomIRResult:
    samples: np.ndarray  # [time, microphones], full tails kept
    rir: tuple[tuple[np.ndarray, ...], ...]  # [microphone][source]
    sample_rate: int
    runtime_metadata: dict


def simulate_shoebox(
    signals: list[np.ndarray],
    sources_m: np.ndarray,
    microphones_m: np.ndarray,
    dimensions_m: tuple[float, float, float],
    *,
    sample_rate: int = 16000,
    max_order: int = 3,
    energy_absorption: float | None = None,
    air_absorption: bool = False,
    max_rir_duration_s: float = 3.0,
) -> RoomIRResult:
    """Simulate a static reflective room without suppressing reflected tails.

    energy_absorption is an **explicit**, frequency-flat surface energy
    absorption value, *not* a claimed preset for particular physical walls.
    For measured octaves, build a project-specific pra.Material externally.
    Receivers and sources must be INSIDE the room bounds (origin at corner).
    """
    if energy_absorption is None:
        raise ValueError("explicit energy_absorption required (calibration provenance)")
    if not (0. <= float(energy_absorption) <= 1.) or not np.isfinite(energy_absorption):
        raise ValueError("energy_absorption must be finite in [0,1]")
    if not isinstance(sample_rate, int) or sample_rate < 2000:
        raise ValueError("invalid sample_rate")
    if not isinstance(max_order, int) or not 0 <= max_order <= 12:
        raise ValueError("max_order must be in [0,12]")
    if not np.isfinite(max_rir_duration_s) or max_rir_duration_s <= 0:
        raise ValueError("invalid maximum RIR duration")
    dims = np.asarray(dimensions_m, dtype=np.float64)
    src = np.asarray(sources_m, dtype=np.float64)
    mic = np.asarray(microphones_m, dtype=np.float64)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("positive 3D room dimensions required")
    if src.ndim != 2 or src.shape[1] != 3 or len(src) != len(signals) or len(src) < 1:
        raise ValueError("sources_m must be [number of signals, 3]")
    if mic.ndim != 2 or mic.shape[1] != 3 or len(mic) < 1:
        raise ValueError("microphones_m must be [microphone count,3]")
    for label, pts in (("sources", src), ("microphones", mic)):
        if not np.isfinite(pts).all() or np.any(pts <= 0) or np.any(pts >= dims):
            raise ValueError(f"{label} must lie strictly inside the room")
    arrays = []
    for signal in signals:
        s = np.asarray(signal, dtype=np.float64)
        if s.ndim != 1 or len(s) < 1 or not np.isfinite(s).all():
            raise ValueError("each source signal must be a finite nonempty mono array")
        arrays.append(s)

    try:
        import pyroomacoustics as pra
        from scipy.signal import fftconvolve
    except ImportError as exc:
        raise ImportError(
            "Static room backend requires pyroomacoustics==0.10.1 and scipy"
        ) from exc

    room = pra.ShoeBox(
        dims, fs=sample_rate, max_order=max_order,
        materials=pra.Material(energy_absorption=float(energy_absorption)),
        air_absorption=air_absorption,
    )
    for point in src:
        room.add_source(point)
    room.add_microphone_array(mic.T)
    room.compute_rir()
    rirs = tuple(tuple(np.asarray(room.rir[m][s], dtype=np.float64).copy()
                       for s in range(len(signals))) for m in range(len(mic)))
    max_length = max(map(len, (rir for one_mic in rirs for rir in one_mic)))
    if max_length > int(max_rir_duration_s*sample_rate):
        raise ValueError("room impulse response exceeds configured maximum; change parameters")
    total = max(len(x) for x in arrays) + max_length - 1
    output = np.zeros((total, len(mic)), dtype=np.float64)
    for m, receiver_rirs in enumerate(rirs):
        for sig, impulse in zip(arrays, receiver_rirs):
            convolved = fftconvolve(sig, impulse, mode="full")
            output[:len(convolved), m] += convolved
    if not np.isfinite(output).all():
        raise FloatingPointError("invalid output from Pyroomacoustics convolution")
    return RoomIRResult(
        output.astype(np.float32), rirs, sample_rate,
        {
            "backend": "pyroomacoustics_shoebox_static",
            "room_dimensions_m": list(map(float, dims)),
            "energy_absorption": float(energy_absorption),
            "absorption_evidence": "user_supplied_not_calibrated_by_sdk",
            "max_order": max_order,
            "air_absorption": air_absorption,
            "source_count": len(arrays),
            "microphone_count": len(mic),
            "max_rir_samples": max_length,
            "peak_abs": float(np.max(np.abs(output))),
            "clipping_applied": False,
        },
    )
