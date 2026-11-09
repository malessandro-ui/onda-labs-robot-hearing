"""Reusable, bounded-memory soak-test utilities. No Isaac Lab dependency.

Not part of Onda's public SDK. These helpers deliberately separate capture-
clock latency from simulator time and never call into an inference worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import numpy as np


NS = 1_000_000_000


def angle_error(actual: float, expected: float) -> float:
    return abs((float(actual) - float(expected) + 180.0) % 360.0 - 180.0)


def yaw_xyzw(degrees: float) -> tuple[float, float, float, float]:
    a = math.radians(degrees) * 0.5
    return 0.0, 0.0, math.sin(a), math.cos(a)


def yaw_at_seconds(seconds: float, period_seconds: float = 24.0) -> float:
    """Continuously differentiable synthetic microphone yaw, amplitude 50 deg."""
    return 50.0 * math.sin(2.0 * math.pi * seconds / period_seconds)


def sample_timestamp_ns(origin_ns: int, sample: int, rate: int) -> int:
    return origin_ns + (int(sample) * NS + rate // 2) // rate


class CyclicSource:
    """Bounded-memory broadband source for analytic microphone simulations.

    The waveform repeats every 131072 samples (8.192 s at 16kHz). It must NOT
    be used to claim real-world robustness to reverberation, noise or speech.
    """
    def __init__(self, seed: int = 4821, length: int = 131_072):
        if length < 4096:
            raise ValueError('waveform period too short')
        self.wave = np.random.default_rng(seed).standard_normal(length).astype(np.float32)

    def render(self, start_sample: int, frames: int, sample_rate: int,
               array_pos, array_quat, source_pos, mic_positions,
               gain: float = 0.35, sound_speed: float = 343.0) -> np.ndarray:
        from onda_hearing.streaming.core import rotate_xyzw
        tick = np.arange(start_sample, start_sample + frames, dtype=np.float64)
        period = len(self.wave)
        x = np.empty((frames, len(mic_positions)), dtype=np.float32)
        center = np.asarray(array_pos, dtype=np.float64)
        source = np.asarray(source_pos, dtype=np.float64)
        for i, local in enumerate(mic_positions):
            microphone = center + rotate_xyzw(array_quat, local)
            d = float(np.linalg.norm(source - microphone))
            pos = (tick - (d * sample_rate / sound_speed)) % period
            low = np.floor(pos).astype(np.int64)
            fraction = pos - low
            x[:, i] = ((1.0 - fraction) * self.wave[low] +
                        fraction * self.wave[(low + 1) % period]) * (gain / max(1.0, d))
        return x


class LatencyHistogram:
    """Fixed-memory ~0.1 ms quantile histogram, with explicit high overflow.

    No arbitrary truncation of recorded data: samples outside the range are
    accounted for and percentile is reported >= max range when appropriate.
    """
    def __init__(self, bin_ms: float = 0.1, max_ms: float = 5000.0):
        self.bin_ms = bin_ms
        self.max_ms = max_ms
        self.buckets = np.zeros(math.ceil(max_ms/bin_ms), dtype=np.uint64)
        self.overflow = 0
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def add(self, milliseconds: float):
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError('latency must be nonnegative and finite')
        self.count += 1
        self.total += milliseconds
        self.maximum = max(self.maximum, milliseconds)
        index = int(milliseconds / self.bin_ms)
        if index >= len(self.buckets):
            self.overflow += 1
        else:
            self.buckets[index] += 1

    def percentile(self, q: float) -> float | None:
        if not self.count:
            return None
        rank = math.ceil(self.count*q/100)
        running = 0
        for k in np.flatnonzero(self.buckets):
            running += int(self.buckets[k])
            if running >= rank:
                return round((int(k)+1) * self.bin_ms, 4)
        return self.max_ms  # lower bound if overflow dominates the percentile

    def summary(self):
        return {'count': self.count, 'p50_ms': self.percentile(50),
                'p95_ms': self.percentile(95), 'p99_ms': self.percentile(99),
                'mean_ms': self.total/self.count if self.count else None,
                'maximum_ms': self.maximum if self.count else None,
                'over_range_count': self.overflow,
                'quantile_bin_ms': self.bin_ms}


def rss_bytes():
    """Current resident memory, not high-water RSS (Linux host)."""
    try:
        with open('/proc/self/statm', 'r', encoding='ascii') as f:
            fields = f.read().split()
        return int(fields[1])*os.sysconf('SC_PAGE_SIZE')
    except (FileNotFoundError, ValueError, IndexError, OSError):
        return None


def atomic_json(path: Path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Checkpoints:
    """Append-only per-minute checkpoints; never accumulates per-step logs."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, 'w', encoding='utf-8', buffering=1)

    def append(self, item):
        self.file.write(json.dumps(item, allow_nan=False, separators=(',', ':')) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


@dataclass
class Counter:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    stale: int = 0
    unique_valid: int = 0
    bearing_errors: int = 0
    max_bearing_error: float = 0.0
    last_seen_sample_end: int = -1

    def track(self, observation, expected_world_bearing: float | None, max_error_deg: float):
        self.total += 1
        if observation.valid:
            self.valid += 1
            if observation.source_sample_end != self.last_seen_sample_end:
                self.unique_valid += 1
                self.last_seen_sample_end = observation.source_sample_end
                if expected_world_bearing is not None:
                    error = angle_error(observation.bearing_world_deg, expected_world_bearing)
                    self.max_bearing_error = max(self.max_bearing_error, error)
                    if error > max_error_deg:
                        self.bearing_errors += 1
        else:
            self.invalid += 1
            if observation.reason == 'stale':
                self.stale += 1
