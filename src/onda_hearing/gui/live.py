"""Wall-clock paced, continuous PCM from a GUI scene snapshot."""
from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from time import monotonic_ns, sleep

import numpy as np

from onda_hearing.streaming.core import GCCPHATSRP, Pose, StreamConfig, StreamingHearing

from .backend import SAMPLE_RATE, _beats_capability, _classify_semantics, _monitor_pcm16, _waveform

CHUNK_SAMPLES = 2048
_semantic_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onda-gui-beats")
_semantic_slot = Semaphore(1)


def live_frames(renderer, positions):
    """Stream new observations and monitor audio at about eight updates per second."""
    config = StreamConfig(sample_rate=SAMPLE_RATE, channels=len(positions))
    start_ns = monotonic_ns()
    hearing = StreamingHearing(config, start_ns, GCCPHATSRP(positions, SAMPLE_RATE))
    quaternion = (0., 0., 0., 1.)
    hearing.push_pose(Pose(start_ns, (0., 0., 0.), quaternion))
    semantic_available = _beats_capability()["available"]
    semantic = None
    pending = None
    recent_pcm = deque(maxlen=(SAMPLE_RATE * 2 + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES)
    next_semantic_sample = SAMPLE_RATE

    try:
        while True:
            start = renderer.next_sample
            end = start + CHUNK_SAMPLES
            capture_ns = start_ns + end * 1_000_000_000 // SAMPLE_RATE
            pcm = renderer.render(start, CHUNK_SAMPLES)
            sleep(max(0., (capture_ns - monotonic_ns()) / 1e9))
            observation, _ = hearing.poll(capture_ns)
            hearing.push_pose(Pose(capture_ns, (0., 0., 0.), quaternion))
            hearing.push_audio(pcm, start)

            if semantic_available:
                recent_pcm.append(pcm)
                if pending is not None and pending.done():
                    semantic = pending.result()
                    pending = None
                if end >= next_semantic_sample and pending is None and _semantic_slot.acquire(blocking=False):
                    try:
                        pending = _semantic_pool.submit(_classify_semantics, np.concatenate(recent_pcm))
                    except BaseException:
                        _semantic_slot.release()
                        raise
                    pending.add_done_callback(lambda _: _semantic_slot.release())
                    next_semantic_sample = end + SAMPLE_RATE * 2

            yield {
                "time_s": end / SAMPLE_RATE,
                "sample_rate_hz": SAMPLE_RATE,
                "spatial": {
                    "valid": observation.valid,
                    "bearing_deg": observation.bearing_local_deg if observation.valid else None,
                    "confidence": observation.confidence,
                },
                "semantic": semantic,
                "audio": {
                    "peak": float(np.max(np.abs(pcm))),
                    "waveform": _waveform(pcm[:, :1], points=96),
                    "pcm16_base64": base64.b64encode(_monitor_pcm16(pcm)).decode("ascii"),
                },
            }
    finally:
        hearing.close()
