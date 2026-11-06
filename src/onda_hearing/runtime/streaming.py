"""Bounded, sample-contiguous audio and pose history for motion-aware inference.

Audio stays in its named clock domain. The fixed sensor geometry is always
array-local, while stored poses describe the local-to-world rigid transform.
No simulator source location enters an inference window.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from onda_hearing.contracts import AudioPacket, MicrophoneArray
from onda_hearing.physics.propagation import PoseTrajectory
from .scene import SceneSnapshot, rotation_from_xyzw


@dataclass(frozen=True, slots=True)
class PerceptionWindow:
    audio: AudioPacket
    poses: PoseTrajectory
    pose_clock_id: str


class HearingWindowBuffer:
    """Bounded packet buffer with exact sample-based window assembly.

    A pose sampled after each physics step can be supplied independently from
    audio emission; it is retained only while needed for localization.
    Zero-length simulation steps must not create AudioPackets.
    """
    def __init__(self, array: MicrophoneArray, *, sample_rate_hz: int,
                 clock_id: str = 'simulation', window_samples: int = 4096,
                 capacity_samples: int = 8192):
        if not isinstance(array, MicrophoneArray):
            raise TypeError('array must be MicrophoneArray')
        if type(sample_rate_hz) is not int or sample_rate_hz < 8000:
            raise ValueError('invalid sample rate')
        if type(window_samples) is not int or type(capacity_samples) is not int or not 320 <= window_samples <= capacity_samples <= sample_rate_hz * 10:
            raise ValueError('invalid bounded window/capacity')
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError('clock_id required')
        self.array = array
        self.fs = sample_rate_hz
        self.clock_id = clock_id
        self.window_samples = window_samples
        self.capacity_samples = capacity_samples
        self.reset()

    def reset(self):
        self._segments: deque[np.ndarray] = deque()
        self._count = 0
        self._last_end_ns: int | None = None
        self._last_sequence: int | None = None
        self._poses: deque[SceneSnapshot] = deque()

    def push_pose(self, snapshot: SceneSnapshot):
        if not isinstance(snapshot, SceneSnapshot):
            raise TypeError('snapshot must be SceneSnapshot')
        if self._poses and snapshot.time_ns <= self._poses[-1].time_ns:
            raise ValueError('pose timestamps must strictly increase')
        self._poses.append(snapshot)
        self._trim_poses()

    def push_audio(self, packet: AudioPacket):
        packet.validate_array(self.array)
        if packet.sample_rate_hz != self.fs or packet.capture.clock_id != self.clock_id:
            raise ValueError('audio rate/clock mismatch')
        if packet.discontinuity:
            raise ValueError('audio discontinuity requires explicit reset')
        if self._last_end_ns is not None:
            if abs(packet.capture.start_ns - self._last_end_ns) > 1:
                raise ValueError('audio capture intervals are not contiguous')
            if packet.sequence != self._last_sequence + 1:
                raise ValueError('audio packet sequence gap')
        chunk = np.array(packet.pcm, dtype=np.float32, copy=True)
        self._segments.append(chunk)
        self._count += len(chunk)
        self._last_end_ns = packet.capture.end_ns
        self._last_sequence = packet.sequence
        while self._count > self.capacity_samples:
            excess = self._count - self.capacity_samples
            first = self._segments.popleft()
            if len(first) > excess:
                self._segments.appendleft(first[excess:])
                self._count -= excess
                break
            self._count -= len(first)
        self._trim_poses()

    def _trim_poses(self):
        if self._last_end_ns is None or len(self._poses) < 3:
            return
        # Preserve the pose immediately preceding the earliest retained audio.
        start_ns = self._last_end_ns - math.ceil(self._count * 1e9 / self.fs)
        while len(self._poses) > 2 and self._poses[1].time_ns < start_ns:
            self._poses.popleft()

    @property
    def ready(self) -> bool:
        return self._count >= self.window_samples and len(self._poses) >= 2

    def window(self) -> PerceptionWindow:
        if not self.ready:
            raise ValueError('not enough audio and pose history')
        remaining = self.window_samples
        pieces = []
        for part in reversed(self._segments):
            if remaining <= 0:
                break
            take = min(len(part), remaining)
            pieces.append(part[-take:])
            remaining -= take
        samples = np.concatenate(list(reversed(pieces)))
        if len(samples) != self.window_samples:
            raise RuntimeError('internal window length error')
        assert self._last_end_ns is not None
        start_ns = self._last_end_ns - round(self.window_samples * 1e9 / self.fs)
        relevant = list(self._poses)
        if relevant[0].time_ns > start_ns or relevant[-1].time_ns < self._last_end_ns:
            raise ValueError('pose history does not bracket audio window')
        # Keep the preceding pose and every subsequent sample up to window end.
        first = 0
        for i in range(len(relevant) - 1):
            if relevant[i + 1].time_ns >= start_ns:
                first = i
                break
        relevant = relevant[first:]
        traj = PoseTrajectory(
            np.array([p.time_ns / 1e9 for p in relevant]),
            np.array([p.array_position_m for p in relevant]),
            np.array([rotation_from_xyzw(p.array_orientation_xyzw) for p in relevant]),
        )
        audio = AudioPacket.from_pcm(samples, self.fs,
                                     start_ns=start_ns, clock_id=self.clock_id,
                                     array=self.array, sequence=0)
        return PerceptionWindow(audio, traj, self.clock_id)
