"""Bearing-only track lifecycle; does not claim waveform separation or identities.

Track IDs follow measured bearings in one fixed frame. Moving arrays require
upstream timestamped pose transforms into a stable frame.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .core import Direction, SourceTrack


def wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


@dataclass
class _State:
    id: int
    az: float
    score: float
    first: float
    last: float
    hits: int
    missed: int
    consecutive: int


class BearingTracker:
    def __init__(self, *, gate_deg: float = 25., confirm_hits: int = 3,
                 max_missed: int = 3, smoothing: float = .35,
                 min_observation_score: float = 0., max_idle_s: float = 2.):
        if (not 0 < gate_deg < 180 or confirm_hits < 1 or max_missed < 0 or
            not 0 < smoothing <= 1 or not math.isfinite(min_observation_score) or
            min_observation_score < 0 or not math.isfinite(max_idle_s) or max_idle_s <= 0):
            raise ValueError('Invalid tracker configuration')
        self.gate = math.radians(gate_deg)
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self.smoothing = smoothing
        self.min_observation_score = min_observation_score
        self.max_idle_s = max_idle_s
        self.reset()

    def reset(self):
        self._states: list[_State] = []
        self._next = 1
        self._last_t: float | None = None
        self._frame: str | None = None

    def update(self, observations: tuple[Direction, ...], timestamp_s: float) -> tuple[SourceTrack, ...]:
        if not math.isfinite(timestamp_s) or (self._last_t is not None and timestamp_s <= self._last_t):
            raise ValueError('Tracker requires strictly increasing timestamps')
        if any(not math.isfinite(o.azimuth_rad) or not math.isfinite(o.score) or o.score < 0 or not o.frame_id for o in observations):
            raise ValueError('Invalid direction measurement')
        if observations and len({o.frame_id for o in observations}) != 1:
            raise ValueError('Mixed measurement frames')
        frame = observations[0].frame_id if observations else self._frame
        if self._frame is not None and frame != self._frame:
            raise ValueError('Frame changed: provide world-frame bearings or reset')
        self._frame = frame
        self._last_t = timestamp_s
        self._states = [s for s in self._states if timestamp_s - s.last <= self.max_idle_s]
        valid = [(i,o) for i,o in enumerate(observations) if o.score >= self.min_observation_score]
        # Hungarian association is not needed for this provisional baseline; global
        # greedy is deterministic, one-to-one, and explicitly non-optimal for crossings.
        pairs = sorted((abs(wrap(s.az - o.azimuth_rad)), si, oi)
                       for si,s in enumerate(self._states) for oi,o in valid)
        used_s: set[int] = set()
        used_o: set[int] = set()
        for dist, si, oi in pairs:
            if dist >= self.gate or si in used_s or oi in used_o:
                continue
            s, o = self._states[si], observations[oi]
            s.az = wrap(s.az + self.smoothing * wrap(o.azimuth_rad - s.az))
            s.score = o.score
            s.hits += 1
            s.consecutive = s.consecutive + 1 if s.missed == 0 else 1
            s.missed = 0
            s.last = timestamp_s
            used_s.add(si); used_o.add(oi)
        for si,s in enumerate(self._states):
            if si not in used_s:
                s.missed += 1
                s.consecutive = 0
        self._states = [s for s in self._states if s.missed <= self.max_missed]
        for oi,o in valid:
            if oi not in used_o:
                self._states.append(_State(self._next, wrap(o.azimuth_rad), o.score, timestamp_s, timestamp_s, 1, 0, 1))
                self._next += 1
        return tuple(SourceTrack(s.id, s.az, s.score, self._frame or '', s.first,
                                 s.last, s.consecutive >= self.confirm_hits and s.missed == 0,
                                 s.missed) for s in sorted(self._states, key=lambda s:s.id))
