"""Capture time and clock-domain semantics. Never silently equate clocks."""
from __future__ import annotations
from dataclasses import dataclass
from .geometry import frame_id

@dataclass(frozen=True,slots=True)
class CaptureInterval:
    clock_id: str
    start_ns: int
    end_ns: int
    def __post_init__(self):
        frame_id(self.clock_id)
        if type(self.start_ns) is not int or type(self.end_ns) is not int or self.start_ns<0 or self.end_ns<=self.start_ns:
            raise ValueError('capture interval requires positive duration and nonnegative integer nanosecond timestamps')
    @property
    def duration_s(self)->float:return (self.end_ns-self.start_ns)/1e9

@dataclass(frozen=True,slots=True)
class ClockMapping:
    source_clock: str
    target_clock: str
    offset_ns: int
    uncertainty_ns: int
    def __post_init__(self):
        frame_id(self.source_clock);frame_id(self.target_clock)
        if type(self.offset_ns) is not int or type(self.uncertainty_ns) is not int or self.uncertainty_ns<0:
            raise ValueError('invalid clock mapping')
