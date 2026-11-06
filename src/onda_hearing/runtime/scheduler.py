"""Integer timeline scheduling; no cumulative floating-point step drift."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SampleRange:
    first: int
    stop: int
    start_ns: int
    end_ns: int
    @property
    def count(self)->int:return self.stop-self.first

class SampleScheduler:
    def __init__(self,sample_rate_hz:int,epoch_ns:int):
        if type(sample_rate_hz) is not int or not 8000<=sample_rate_hz<=384000:
            raise ValueError('invalid sample rate')
        if type(epoch_ns) is not int or epoch_ns<0:
            raise ValueError('invalid epoch')
        self.sample_rate_hz=sample_rate_hz;self.epoch_ns=epoch_ns
        self.last_ns=epoch_ns;self.emitted=0

    def preview(self,next_ns:int)->SampleRange:
        if type(next_ns) is not int or next_ns<=self.last_ns:
            raise ValueError('simulation time must strictly increase; reset on rewind')
        count=((next_ns-self.epoch_ns)*self.sample_rate_hz)//1_000_000_000
        a=self.emitted;b=count
        start=self.epoch_ns+round(a*1_000_000_000/self.sample_rate_hz)
        end=self.epoch_ns+round(b*1_000_000_000/self.sample_rate_hz)
        return SampleRange(a,b,start,end)

    def commit(self,next_ns:int,portion:SampleRange)->None:
        if portion.first!=self.emitted or portion.stop<portion.first:
            raise ValueError('cannot commit stale sample range')
        self.last_ns=next_ns;self.emitted=portion.stop
