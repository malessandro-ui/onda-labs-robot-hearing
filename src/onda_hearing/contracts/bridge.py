"""Explicit transition from legacy simulated/recorded audio into v1 contracts.

Does not assert provenance; caller must supply geometry identity and a clock.
"""
from __future__ import annotations
from .audio import AudioPacket
from .microphones import MicrophoneArray
from ..core import AudioBlock


def from_legacy_block(block: AudioBlock, array: MicrophoneArray, *, clock_id: str,
                      sequence: int = 0, discontinuity: bool = False) -> AudioPacket:
    if not isinstance(block,AudioBlock):
        raise TypeError('expected legacy AudioBlock')
    if block.frame_id != array.frame_id or block.pcm.shape[1] != array.channels:
        raise ValueError('legacy audio does not match microphone array')
    start_ns = round(block.t0_s * 1e9)
    if start_ns < 0:
        raise ValueError('capture timestamps must be nonnegative')
    return AudioPacket.from_pcm(block.pcm, block.sample_rate_hz, start_ns=start_ns,
                                clock_id=clock_id, array=array, sequence=sequence,
                                discontinuity=discontinuity)
