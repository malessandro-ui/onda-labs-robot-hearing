"""Separation interface with explicit unsupported reference implementation.

Delay-and-sum beamforming is NOT certified source separation. Avoid shipping
false 'separated' streams in a production SDK.
"""
from __future__ import annotations
from onda_hearing.contracts import AudioPacket, MicrophoneArray
from .interfaces import LocalizationFrame, SeparatedAudio

class SeparationUnavailableError(NotImplementedError): pass

class UnsupportedSeparator:
    def separate(self,audio:AudioPacket,array:MicrophoneArray,observations:LocalizationFrame)->tuple[SeparatedAudio,...]:
        audio.validate_array(array)
        raise SeparationUnavailableError('Waveform separation is not implemented or validated in v0.5; integrate a qualified backend explicitly')
