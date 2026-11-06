"""Waveform-based candidate generator and provisional bearing tracker."""
from __future__ import annotations
from .core import AudioBlock, HearingObservation, MicrophoneArray
from .localization import estimate_azimuth
from .tracking import BearingTracker

class HearingPipeline:
    def __init__(self, array: MicrophoneArray, *, max_candidates: int = 1,
                 tracker: BearingTracker | None = None,
                 min_peak_relative: float = .65,
                 min_peak_prominence: float = .015):
        self.array = array
        self.max_candidates = max_candidates
        self.min_peak_relative = min_peak_relative
        self.min_peak_prominence = min_peak_prominence
        self.tracker = tracker if tracker is not None else BearingTracker()
        self._last_block_end: float | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self._last_block_end = None

    def process(self, block: AudioBlock) -> HearingObservation:
        if self._last_block_end is not None and block.t0_s < self._last_block_end - 1e-9:
            raise ValueError('Overlapping/out-of-order audio blocks require reset')
        directions = estimate_azimuth(block, self.array, max_sources=self.max_candidates,
                                      min_peak_relative=self.min_peak_relative,
                                      min_peak_prominence=self.min_peak_prominence)
        tracks = self.tracker.update(directions, block.t1_s)
        self._last_block_end = block.t1_s
        return HearingObservation(block.t1_s, directions, tracks, block.frame_id,
                                  'srp_phat_planar_conservative', 'ok' if directions else 'no_direction')
