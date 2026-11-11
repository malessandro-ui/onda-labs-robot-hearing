"""Thread-safe completion mailbox for Kit-owned USD stage-opening operations.

The Kit application thread owns stage creation and calls start/poll. A USD
completion callback may execute on a different thread; it only posts a small
immutable result into the mailbox. No Kit or USD APIs are imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Callable
import time
import uuid


@dataclass(frozen=True)
class StageOpenRequest:
    token: str
    url: str
    started_at: float


@dataclass(frozen=True)
class StageOpenResult:
    token: str
    url: str
    success: bool
    reason: str
    timed_out: bool = False


class StageOpenTracker:
    """One outstanding stage open; idempotent retries and bounded wait.

    A timeout releases Onda's UI state but does NOT cancel the underlying USD
    operation. Before retrying, the caller must query UsdContext.can_open_stage().
    Late callbacks from a prior generation are ignored by token matching.
    """

    def __init__(self, timeout_seconds: float = 180.0):
        if not 5.0 <= float(timeout_seconds) <= 3600.0:
            raise ValueError('timeout_seconds must be between 5 and 3600')
        self.timeout_seconds = float(timeout_seconds)
        self._current: StageOpenRequest | None = None
        self._mailbox: SimpleQueue = SimpleQueue()

    @property
    def current(self) -> StageOpenRequest | None:
        return self._current

    def start(self, url: str, *, now: float | None = None) -> tuple[StageOpenRequest, bool]:
        if not isinstance(url, str) or not url.strip():
            raise ValueError('stage URL must be nonempty')
        if self._current is not None:
            if self._current.url != url:
                raise ValueError('a different USD stage is already loading')
            return self._current, False
        timestamp = time.monotonic() if now is None else float(now)
        request = StageOpenRequest(uuid.uuid4().hex, url, timestamp)
        self._current = request
        return request, True

    def callback_for(self, token: str) -> Callable[[bool, str], None]:
        def callback(success: bool, reason: str):
            self._mailbox.put((token, bool(success), str(reason or '')))
        return callback

    def abandon(self, token: str) -> None:
        if self._current is not None and self._current.token == token:
            self._current = None

    def poll(self, *, now: float | None = None) -> StageOpenResult | None:
        while True:
            try:
                token, success, reason = self._mailbox.get_nowait()
            except Empty:
                break
            current = self._current
            if current is not None and token == current.token:
                self._current = None
                return StageOpenResult(token, current.url, success, reason)
        current = self._current
        if current is not None:
            timestamp = time.monotonic() if now is None else float(now)
            if timestamp - current.started_at >= self.timeout_seconds:
                self._current = None
                return StageOpenResult(current.token, current.url, False,
                     f'USD stage open exceeded {self.timeout_seconds:g}s; Kit may still be opening it', True)
        return None

    def reset(self) -> None:
        self._current = None
