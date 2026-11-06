"""Thin simulator adapter contract. No acoustic or perception computation."""
from __future__ import annotations
from typing import Protocol,runtime_checkable
from onda_hearing.runtime import SceneSnapshot

@runtime_checkable
class SceneAdapter(Protocol):
    def snapshot(self, time_ns:int)->SceneSnapshot: ...
