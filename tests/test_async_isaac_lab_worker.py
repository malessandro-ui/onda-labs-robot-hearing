"""Deterministic unit tests; no Kit/GPU required."""
import importlib.util
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

# Import add-on without requiring the complete NVIDIA or Onda installation.
# In an installed SDK, actual manager/reader implementations are used instead.
module_name = 'onda_hearing.simulators.isaac_lab_runtime'
reader_name = 'onda_hearing.simulators.isaac_lab_wrapper'
if module_name not in sys.modules:
    try:
        from onda_hearing.simulators.isaac_lab_runtime import IsaacLabHearingManager as RealManager
        from onda_hearing.simulators.isaac_lab_wrapper import IsaacLabSceneReader as RealReader
    except ImportError:
        real_runtime = types.ModuleType(module_name)
        real_wrapper = types.ModuleType(reader_name)
        real_runtime.IsaacLabHearingManager = type('IsaacLabHearingManager', (), {})
        real_wrapper.IsaacLabSceneReader = type('IsaacLabSceneReader', (), {})
        sys.modules[module_name] = real_runtime
        sys.modules[reader_name] = real_wrapper

from onda_hearing.simulators.async_isaac_lab import AsyncHearingWorker, AsyncHearingConfig
from onda_hearing.simulators.isaac_lab_runtime import IsaacLabHearingManager


def poses(n=2):
    return (np.zeros((n, 3)), np.tile([1., 0., 0., 0.], (n, 1)),
            {'sound': np.tile([1., 0., 0.], (n, 1))})


class FakeManager(IsaacLabHearingManager):
    def __init__(self, *, delay=0.0, boom=False):
        self.num_envs = 2
        self.config = types.SimpleNamespace(step_dt_ns=16_666_667)
        self.features = np.zeros((2, 5), np.float32)
        self.ticks = [0, 0]
        self.delay = delay
        self.boom = boom
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.unblock.set()

    def reset(self, env_ids=None):
        for i in range(2) if env_ids is None else env_ids:
            self.ticks[i] = 0
            self.features[i] = 0

    def seed(self, *unused, env_ids=None):
        return None

    def advance(self, *unused, active_env_ids=None):
        self.entered.set()
        self.unblock.wait(timeout=4)
        if self.boom:
            raise RuntimeError('injected failure')
        if self.delay:
            time.sleep(self.delay)
        for i in range(2) if active_env_ids is None else active_env_ids:
            self.ticks[i] += 1
            self.features[i] = np.array([0.6, 0.8, 0.5, 1., 0.], dtype=np.float32)

    def read(self):
        return self.features.copy()


def until(condition, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(.005)
    return False


def test_clean_seed_and_output_is_finite():
    worker = AsyncHearingWorker(FakeManager())
    try:
        worker.submit(poses(), reset_all=True)
        assert until(lambda: worker.read()[1]['processed_frames'] >= 1)
        worker.submit(poses())
        assert until(lambda: worker.read()[1]['processed_frames'] >= 2)
        hearing, meta = worker.read()
        assert hearing.shape == (2, 5)
        assert np.isfinite(hearing).all()
        assert hearing[:, 3].tolist() == [1.0, 1.0]
        assert meta['dropped_frames'] == 0
    finally:
        assert worker.close()


def test_partial_done_does_not_reset_other_environment():
    manager = FakeManager()
    worker = AsyncHearingWorker(manager)
    try:
        worker.submit(poses(), reset_all=True)
        assert until(lambda: worker.read()[1]['processed_frames'] == 1)
        worker.submit(poses())
        assert until(lambda: worker.read()[1]['processed_frames'] == 2)
        worker.submit(poses(), done_env_ids=[0])
        assert until(lambda: worker.read()[1]['processed_frames'] == 3)
        assert manager.ticks == [0, 2]
        hearing, _ = worker.read()
        assert hearing[0, 3] == 0. and hearing[1, 3] == 1.
    finally:
        assert worker.close()


def test_overflow_invalidates_and_resynchronizes_without_old_output():
    manager = FakeManager()
    worker = AsyncHearingWorker(manager, AsyncHearingConfig(queue_capacity=1))
    try:
        worker.submit(poses(), reset_all=True)
        assert until(lambda: worker.read()[1]['processed_frames'] == 1)
        manager.unblock.clear()
        worker.submit(poses())
        assert manager.entered.wait(.5)
        # Existing event may have been set from prior advance; allow worker to enter.
        time.sleep(.02)
        worker.submit(poses())
        assert worker.submit(poses()) is False
        hearing, meta = worker.read()
        assert not hearing.any()
        assert meta['dropped_frames'] >= 1
        assert meta['resynchronizations'] >= 2
        manager.unblock.set()
        assert until(lambda: worker.read()[1]['processed_frames'] >= 2)
    finally:
        manager.unblock.set()
        assert worker.close()


def test_worker_exception_propagates():
    manager = FakeManager(boom=True)
    worker = AsyncHearingWorker(manager)
    try:
        worker.submit(poses(), reset_all=True)
        assert until(lambda: worker.read()[1]['processed_frames'] == 1)
        worker.submit(poses())
        assert until(lambda: worker._last_error is not None)
        with pytest.raises(RuntimeError, match='worker failed'):
            worker.read()
    finally:
        assert worker.close()


def test_stale_hearing_is_invalidated():
    manager = FakeManager()
    worker = AsyncHearingWorker(manager, AsyncHearingConfig(max_staleness_ms=16.7))
    try:
        worker.submit(poses(), reset_all=True)
        assert until(lambda: worker.read()[1]['processed_frames'] == 1)
        worker.submit(poses())
        assert until(lambda: worker.read()[1]['processed_frames'] == 2)
        manager.unblock.clear()
        for _ in range(2):
            worker.submit(poses())
        hearing, meta = worker.read()
        assert meta['stale']
        assert not hearing.any()
    finally:
        manager.unblock.set()
        assert worker.close()
