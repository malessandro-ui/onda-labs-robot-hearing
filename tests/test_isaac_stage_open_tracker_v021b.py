"""Kit-independent regression for the v0.21B stage-loader recovery."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from onda_hearing.simulators.isaac_stage_open_tracker_v021b import StageOpenTracker


def test_callback_success_and_idempotent_repeated_open():
    tracker=StageOpenTracker(timeout_seconds=10)
    request, started=tracker.start('https://example.invalid/warehouse.usd',now=10)
    repeat, new=tracker.start('https://example.invalid/warehouse.usd',now=11)
    assert started and not new and request.token==repeat.token
    callback=tracker.callback_for(request.token)
    callback(True,'')
    result=tracker.poll(now=12)
    assert result is not None and result.success and result.url==request.url
    assert tracker.current is None


def test_failure_completion_and_timeout_release_state():
    tracker=StageOpenTracker(timeout_seconds=10)
    req,_=tracker.start('url-a',now=0)
    tracker.callback_for(req.token)(False,'remote asset offline')
    result=tracker.poll(now=1)
    assert not result.success and result.reason=='remote asset offline'
    req,_=tracker.start('url-b',now=2)
    assert tracker.poll(now=12).timed_out
    assert tracker.current is None


def test_stale_callback_ignored_after_timeout_and_retry():
    tracker=StageOpenTracker(timeout_seconds=10)
    old,_=tracker.start('url',now=0)
    assert tracker.poll(now=11).timed_out
    new,_=tracker.start('url',now=12)
    tracker.callback_for(old.token)(True,'late')
    assert tracker.poll(now=13) is None
    assert tracker.current.token==new.token
    tracker.callback_for(new.token)(True,'')
    assert tracker.poll(now=14).token==new.token


def test_mismatched_new_url_rejected_while_open_in_progress():
    tracker=StageOpenTracker(timeout_seconds=10)
    tracker.start('url-a',now=0)
    with pytest.raises(ValueError):tracker.start('url-b',now=1)


def test_failed_start_can_be_abandoned():
    tracker=StageOpenTracker(timeout_seconds=10)
    req,_=tracker.start('url-a',now=0)
    tracker.abandon(req.token)
    assert tracker.current is None


def _load_extension(monkeypatch):
    # A real Kit installation is not required for testing the bridge lifecycle.
    omni=types.ModuleType('omni');omni.__path__=[]
    ext=types.ModuleType('omni.ext');ext.IExt=object
    kit=types.ModuleType('omni.kit');kit.__path__=[]
    kitapp=types.ModuleType('omni.kit.app')
    ui=types.ModuleType('omni.ui')
    usd=types.ModuleType('omni.usd')
    omni.ext=ext;omni.kit=kit;omni.ui=ui;omni.usd=usd;kit.app=kitapp
    for k,v in [('omni',omni),('omni.ext',ext),('omni.kit',kit),
                ('omni.kit.app',kitapp),('omni.ui',ui),('omni.usd',usd)]:
        monkeypatch.setitem(sys.modules,k,v)
    # Import the extension module under an isolated name.
    file=Path(__file__).resolve().parents[1]/'exts'/'omni.onda.hearing'/'omni'/'onda'/'hearing'/'extension.py'
    if not file.is_file(): pytest.skip('native extension source not available in test tree')
    spec=importlib.util.spec_from_file_location('_onda_test_extension',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.OndaHearingExtension


def test_native_extension_callback_lifecycle(monkeypatch):
    Extension=_load_extension(monkeypatch)
    from types import SimpleNamespace
    isaacsim=types.ModuleType('isaacsim');isaacsim.__path__=[]
    storage=types.ModuleType('isaacsim.storage');storage.__path__=[]
    native=types.ModuleType('isaacsim.storage.native')
    native.get_assets_root_path=lambda: 'https://example.invalid/Assets/Isaac/5.1'
    isaacsim.storage=storage;storage.native=native
    monkeypatch.setitem(sys.modules,'isaacsim',isaacsim)
    monkeypatch.setitem(sys.modules,'isaacsim.storage',storage)
    monkeypatch.setitem(sys.modules,'isaacsim.storage.native',native)

    class FakeState:
        def __init__(self):self.data={}
        def set_status(self,**kw):self.data.update(kw)
    class FakeUSD:
        def __init__(self):self.stage=None;self.calls=0;self.fn=None
        def get_stage(self):return self.stage
        def can_open_stage(self):return True
        def open_stage_with_callback(self,url,fn):
            self.calls+=1;self.fn=fn;self.url=url;return True
    fake=FakeUSD();state=FakeState()
    obj=Extension.__new__(Extension)
    obj._stage_opens=StageOpenTracker(timeout_seconds=10)
    obj._usd=fake;obj._state=state
    sync=[]
    def sync_stage(force=False):sync.append(force);state.set_status(scene_loaded=True)
    obj._sync_stage=sync_stage

    result=obj._open_warehouse({})
    assert result['state']=='loading' and fake.calls==1
    assert obj._open_warehouse({})['state']=='loading' and fake.calls==1
    assert state.data['loading_stage'] is True
    fake.stage=SimpleNamespace(GetRootLayer=lambda:SimpleNamespace(identifier=fake.url))
    fake.fn(True,'')
    obj._poll_stage_open()
    assert state.data['load_phase']=='loaded'
    assert state.data['loading_stage'] is False
    assert state.data['scene_loaded'] and sync==[True]


def test_native_extension_reports_callback_failure(monkeypatch):
    Extension=_load_extension(monkeypatch)
    from types import SimpleNamespace
    obj=Extension.__new__(Extension)
    obj._stage_opens=StageOpenTracker(timeout_seconds=10)
    obj._usd=SimpleNamespace(get_stage=lambda:None)
    class State:
        status={}
        def set_status(self,**kw):self.status.update(kw)
    obj._state=State()
    request,_=obj._stage_opens.start('url',now=10)
    obj._stage_opens.callback_for(request.token)(False,'Asset not found')
    obj._poll_stage_open()
    assert obj._state.status['load_phase']=='failed'
    assert not obj._state.status['loading_stage']
    assert 'Asset not found' in obj._state.status['last_error']
