"""Onda v0.21 HTTP/GUI bridge. No Kit or USD access in HTTP threads.

Kit application thread owns stage, emitter registry and acoustic geometry.
GET endpoints expose only immutable, lock-protected snapshots. POST/DELETE
are bounded commands whose handlers run on the Kit update thread.

Security: loopback only, explicit X-Onda-Client header for mutations, no CORS,
no arbitrary files/paths served or arbitrary USD commands over HTTP.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import RLock, Thread
import time
from typing import Any, Callable
import json
import re
import os
import uuid
import tempfile
import wave
from starlette.requests import Request


class BridgeError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _Command:
    name: str
    payload: dict
    response: Queue = field(default_factory=lambda: Queue(maxsize=1))


class IsaacBridgeState:
    """Sole cross-thread communication object between Kit and localhost API."""

    def __init__(self, *, command_capacity: int = 64):
        self._lock = RLock()
        self._commands: Queue[_Command] = Queue(maxsize=command_capacity)
        self._revision = 0
        self._scene: dict = {"loaded": False, "source": None, "acoustics": None}
        self._selection: dict = {"prim_path": None, "type": None, "position_m": None}
        self._emitters: list[dict] = []
        self._microphones: dict = {"count":4, "pose_available":False}
        self._results: dict = {"last_render": None}
        self._status: dict = {"connected_to_isaac": False, "acoustic_scene_built": False,
                              "live_pcm_streaming": False, "last_error": None,
                              "mode": "usd_static_snapshot_v021"}
        self._events: deque = deque(maxlen=256)
        self._closed = False

    def _publish(self, kind: str, data: dict):
        self._revision += 1
        self._events.append({"revision": self._revision, "event": kind, "data": deepcopy(data)})

    def snapshot(self, kind: str) -> dict:
        with self._lock:
            if kind not in ("scene", "selection", "emitters", "microphones", "results", "status"):
                raise KeyError(kind)
            value = getattr(self, "_"+kind)
            return {"revision": self._revision, "data": deepcopy(value)}

    def events(self, since: int = 0) -> dict:
        with self._lock:
            if since < 0: raise BridgeError("since must be >= 0")
            oldest = self._events[0]["revision"] if self._events else self._revision
            return {"revision": self._revision, "truncated": since < oldest - 1,
                    "events": deepcopy([e for e in self._events if e["revision"] > since])}

    def set_scene(self, data: dict):
        with self._lock:
            self._scene = deepcopy(data)
            self._publish("scene", self._scene)

    def set_selection(self, data: dict):
        with self._lock:
            if data == self._selection: return
            self._selection = deepcopy(data)
            self._publish("selection", self._selection)

    def set_emitters(self, records: list[dict]):
        with self._lock:
            self._emitters = deepcopy(records)
            self._publish("emitters", {"count": len(self._emitters)})

    def set_microphones(self, data:dict):
        with self._lock:
            self._microphones = deepcopy(data)
            self._publish("microphones", self._microphones)

    def set_status(self, **values):
        with self._lock:
            self._status.update(deepcopy(values))
            self._publish("status", self._status)

    def set_results(self, data: dict):
        with self._lock:
            self._results.update(deepcopy(data))
            self._publish("results", self._results)

    def submit(self, name: str, payload: dict, *, timeout: float = 15.0) -> Any:
        with self._lock:
            if self._closed:
                raise BridgeError("Kit bridge is shutting down", 503)
            if not self._status["connected_to_isaac"]:
                raise BridgeError("Isaac Sim extension is not connected", 503)
        request = _Command(name=name, payload=deepcopy(payload))
        try:
            self._commands.put_nowait(request)
        except Full as exc:
            raise BridgeError("Isaac command queue full", 503) from exc
        try:
            ok, value = request.response.get(timeout=timeout)
        except Empty as exc:
            raise BridgeError("Isaac did not process the command in time", 504) from exc
        if ok: return value
        message, code = value
        raise BridgeError(message, code)

    def drain(self, handler: Callable[[str, dict], Any], *, max_count: int = 8) -> int:
        """ONLY call on the Kit main/update thread."""
        processed = 0
        for _ in range(max_count):
            try:
                request = self._commands.get_nowait()
            except Empty:
                break
            try:
                answer = handler(request.name, request.payload)
                request.response.put_nowait((True, answer))
            except Exception as exc:
                # Never expose a traceback or secret paths to the HTTP client.
                code = 409 if type(exc).__name__ == "RegistryConflict" else (
                    400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500
                )
                request.response.put_nowait((False, (str(exc)[:500], code)))
                self.set_status(last_error=f"{type(exc).__name__}: {str(exc)[:300]}")
            processed += 1
        return processed

    def close(self):
        with self._lock:
            self._closed = True
            self._status["connected_to_isaac"] = False
            self._publish("status", self._status)
        while True:
            try: req = self._commands.get_nowait()
            except Empty: break
            try: req.response.put_nowait((False, ("Kit bridge stopped", 503)))
            except Full: pass


def create_app(state: IsaacBridgeState, *, web_root: Path | None = None,
               render_root: Path | None = None, audio_root: Path | None = None,
               sample_rate: int = 16000):
    """Create a normal FastAPI app; may be tested without Isaac Sim installed."""
    try:
        from fastapi import Body, FastAPI, Header, HTTPException, Query
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for optional GUI bridge: pip install fastapi uvicorn") from exc

    app = FastAPI(title="Onda Isaac Scene Bridge", version="0.21.0",
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/isaac/scene")
    def get_scene(): return state.snapshot("scene")

    @app.get("/api/isaac/microphones")
    def get_microphones(): return state.snapshot("microphones")

    @app.get("/api/isaac/selection")
    def get_selection(): return state.snapshot("selection")

    @app.get("/api/isaac/emitters")
    def get_emitters(): return state.snapshot("emitters")

    @app.get("/api/isaac/status")
    def get_status(): return state.snapshot("status")

    @app.get("/api/isaac/results")
    def get_results(): return state.snapshot("results")

    @app.get("/api/isaac/events")
    def get_events(since: int = Query(default=0, ge=0)):
        return state.events(since)

    def mutate(name: str, payload: dict, header: str | None):
        if header != "OndaGUI":
            raise HTTPException(status_code=403, detail="X-Onda-Client: OndaGUI required")
        try: return state.submit(name, payload, timeout=120.0 if name == "build_scene" else 15.0)
        except BridgeError as exc: raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/isaac/emitters", status_code=201)
    def post_emitter(payload: dict = Body(...), x_onda_client: str | None = Header(default=None)):
        return mutate("add_emitter", payload, x_onda_client)

    @app.delete("/api/isaac/emitters/{emitter_id}")
    def delete_emitter(emitter_id: str, x_onda_client: str | None = Header(default=None)):
        return mutate("remove_emitter", {"emitter_id": emitter_id}, x_onda_client)

    @app.patch("/api/isaac/emitters/{emitter_id}")
    def patch_emitter(emitter_id: str, payload: dict = Body(...),
                      x_onda_client: str | None = Header(default=None)):
        return mutate("edit_emitter", {"emitter_id":emitter_id,"changes":payload},x_onda_client)

    @app.get("/api/isaac/objects")
    def list_scene_objects(q: str = Query(default="",max_length=256),
                           category: str = Query(default="",max_length=80),
                           offset: int = Query(default=0,ge=0,le=100000),
                           limit: int = Query(default=40,ge=1,le=100)):
        try:
            return state.submit("list_objects", {"q":q,"category":category,
                    "offset":offset,"limit":limit},timeout=30.)
        except BridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/isaac/open-warehouse",status_code=202)
    def open_warehouse(payload: dict = Body(default={}),
                       x_onda_client: str | None = Header(default=None)):
        return mutate("open_warehouse",payload,x_onda_client)

    @app.post("/api/isaac/selection")
    def select_object(payload: dict = Body(...),x_onda_client: str | None = Header(default=None)):
        return mutate("select_prim",payload,x_onda_client)

    @app.put("/api/isaac/microphones")
    def update_microphones(payload: dict = Body(...),x_onda_client: str | None = Header(default=None)):
        return mutate("set_microphones",payload,x_onda_client)

    @app.post("/api/isaac/microphones/marker")
    def ensure_microphone_marker(payload: dict = Body(default={}),
                                 x_onda_client: str | None = Header(default=None)):
        return mutate("ensure_microphone_marker",payload,x_onda_client)

    if audio_root is not None:
        audioroot = Path(audio_root).resolve(strict=True)
        @app.get("/api/isaac/audio-assets")
        def list_audio_assets():
            # Only inside Onda's configured audio directory, never arbitrary disk paths.
            items=[]
            for entry in sorted(audioroot.rglob('*.wav')):
                if len(items)>=200: break
                try:
                    resolved=entry.resolve(strict=True)
                    if audioroot not in resolved.parents or resolved.stat().st_size>64*1024*1024: continue
                    with wave.open(str(resolved),'rb') as wf:
                        if wf.getframerate()!=sample_rate or wf.getnchannels()!=1 or wf.getcomptype()!='NONE': continue
                        items.append({"asset":str(entry.relative_to(audioroot)),
                                      "duration_s":round(wf.getnframes()/wf.getframerate(),3)})
                except (OSError,ValueError,wave.Error):continue
            return {"data":items,"limit":200,"sample_rate":sample_rate}

        @app.post("/api/isaac/audio-assets",status_code=201)
        async def upload_audio_asset(request: Request,
              x_onda_client: str | None = Header(default=None)):
            if x_onda_client!="OndaGUI":
                raise HTTPException(status_code=403,detail="X-Onda-Client: OndaGUI required")
            # Browser upload is streamed to a bounded, private temporary file;
            # validation occurs before the file can become an emitter asset.
            fd,tmp=tempfile.mkstemp(prefix=".onda_incoming_",suffix=".tmp",dir=str(audioroot))
            bytes_written=0
            try:
                with os.fdopen(fd,"wb") as out:
                    async for chunk in request.stream():
                        bytes_written+=len(chunk)
                        if bytes_written>64*1024*1024:
                            raise HTTPException(status_code=413,detail="WAV upload exceeds 64 MiB")
                        out.write(chunk)
                    out.flush();os.fsync(out.fileno())
                if bytes_written<44:
                    raise HTTPException(status_code=400,detail="Invalid or empty WAV")
                from onda_hearing.simulators.isaac_wav_import_v021b import import_wav_to_pcm16
                asset="onda_upload_"+uuid.uuid4().hex+".wav"
                dest=audioroot/asset
                try:
                    metrics=import_wav_to_pcm16(tmp,dest,sample_rate=sample_rate)
                except (ValueError,RuntimeError,OSError) as exc:
                    raise HTTPException(status_code=400,detail=f"Invalid WAV: {exc}") from exc
                return {"asset":asset,"size_bytes":bytes_written,**metrics}
            finally:
                if os.path.exists(tmp):os.unlink(tmp)

    @app.post("/api/isaac/scene/build")
    def build_scene(payload: dict = Body(default={}), x_onda_client: str | None = Header(default=None)):
        return mutate("build_scene", payload, x_onda_client)

    @app.post("/api/isaac/render", status_code=202)
    def start_render(payload: dict = Body(default={}), x_onda_client: str | None = Header(default=None)):
        return mutate("render_snapshot", payload, x_onda_client)

    if render_root is not None:
        render_dir = render_root.resolve(strict=False)
        @app.get("/api/isaac/renders/{job_id}.wav")
        def get_rendered_wav(job_id: str):
            if not re.fullmatch(r"[0-9a-f]{32}", job_id):
                raise HTTPException(status_code=404, detail="unknown render")
            file = render_dir / (job_id + ".wav")
            if not file.is_file():
                raise HTTPException(status_code=404, detail="render is not available")
            return FileResponse(str(file), media_type="audio/wav", filename="onda-4ch-"+job_id+".wav")

    if web_root is not None:
        index = (web_root / "index.html").resolve(strict=False)
        if index.is_file():
            @app.get("/", include_in_schema=False)
            def gui(): return FileResponse(str(index), media_type="text/html")

    return app


class IsaacBridgeServer:
    """Optional HTTP server; only loopback is permitted, no network exposure."""
    def __init__(self, state: IsaacBridgeState, *, port: int = 8765,
                 web_root: Path | None = None, render_root: Path | None = None,
                 audio_root: Path | None = None):
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("port must be in 1024..65535")
        self.state = state
        self.port = port
        self.web_root = web_root
        self.render_root = render_root
        self.audio_root = audio_root
        self.thread = None
        self.server = None

    def start(self):
        import uvicorn
        if self.thread is not None:
            raise RuntimeError("already started")
        app = create_app(self.state, web_root=self.web_root,
                         render_root=self.render_root, audio_root=self.audio_root)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = Thread(target=self.server.run, name="onda-isaac-loopback-api", daemon=True)
        self.thread.start()
        # Detect startup failure rather than claiming an API is listening.
        for _ in range(100):
            if self.server.started:
                return
            if not self.thread.is_alive():
                raise RuntimeError(f"Onda API failed to bind 127.0.0.1:{self.port}")
            time.sleep(0.025)
        raise RuntimeError(f"Onda API did not start on 127.0.0.1:{self.port}")

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
