"""v0.21B persistent N-channel microphone rigs, independent of Isaac imports.

All USD reads/writes are performed by the host extension on Kit's application
thread. File persistence is schema-checked, scene-keyed, and atomic. A rig can
be free in world coordinates, attached to a USD object, or attached to a session-
layer marker prim managed by the extension. The rendered microphone coordinates
are calculated from a captured world pose, not inferred from visual bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import math
import os
import tempfile
from typing import Callable

import numpy as np
from onda_hearing.acoustic_scene.types import Pose3, MicArray, Binding, compose_pose, rotate, quat4, vector3
from onda_hearing.simulators.isaac_emitter_registry import validate_prim_path

MARKER_PATH = "/OndaMicrophoneRig"


def _xyz(value, name, *, max_abs=100000.):
    try:
        v = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be XYZ floats") from exc
    if len(v) != 3 or not all(math.isfinite(x) and abs(x) <= max_abs for x in v):
        raise ValueError(f"{name} must be finite three coordinates in meters")
    return v


def _quaternion(value):
    try:
        result = tuple(float(v) for v in value)
        return tuple(float(v) for v in quat4(result))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("orientation_xyzw must be a unit XYZW quaternion") from exc


PRESETS = {
    "triangle3": ((0.08,0,0),(-0.04,0.069282,0),(-0.04,-0.069282,0)),
    "square4": ((0.08,0,0),(0,0.08,0),(-0.08,0,0),(0,-0.08,0)),
    "hexagon6": tuple((0.08*math.cos(2*math.pi*i/6),0.08*math.sin(2*math.pi*i/6),0.) for i in range(6)),
    "circle8": tuple((0.08*math.cos(2*math.pi*i/8),0.08*math.sin(2*math.pi*i/8),0.) for i in range(8)),
}


@dataclass(frozen=True)
class MicrophoneRig:
    rig_id: str = "onda_array"
    # Center in meters in world frame if anchor is null; LOCAL offset if anchored.
    center_m: tuple[float,float,float] = (-1.,3.6,1.5)
    # Orientation in world frame if free, relative to anchor frame otherwise.
    orientation_xyzw: tuple[float,float,float,float] = (0.,0.,0.,1.)
    local_positions_m: tuple[tuple[float,float,float], ...] = PRESETS["square4"]
    anchor_prim_path: str | None = None
    # Only used to recreate session-layer visual marker after reopening a stage.
    marker_world_position_m: tuple[float,float,float] = (-1.,3.6,1.5)
    marker_world_orientation_xyzw: tuple[float,float,float,float] = (0.,0.,0.,1.)

    def __post_init__(self):
        if self.rig_id != "onda_array":
            raise ValueError("only the single primary rig 'onda_array' is supported")
        object.__setattr__(self, "center_m", _xyz(self.center_m,"center_m"))
        object.__setattr__(self, "orientation_xyzw", _quaternion(self.orientation_xyzw))
        positions = tuple(_xyz(v, "microphone offset", max_abs=10.) for v in self.local_positions_m)
        if not 3 <= len(positions) <= 16:
            raise ValueError("v0.21B supports 3..16 microphones")
        if len({tuple(round(v, 8) for v in p) for p in positions}) != len(positions):
            raise ValueError("microphone coordinates must be distinct")
        object.__setattr__(self, "local_positions_m", positions)
        if self.anchor_prim_path is not None:
            validate_prim_path(self.anchor_prim_path)
        object.__setattr__(self, "marker_world_position_m", _xyz(self.marker_world_position_m,"marker_world_position_m"))
        object.__setattr__(self, "marker_world_orientation_xyzw", _quaternion(self.marker_world_orientation_xyzw))

    def capture(self, pose_at: Callable[[str], Pose3] | None = None) -> tuple[MicArray, list[list[float]], Pose3]:
        """Capture rig once for static rendering; caller owns USD thread safety."""
        if self.anchor_prim_path:
            if pose_at is None:
                raise ValueError("pose_at callback required for USD-attached microphone rig")
            base = pose_at(self.anchor_prim_path)
            if not isinstance(base, Pose3):
                raise TypeError("pose_at must return Pose3")
            center = compose_pose(base, self.center_m, self.orientation_xyzw)
        else:
            center = Pose3(self.center_m, self.orientation_xyzw)
        array = MicArray(Binding(world=center), self.local_positions_m)
        world_positions = [list(map(float,np.asarray(center.xyz)+rotate(center.quat_xyzw,p)))
                           for p in self.local_positions_m]
        return array, world_positions, center

    def public(self, pose_at=None) -> dict:
        data = asdict(self)
        data["count"] = len(self.local_positions_m)
        if self.anchor_prim_path is None or pose_at is not None:
            try:
                _, positions, center = self.capture(pose_at)
                data["world_positions_m"] = positions
                data["world_center_m"] = list(center.xyz)
                data["world_orientation_xyzw"] = list(center.quat_xyzw)
                data["pose_available"] = True
            except (KeyError, ValueError, TypeError) as exc:
                data["pose_available"] = False
                data["pose_error"] = str(exc)[:220]
        else:
            data["pose_available"] = False
        return data


class MicrophoneRigStore:
    def __init__(self, manifest: Path | str, scene_id: str):
        self.manifest = Path(manifest).expanduser().resolve(strict=False)
        self.scene_id = scene_id
        self.rig = MicrophoneRig()
        self.revision = 0

    def set_rig(self, payload: dict, *, validate_anchor=None) -> MicrophoneRig:
        if not isinstance(payload, dict):
            raise ValueError("microphone configuration must be an object")
        permitted = set(MicrophoneRig.__dataclass_fields__) | {"preset"}
        extra = set(payload) - permitted
        if extra: raise ValueError(f"unknown microphone fields: {sorted(extra)}")
        values = {k:v for k,v in payload.items() if k != 'preset'}
        if "preset" in payload:
            name = payload["preset"]
            if name not in PRESETS: raise ValueError("unknown microphone preset")
            values["local_positions_m"] = PRESETS[name]
        proposed = replace(self.rig, **values)
        if validate_anchor and proposed.anchor_prim_path:
            validate_anchor(proposed.anchor_prim_path)
        old = self.rig
        self.rig = proposed
        self.revision += 1
        try: self.save()
        except Exception:
            self.rig = old
            self.revision -= 1
            raise
        return proposed

    def save(self):
        path = self.manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema_version":1,"scene_id":self.scene_id,"microphone_rig":asdict(self.rig)}
        fd,tmp = tempfile.mkstemp(prefix=".onda_mics_",suffix=".tmp",dir=str(path.parent))
        try:
            os.chmod(tmp,0o600)
            with os.fdopen(fd,"w",encoding="utf-8") as file:
                json.dump(record,file,sort_keys=True,indent=2,allow_nan=False)
                file.write("\n"); file.flush(); os.fsync(file.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def load(self, *, validate_anchor=None) -> dict:
        if not self.manifest.is_file(): return {"loaded":False,"reason":"no_manifest"}
        data=json.loads(self.manifest.read_text(encoding="utf-8"))
        if data.get("schema_version")!=1: raise ValueError("unsupported microphone rig manifest schema")
        if data.get("scene_id")!=self.scene_id: return {"loaded":False,"reason":"scene_id_mismatch"}
        candidate=MicrophoneRig(**data["microphone_rig"])
        # Session-layer marker is intentionally absent until recreated by host.
        if validate_anchor and candidate.anchor_prim_path not in (None,MARKER_PATH):
            validate_anchor(candidate.anchor_prim_path)
        self.rig=candidate; self.revision+=1
        return {"loaded":True,"count":len(candidate.local_positions_m)}
