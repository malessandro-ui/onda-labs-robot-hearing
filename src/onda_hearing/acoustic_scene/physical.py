"""Onda experimental scene-aware direct-path physics backend (v0.19).

A thin, additive wrapper over v0.17 AcousticSceneRenderer. The original
renderer supplies per-emitter pressure, travel time, microphone geometry,
source scheduling, and continuous motion. This module adds explicit,
frequency-dependent transmission through intersected triangle meshes.

Scope: direct sound with **thin-screen** transmission, not diffraction,
structural vibration, cavity leakage, or geometry-derived reflections.
No physical accuracy claim without calibrated material data and measurement.

No Isaac/Usd or MuJoCo dependency: scene adapters supply world-space meshes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import math
import numpy as np

from .adapters import PoseResolver, resolve
from .renderer import RendererConfig, AcousticSceneRenderer
from .types import Emitter, MicArray, rotate
from .assets import AudioAsset


def _points(x: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{label} must have shape {shape}, all finite")
    return arr.copy()


@dataclass(frozen=True)
class TransmissionMaterial:
    """Measured/author-supplied thin-screen transmission loss, in dB.

    This is NOT an absorption or reflection curve. No generic material names
    are mapped to unsupported measured transmission coefficients.
    """

    frequencies_hz: tuple[float, ...]
    transmission_loss_db: tuple[float, ...]
    provenance: str

    def __post_init__(self):
        f = np.asarray(self.frequencies_hz, dtype=float)
        db = np.asarray(self.transmission_loss_db, dtype=float)
        if (f.ndim != 1 or f.size < 2 or db.shape != f.shape or
                not np.isfinite(f).all() or not np.isfinite(db).all() or
                f[0] <= 0 or np.any(np.diff(f) <= 0) or
                np.any(db < 0) or np.any(db > 120)):
            raise ValueError("invalid monotonically ascending material frequency/loss data")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("material provenance is required")
        object.__setattr__(self, "frequencies_hz", tuple(map(float, f)))
        object.__setattr__(self, "transmission_loss_db", tuple(map(float, db)))


@dataclass(frozen=True)
class AcousticSurface:
    """One geometrically thin, double-sided mesh barrier in world meters.

    Each physical panel must be represented ONCE (not two duplicate faces).
    Triangles use 0-based vertex indices, with no implicit hole filling.
    """

    surface_id: str
    vertices_m: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    material: TransmissionMaterial
    enabled: bool = True

    def __post_init__(self):
        if not self.surface_id or not isinstance(self.material, TransmissionMaterial):
            raise ValueError("surface ID and material required")
        v = np.asarray(self.vertices_m, dtype=float)
        t = np.asarray(self.triangles, dtype=int)
        if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or
                not np.isfinite(v).all() or t.ndim != 2 or
                t.shape[1] != 3 or len(t) < 1 or
                np.any(t < 0) or np.any(t >= len(v))):
            raise ValueError("invalid mesh vertices or triangles")
        if any(len(set(row)) != 3 for row in t):
            raise ValueError("degenerate triangle indices")
        for face in t:
            a, b, c = v[face]
            if np.linalg.norm(np.cross(b-a, c-a)) <= 1.e-10:
                raise ValueError("degenerate or zero-area triangle")
        object.__setattr__(self, "vertices_m", tuple(tuple(map(float, row)) for row in v))
        object.__setattr__(self, "triangles", tuple(tuple(map(int, row)) for row in t))

    @classmethod
    def quad(cls, surface_id: str, corners_m: object,
             material: TransmissionMaterial, enabled: bool = True):
        corners = _points(corners_m, (4, 3), "quad corners")
        n = np.cross(corners[1]-corners[0], corners[2]-corners[0])
        n /= max(np.linalg.norm(n), 1.e-15)
        if abs(float(np.dot(corners[3]-corners[0], n))) > 1.e-6:
            raise ValueError("quad must be planar")
        edge = np.roll(corners, -1, axis=0)-corners
        turns = np.array([np.dot(np.cross(edge[i], edge[(i+1)%4]), n)
                          for i in range(4)])
        if np.any(turns <= 1.e-10):
            raise ValueError("quad corners must be convex, cyclically ordered")
        return cls(surface_id,
                   tuple(map(tuple, corners)), ((0, 1, 2), (0, 2, 3)),
                   material, enabled)


def _triangle_intersects_segment(start: np.ndarray, end: np.ndarray,
                                 a: np.ndarray, b: np.ndarray,
                                 c: np.ndarray) -> bool:
    """Double-sided finite-segment Möller–Trumbore with endpoint exclusion."""
    direction = end-start
    edge1, edge2 = b-a, c-a
    p = np.cross(direction, edge2)
    det = float(np.dot(edge1, p))
    if abs(det) <= 1.e-11:
        return False
    inv = 1. / det
    tvec = start-a
    u = float(np.dot(tvec, p)) * inv
    if u < -1.e-9 or u > 1. + 1.e-9:
        return False
    q = np.cross(tvec, edge1)
    v = float(np.dot(direction, q)) * inv
    if v < -1.e-9 or u + v > 1. + 1.e-9:
        return False
    t = float(np.dot(edge2, q)) * inv
    return 1.e-7 < t < 1. - 1.e-7


class AcousticGeometry:
    """Mutable explicit world-space meshes. Update at the next audio sample boundary.

    Single-threaded by contract; no concurrent mutation during render().
    """
    def __init__(self, surfaces: tuple[AcousticSurface, ...] = ()):
        self.surfaces: dict[str, AcousticSurface] = {}
        self._prepared = {}  # compiled geometry + AABB; invalidated on put()
        self.revision = 0
        for surface in surfaces:
            self.put(surface)

    def put(self, surface: AcousticSurface):
        if not isinstance(surface, AcousticSurface):
            raise TypeError("expected AcousticSurface")
        self.surfaces[surface.surface_id] = surface
        vertices = np.asarray(surface.vertices_m, dtype=np.float64)
        triangles = np.asarray(surface.triangles, dtype=np.int32)
        self._prepared[surface.surface_id] = (
            vertices, triangles,
            vertices.min(axis=0), vertices.max(axis=0),
        )
        self.revision += 1

    def remove(self, surface_id: str):
        del self.surfaces[surface_id]
        del self._prepared[surface_id]
        self.revision += 1

    def occluders(self, start: object, end: object) -> tuple[AcousticSurface, ...]:
        a, b = _points(start, (3,), "start"), _points(end, (3,), "end")
        if np.linalg.norm(a-b) <= 1.e-9:
            return ()
        crossed = []
        path_min, path_max = np.minimum(a, b), np.maximum(a, b)
        for surface in self.surfaces.values():
            if not surface.enabled:
                continue
            v, triangles, box_min, box_max = self._prepared[surface.surface_id]
            if np.any(path_max < box_min) or np.any(path_min > box_max):
                continue  # broad-phase AABB rejection
            # One attenuation per physical panel, even when a ray lies on
            # the diagonal shared by two triangles of that panel.
            if any(_triangle_intersects_segment(a, b, *v[tri])
                   for tri in triangles):
                crossed.append(surface)
        return tuple(crossed)


def minimum_phase_transmission(materials: tuple[TransmissionMaterial, ...],
                               sample_rate: int, taps: int = 129) -> np.ndarray:
    """Causal minimum-phase approximation of magnitude-only TL curves.

    Uses real-cepstrum spectral factorization, preserving a nonnegative
    transmission magnitude without adding a linear-phase (taps-1)/2 delay.
    Finite FIR truncation introduces approximation error; validate frequency
    response against physical measurement before claiming calibrated results.
    """
    if not isinstance(sample_rate, int) or sample_rate < 2000:
        raise ValueError("invalid sample rate")
    if not isinstance(taps, int) or not 17 <= taps <= 1025 or taps % 2 != 1:
        raise ValueError("taps must be odd and between 17 and 1025")
    if not materials:
        h = np.zeros(taps, dtype=np.float64)
        h[0] = 1.
        return h
    nfft = 1
    while nfft < max(8192, 32*taps):
        nfft *= 2
    f = np.fft.rfftfreq(nfft, 1./sample_rate)
    db = np.zeros_like(f)
    for m in materials:
        if not isinstance(m, TransmissionMaterial):
            raise TypeError("material must be TransmissionMaterial")
        db += np.interp(f, m.frequencies_hz, m.transmission_loss_db,
                        left=m.transmission_loss_db[0],
                        right=m.transmission_loss_db[-1])
    # Real minimum-phase cepstrum corresponding to the chosen log magnitude.
    log_mag = np.clip(-np.log(10.) * db / 20., -60., 0.)
    cep = np.fft.irfft(log_mag, nfft)
    min_cep = np.zeros_like(cep)
    min_cep[0] = cep[0]
    min_cep[1:nfft//2] = 2.*cep[1:nfft//2]
    min_cep[nfft//2] = cep[nfft//2]
    h = np.fft.irfft(np.exp(np.fft.rfft(min_cep)), nfft)[:taps]
    if not np.isfinite(h).all():
        raise FloatingPointError("invalid transmission FIR")
    return h.astype(np.float64)


@dataclass(frozen=True)
class PhysicalConfig:
    geometry_step_samples: int = 256
    transition_samples: int = 320
    transmission_taps: int = 129

    def __post_init__(self):
        if not isinstance(self.geometry_step_samples, int) or self.geometry_step_samples <= 0:
            raise ValueError("geometry_step_samples must be positive")
        if not isinstance(self.transition_samples, int) or self.transition_samples < 0:
            raise ValueError("transition_samples must be nonnegative")
        if (not isinstance(self.transmission_taps, int) or
                self.transmission_taps < 17 or self.transmission_taps > 1025 or
                self.transmission_taps % 2 != 1):
            raise ValueError("invalid transmission FIR tap count")


class _FilterState:
    def __init__(self, taps: int, transition: int):
        self.history = np.zeros(taps-1, dtype=np.float64)
        self.current = np.zeros(taps, dtype=np.float64)
        self.current[0] = 1.
        self.previous = None
        self.transition_pos = transition
        self.transition_samples = transition
        self.signature: tuple[str, ...] = ()
        self.initialized = False

    def update(self, impulse: np.ndarray, signature: tuple[str, ...]):
        if self.initialized and signature == self.signature:
            return
        if not self.initialized:
            self.current = impulse.copy()
            self.initialized = True
        else:
            if self.previous is not None and self.transition_pos < self.transition_samples:
                a = self.transition_pos/self.transition_samples
                self.current = (1.-a)*self.previous + a*self.current
            self.previous = self.current.copy() if self.transition_samples else None
            self.current = impulse.copy()
            self.transition_pos = 0 if self.previous is not None else self.transition_samples
        self.signature = signature

    def process(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        x = np.concatenate((self.history, values))
        current = np.convolve(x, self.current, mode="valid")
        if self.previous is not None and self.transition_pos < self.transition_samples:
            previous = np.convolve(x, self.previous, mode="valid")
            alpha = np.minimum(1., (self.transition_pos+np.arange(1, n+1)) /
                               self.transition_samples)
            y = previous*(1.-alpha) + current*alpha
            self.transition_pos += n
            if self.transition_pos >= self.transition_samples:
                self.previous = None
        else:
            y = current
        self.history = x[-len(self.history):].copy()
        return y


class GeometryAwareAcousticRenderer:
    """API-compatible PCM render/reset/trigger/stop/gain for v0.17 scenes.

    A source-isolated direct renderer is used for each emitter, allowing
    distinct occlusion/material processing per emitter-microphone path.
    No blocked sound is rerouted: occlusion is a thin-screen transmission.
    """
    def __init__(self, config: RendererConfig, microphones: MicArray,
                 emitters: list[Emitter], assets: Mapping[str, AudioAsset],
                 geometry: AcousticGeometry,
                 resolver: PoseResolver | None = None,
                 physical_config: PhysicalConfig = PhysicalConfig()):
        if config.rir_taps is not None:
            raise ValueError("legacy rir_taps cannot be combined with this direct-path backend")
        if not isinstance(geometry, AcousticGeometry):
            raise TypeError("geometry must be AcousticGeometry")
        self.config, self.microphones, self.emitters, self.assets = (config, microphones,
                                                                      tuple(emitters), dict(assets))
        self.geometry, self.resolver, self.physical_config = geometry, resolver, physical_config
        self.next_sample = 0
        # The existing renderer handles exact source schedules, geometry and delay.
        dry_config = replace(config, noise_std=0., clip_output=False, rir_taps=None)
        self._sources = {
            e.emitter_id: AcousticSceneRenderer(dry_config, microphones, [e], assets, resolver)
            for e in emitters
        }
        if len(self._sources) != len(emitters):
            raise ValueError("duplicate emitter IDs")
        # Preserve the original renderer's deterministic noise exactly once.
        self._noise = AcousticSceneRenderer(
            replace(config, clip_output=False, rir_taps=None),
            microphones, [], {}, resolver
        )
        self._states = {}
        self._cache: dict[tuple, np.ndarray] = {}
        self._last_revision = -1
        self.total_overrange_samples = 0
        self.clipped_samples = 0
        self.max_abs_sample = 0.
        self.last_paths = {}

    def trigger(self, emitter_id: str, at_sample: int):
        self._sources[emitter_id].trigger(emitter_id, at_sample)

    def stop(self, emitter_id: str, at_sample: int):
        self._sources[emitter_id].stop(emitter_id, at_sample)

    def set_gain_db(self, emitter_id: str, gain_db: float):
        self._sources[emitter_id].set_gain_db(emitter_id, gain_db)

    def reset(self):
        self.next_sample = 0
        for renderer in self._sources.values():
            renderer.reset()
        self._noise.reset()
        self._states.clear()
        self._last_revision = -1
        self.total_overrange_samples = 0
        self.clipped_samples = 0
        self.max_abs_sample = 0.
        self.last_paths.clear()
        # Geometry object intentionally retained. Reset scene geometry separately.

    def _update_paths(self, sample: int):
        now = sample / self.config.sample_rate
        array_pose = resolve(self.microphones.binding, self.resolver, now)
        for emitter in self.emitters:
            source_pose = resolve(emitter.binding, self.resolver, now)
            for ch, local in enumerate(self.microphones.local_positions_m):
                microphone_world = np.asarray(array_pose.xyz) + rotate(array_pose.quat_xyzw, local)
                hits = self.geometry.occluders(source_pose.xyz, microphone_world)
                signature = tuple(s.surface_id for s in hits)
                key = (emitter.emitter_id, ch)
                materials_key = tuple((s.surface_id, s.material.frequencies_hz,
                                       s.material.transmission_loss_db) for s in hits)
                if materials_key not in self._cache:
                    # Bounded cache: changing thousands of authored materials
                    # cannot grow the FIR cache without bound.
                    if len(self._cache) >= 512:
                        self._cache.clear()
                    self._cache[materials_key] = minimum_phase_transmission(
                        tuple(s.material for s in hits), self.config.sample_rate,
                        self.physical_config.transmission_taps)
                state = self._states.setdefault(
                    key, _FilterState(self.physical_config.transmission_taps,
                                      self.physical_config.transition_samples))
                state.update(self._cache[materials_key], materials_key)
                self.last_paths[key] = {
                    "blocked_by": list(signature),
                    "receiver_time_s": now,
                    "distance_m": float(np.linalg.norm(
                        microphone_world - np.asarray(source_pose.xyz))),
                    "approximation": "thin_screen_frequency_dependent_transmission",
                }
        self._last_revision = self.geometry.revision

    def render(self, start_sample: int, frames: int) -> np.ndarray:
        if not isinstance(start_sample, int) or start_sample != self.next_sample:
            raise ValueError("render must be called sequentially; reset for new episode")
        if not isinstance(frames, int) or not 1 <= frames <= self.config.sample_rate*2:
            raise ValueError("frames outside original renderer's 2-second maximum")
        total = np.zeros((frames, len(self.microphones.local_positions_m)), dtype=np.float64)
        cursor, offset, end = start_sample, 0, start_sample+frames
        step = self.physical_config.geometry_step_samples
        while cursor < end:
            if (not self._states or cursor % step == 0 or
                    self._last_revision != self.geometry.revision):
                self._update_paths(cursor)
            length = min(end-cursor, step-(cursor % step))
            mixture = np.zeros((length, total.shape[1]), dtype=np.float64)
            for e in self.emitters:
                dry = self._sources[e.emitter_id].render(cursor, length)
                for ch in range(total.shape[1]):
                    mixture[:, ch] += self._states[(e.emitter_id, ch)].process(dry[:, ch])
            # One global receiver noise stream, not one noise field per emitter.
            mixture += self._noise.render(cursor, length).astype(np.float64)
            total[offset:offset+length] = mixture
            cursor += length
            offset += length
        if not np.isfinite(total).all():
            raise FloatingPointError("nonfinite geometry-rendered audio")
        peak = float(np.max(np.abs(total)))
        self.max_abs_sample = max(self.max_abs_sample, peak)
        overranges = int(np.count_nonzero(np.abs(total) > 1.))
        self.total_overrange_samples += overranges
        if self.config.clip_output:
            np.clip(total, -1., 1., out=total)
            self.clipped_samples += overranges
        self.next_sample = end
        return np.ascontiguousarray(total, dtype=np.float32)

    def telemetry(self):
        return {
            "samples_rendered_per_channel": self.next_sample,
            "max_abs_sample": self.max_abs_sample,
            "overrange_samples": self.total_overrange_samples,
            "clipped_samples": self.clipped_samples,
            "scene_geometry_revision": self.geometry.revision,
            "ray_tested_paths": len(self.last_paths),
            "physics_mode": "geometry_direct_thin_screen_v019_experimental",
        }
