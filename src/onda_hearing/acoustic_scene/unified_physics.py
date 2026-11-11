"""Onda v0.20: unified direct + specular geometry propagation (experimental).

Direct transmission is rendered EXACTLY ONCE by v0.19's source-isolated
GeometryAwareAcousticRenderer. Explicit finite-surface image paths of orders
1-2 are synthesized separately and pressure-summed at the microphones.

Assumptions: geometric acoustics, quasi-static scene at receiver clock ticks,
point emitters, locally planar convex/triangulated reflection facets,
uncalibrated phase (minimum phase), no edge diffraction, aperture leakage,
structural vibrations or late stochastic reverberation. Real-world fidelity
requires measured materials and validated geometry.

Implementation is independent of MuJoCo, USD and Pyroomacoustics at runtime.
DO NOT sum with another full (direct + reflected) RIR: direct double-counting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
from typing import Mapping
import math
import numpy as np

from .adapters import PoseResolver, resolve
from .assets import AudioAsset
from .physical import (AcousticGeometry, PhysicalConfig, GeometryAwareAcousticRenderer,
                       TransmissionMaterial, AcousticSurface,
                       minimum_phase_transmission, _FilterState)
from .renderer import AcousticSceneRenderer, RendererConfig
from .types import Emitter, MicArray, rotate


@dataclass(frozen=True)
class ReflectionMaterial:
    """Author/measured ENERGY absorption + scattering by frequency.

    Reflection *pressure* amplitude for the specular branch is approximated as
    sqrt((1 - energy_absorption) * (1 - scattering)). Phase and angle-dependent
    impedance are not modeled; these coefficients are NOT wall transmission.
    """
    frequencies_hz: tuple[float, ...]
    energy_absorption: tuple[float, ...]
    scattering: tuple[float, ...]
    provenance: str
    pressure_polarity: int = 1

    def __post_init__(self):
        f = np.asarray(self.frequencies_hz, dtype=float)
        a = np.asarray(self.energy_absorption, dtype=float)
        s = np.asarray(self.scattering, dtype=float)
        if (f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all() or
                np.any(f <= 0) or np.any(np.diff(f) <= 0)):
            raise ValueError('reflection frequencies must be finite, positive and increasing')
        for label, values in (('absorption', a), ('scattering', s)):
            if values.shape != f.shape or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f'{label} must be a finite 0..1 energy coefficient at each frequency')
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('reflection coefficient provenance required')
        if self.pressure_polarity not in (-1, 1):
            raise ValueError('pressure_polarity must be +1 or -1')
        object.__setattr__(self, 'frequencies_hz', tuple(float(x) for x in f))
        object.__setattr__(self, 'energy_absorption', tuple(float(x) for x in a))
        object.__setattr__(self, 'scattering', tuple(float(x) for x in s))

    def equivalent_loss(self) -> TransmissionMaterial:
        a = np.asarray(self.energy_absorption)
        s = np.asarray(self.scattering)
        # Intensity-to-pressure conversion: 20log10(sqrt(E)) = 10log10(E).
        losses = np.minimum(120., -10. * np.log10(np.maximum((1.-a)*(1.-s), 1.e-12)))
        return TransmissionMaterial(self.frequencies_hz, tuple(losses),
                                    f'specular reflection approximation: {self.provenance}')


@dataclass(frozen=True)
class UnifiedPhysicsConfig:
    max_specular_order: int = 1
    max_reflectors: int = 12
    max_path_states: int = 768
    reflection_taps: int = 129
    transition_samples: int = 320
    max_path_length_m: float = 100.
    geometry_step_samples: int = 256

    def __post_init__(self):
        if self.max_specular_order not in (0, 1, 2):
            raise ValueError('max_specular_order must be 0, 1 or 2')
        if not isinstance(self.max_reflectors, int) or not 1 <= self.max_reflectors <= 24:
            raise ValueError('max_reflectors must be between 1 and 24')
        if not isinstance(self.max_path_states, int) or not 1 <= self.max_path_states <= 10000:
            raise ValueError('max_path_states out of range')
        if (not isinstance(self.reflection_taps, int) or
                not 17 <= self.reflection_taps <= 1025 or self.reflection_taps % 2 != 1):
            raise ValueError('reflection_taps must be odd in 17..1025')
        if not isinstance(self.transition_samples, int) or self.transition_samples < 0:
            raise ValueError('invalid transition length')
        if not isinstance(self.geometry_step_samples, int) or self.geometry_step_samples < 1:
            raise ValueError('invalid geometry update period')
        if not math.isfinite(self.max_path_length_m) or self.max_path_length_m <= 0:
            raise ValueError('invalid path-length limit')


def _plane(surface: AcousticSurface):
    """Return a single finite planar facet. Fail rather than invent geometry."""
    v = np.asarray(surface.vertices_m, dtype=np.float64)
    a,b,c = v[np.asarray(surface.triangles[0], dtype=int)]
    n = np.cross(b-a, c-a)
    d = float(np.linalg.norm(n))
    if d <= 1.e-10:
        raise ValueError(f'zero-area reflector: {surface.surface_id}')
    n /= d
    scale = max(1., float(np.max(np.linalg.norm(v-a, axis=1))))
    if np.max(np.abs((v-a) @ n)) > 1.e-6*scale:
        raise ValueError(f'reflector must be planar: {surface.surface_id}')
    return a, n


def _inside_triangles(point, surface: AcousticSurface):
    v = np.asarray(surface.vertices_m, dtype=np.float64)
    for indices in surface.triangles:
        a,b,c = v[np.asarray(indices, dtype=int)]
        e0, e1, p = b-a, c-a, point-a
        aa,ab,bb = np.dot(e0,e0), np.dot(e0,e1), np.dot(e1,e1)
        pa,pb = np.dot(p,e0), np.dot(p,e1)
        det = aa*bb-ab*ab
        if det < 1.e-15:
            continue
        u = (pa*bb-pb*ab)/det
        w = (pb*aa-pa*ab)/det
        if u >= -1.e-8 and w >= -1.e-8 and u+w <= 1.+1.e-8:
            return True
    return False


def _mirror(point, a, n):
    return point-2.*float(np.dot(point-a,n))*n


def _segment_plane(a, b, p, n):
    denom = float(np.dot(b-a,n))
    if abs(denom) < 1.e-10:
        return None
    u = float(np.dot(p-a,n))/denom
    if not 1.e-7 < u < 1.-1.e-7:
        return None
    return a+u*(b-a)


@dataclass(frozen=True)
class SpecularPath:
    surface_ids: tuple[str, ...]
    points_m: tuple[tuple[float, float, float], ...]
    length_m: float
    penetrated: tuple[str, ...]
    direction_from_source: tuple[float, float, float]


def find_specular_path(source_m, mic_m, surfaces: tuple[AcousticSurface, ...],
                       geometry: AcousticGeometry, *, max_path_length_m=100.) -> SpecularPath | None:
    """Finite-image path with per-bounce visibility and segment obstructions.

    Ordered planes: first surface receives the first source-side bounce.
    Reverse tracing uses the successive image sources. For order 2 this
    prevents the geometrically impossible straight-line 'reflection' shortcut.
    """
    s = np.asarray(source_m, dtype=float)
    m = np.asarray(mic_m, dtype=float)
    if s.shape != (3,) or m.shape != (3,) or not np.isfinite(s).all() or not np.isfinite(m).all():
        raise ValueError('source/microphone must be finite 3-vectors')
    if not 1 <= len(surfaces) <= 2:
        raise ValueError('only one or two ordered bounces supported')
    if len({x.surface_id for x in surfaces}) != len(surfaces):
        return None
    if any(not x.enabled for x in surfaces):
        return None
    planes = [_plane(x) for x in surfaces]
    images = [s]
    for a,n in planes:
        images.append(_mirror(images[-1],a,n))
    current = m
    bounces_reversed = []
    for i in range(len(surfaces)-1, -1, -1):
        a,n = planes[i]
        hit = _segment_plane(current,images[i+1],a,n)
        if hit is None or not _inside_triangles(hit,surfaces[i]):
            return None
        bounces_reversed.append(hit)
        current = hit
    bounces = list(reversed(bounces_reversed))
    vertices = [s, *bounces, m]
    for i,(a,n) in enumerate(planes):
        incident = float(np.dot(vertices[i]-a,n))
        outgoing = float(np.dot(vertices[i+2]-a,n))
        # Specular reflection requires both legs to be on the SAME side.
        if incident*outgoing <= 1.e-11:
            return None
    lengths = [float(np.linalg.norm(q-p)) for p,q in zip(vertices[:-1],vertices[1:])]
    total = sum(lengths)
    if not (1.e-6 < total <= max_path_length_m):
        return None
    penetrated = []
    for p,q in zip(vertices[:-1],vertices[1:]):
        # Intersection exactly at the intentional bounce endpoint is
        # excluded by v0.19's finite-segment ray test. An additional *interior*
        # crossing (including the same wall) is real transmission and must
        # NOT be skipped just because that wall also appears as a reflector.
        for occluder in geometry.occluders(p,q):
            penetrated.append(occluder.surface_id)
    direction = (bounces[0]-s)/lengths[0]
    return SpecularPath(
        tuple(x.surface_id for x in surfaces),
        tuple(tuple(float(z) for z in p) for p in bounces),
        total, tuple(penetrated), tuple(float(z) for z in direction),
    )


class _ReflectedState:
    """Streaming wet path with continuous, bounded sample-delay/gain ramps."""
    def __init__(self, taps, transition):
        self.filter = _FilterState(taps, transition)
        self.transition = max(1, transition)
        self.old_delay = self.target_delay = 0.
        self.old_gain = self.target_gain = 0.
        self.pos = self.transition
        self.initialized = False
        self.signature = None

    def _current(self):
        alpha = min(1., self.pos/self.transition)
        return ((1-alpha)*self.old_delay + alpha*self.target_delay,
                (1-alpha)*self.old_gain + alpha*self.target_gain)

    def update(self, delay, gain, impulse, signature):
        if not self.initialized:
            self.old_delay = self.target_delay = float(delay)
            self.old_gain = self.target_gain = float(gain)
            self.initialized = True
            self.pos = self.transition
            self.filter.update(impulse, signature)
            self.signature = signature
            return
        if (abs(delay-self.target_delay) > 1.e-7 or
                abs(gain-self.target_gain) > 1.e-10 or signature != self.signature):
            self.old_delay, self.old_gain = self._current()
            self.target_delay = float(delay)
            self.target_gain = float(gain)
            self.pos = 0
            self.filter.update(impulse, signature)
            self.signature = signature

    def render(self, samples, emitted):
        count = len(samples)
        if count == 0:
            return np.empty(0)
        frac = np.minimum(1., (self.pos + np.arange(1, count+1))/self.transition)
        delay = self.old_delay*(1.-frac) + self.target_delay*frac
        gain = self.old_gain*(1.-frac) + self.target_gain*frac
        input_samples = emitted(samples.astype(np.float64)-delay)
        filtered = self.filter.process(input_samples)
        self.pos += count
        return gain*filtered


class UnifiedGeometryAcousticRenderer:
    """Combined direct-transmission + finite order 1/2 specular renderer.

    ``reflectors`` maps *AcousticGeometry surface IDs* to explicit reflection
    energy coefficients. Surfaces not in this mapping can still occlude rays.
    A single source emission schedule serves direct and reflected paths.

    `geometry.put(...)` updates panel/door mesh between render() calls.
    No concurrent scene writes. Moving surfaces are sampled at update ticks;
    sound scattered from moving boundaries is quasi-static (not Doppler exact).
    """
    def __init__(self, config: RendererConfig, microphones: MicArray,
                 emitters: list[Emitter], assets: Mapping[str, AudioAsset],
                 geometry: AcousticGeometry,
                 reflectors: Mapping[str, ReflectionMaterial],
                 resolver: PoseResolver | None = None,
                 unified_config: UnifiedPhysicsConfig = UnifiedPhysicsConfig()):
        if config.rir_taps is not None:
            raise ValueError('legacy full RIR would double-count direct sound')
        if not isinstance(geometry, AcousticGeometry):
            raise TypeError('geometry must be AcousticGeometry')
        if len(reflectors) > unified_config.max_reflectors:
            raise ValueError('too many reflector surfaces for configured budget')
        for sid, material in reflectors.items():
            if sid not in geometry.surfaces:
                raise KeyError(f'reflector surface absent from geometry: {sid}')
            if not isinstance(material, ReflectionMaterial):
                raise TypeError('reflectors must map IDs to ReflectionMaterial')
            _plane(geometry.surfaces[sid])
        self.config, self.microphones = config, microphones
        self.emitters = tuple(emitters)
        self.geometry, self.reflectors, self.resolver = geometry, dict(reflectors), resolver
        self.unified_config = unified_config
        self.next_sample = 0
        self._names = tuple(sorted(reflectors))
        self._sequences = []
        for order in range(1, unified_config.max_specular_order+1):
            self._sequences.extend(permutations(self._names, order))
        count = len(self._sequences)*len(emitters)*len(microphones.local_positions_m)
        if count > unified_config.max_path_states:
            raise ValueError(f'path-state budget exceeded: {count} > {unified_config.max_path_states}')
        direct_cfg = replace(config, noise_std=0., clip_output=False)
        self.direct = GeometryAwareAcousticRenderer(
            direct_cfg, microphones, emitters, assets, geometry, resolver,
            physical_config=PhysicalConfig(
                geometry_step_samples=unified_config.geometry_step_samples,
                transition_samples=unified_config.transition_samples,
                transmission_taps=unified_config.reflection_taps,
            ),
        )
        self._noise = AcousticSceneRenderer(replace(config, clip_output=False),
                                           microphones, [], {}, resolver)
        self._states: dict[tuple[str,int,tuple[str,...]],_ReflectedState] = {}
        self._firs: dict[tuple, np.ndarray] = {}
        self._last_revision = -1
        self.last_reflections: dict[tuple, dict] = {}
        self.total_overrange_samples = 0
        self.clipped_samples = 0
        self.max_abs_sample = 0.
        self.geometry_updates = 0
        self.active_specular_paths = 0
        self.reflection_model = 'finite_image_specular_order_1_2'

    def reset(self):
        self.direct.reset()
        self._noise.reset()
        self._states.clear()
        self._last_revision = -1
        self.next_sample = 0
        self.last_reflections.clear()
        self.total_overrange_samples = 0
        self.clipped_samples = 0
        self.max_abs_sample = 0.
        self.geometry_updates = 0
        self.active_specular_paths = 0

    def trigger(self, emitter_id: str, at_sample: int):
        self.direct.trigger(emitter_id, at_sample)

    def stop(self, emitter_id: str, at_sample: int):
        self.direct.stop(emitter_id, at_sample)

    def set_gain_db(self, emitter_id: str, gain_db: float):
        self.direct.set_gain_db(emitter_id, gain_db)

    def _path_fir(self, sequence: tuple[str,...], penetrated: tuple[str,...]):
        # Physical property changes also invalidate the cache signature.
        refl = tuple((sid, self.reflectors[sid]) for sid in sequence)
        transmitted = tuple((sid, self.geometry.surfaces[sid].material) for sid in penetrated)
        key = (refl, transmitted)
        if key not in self._firs:
            if len(self._firs) >= 512:
                self._firs.clear()
            materials = [self.reflectors[sid].equivalent_loss() for sid in sequence]
            materials.extend(self.geometry.surfaces[sid].material for sid in penetrated)
            self._firs[key] = minimum_phase_transmission(
                tuple(materials), self.config.sample_rate,
                taps=self.unified_config.reflection_taps,
            )
            sign = math.prod(self.reflectors[sid].pressure_polarity for sid in sequence)
            self._firs[key] *= sign
        return self._firs[key], key

    def _update_paths(self, sample: int):
        t = sample / self.config.sample_rate
        array_pose = resolve(self.microphones.binding, self.resolver, t)
        active = 0
        self.last_reflections.clear()
        for e in self.emitters:
            src_renderer = self.direct._sources[e.emitter_id]
            if not e.enabled:
                continue
            for ch, local in enumerate(self.microphones.local_positions_m):
                mic = np.asarray(array_pose.xyz) + rotate(array_pose.quat_xyzw, local)
                for sequence in self._sequences:
                    key = (e.emitter_id, ch, sequence)
                    state = self._states.setdefault(
                        key, _ReflectedState(self.unified_config.reflection_taps,
                                             self.unified_config.transition_samples))
                    surfaces = tuple(self.geometry.surfaces[sid] for sid in sequence)
                    source_pose = resolve(e.binding, self.resolver, t)
                    p = find_specular_path(source_pose.xyz, mic, surfaces, self.geometry,
                                           max_path_length_m=self.unified_config.max_path_length_m)
                    # One retarded-time refinement for source movement.
                    if p is not None:
                        emit_t = max(0., t-p.length_m/self.config.sound_speed_m_s)
                        source_pose = resolve(e.binding, self.resolver, emit_t)
                        p = find_specular_path(source_pose.xyz, mic, surfaces, self.geometry,
                                               max_path_length_m=self.unified_config.max_path_length_m)
                    if p is None:
                        if state.initialized:
                            # Retain the previous delay/FIR during the fade-out.
                            state.update(state.target_delay, 0., state.filter.current,
                                         state.signature)
                        continue
                    delay = p.length_m*self.config.sample_rate/self.config.sound_speed_m_s
                    gain = self.config.reference_distance_m/max(self.config.min_distance_m,p.length_m)
                    if e.directivity == 'cardioid':
                        fwd = rotate(source_pose.quat_xyzw, (1.,0.,0.))
                        gain *= max(0., .5*(1.+float(np.dot(fwd,p.direction_from_source))))
                    gain *= 10.**(src_renderer._gain_overrides.get(e.emitter_id, e.gain_db)/20.)
                    fir, signature = self._path_fir(sequence,p.penetrated)
                    state.update(delay,gain,fir,signature)
                    self.last_reflections[key] = {
                        'path_length_m':p.length_m,
                        'delay_s':p.length_m/self.config.sound_speed_m_s,
                        'bounce_points_m':p.points_m,
                        'penetrated':p.penetrated,
                    }
                    active += 1
        self._last_revision = self.geometry.revision
        self.geometry_updates += 1
        self.active_specular_paths = active

    def render(self, start_sample: int, frames: int) -> np.ndarray:
        if not isinstance(start_sample, int) or start_sample != self.next_sample:
            raise ValueError('sequential render required (reset before new episode)')
        if not isinstance(frames, int) or not 1 <= frames <= 2*self.config.sample_rate:
            raise ValueError('invalid render chunk size')
        channels = len(self.microphones.local_positions_m)
        out = np.zeros((frames,channels),dtype=np.float64)
        step = self.unified_config.geometry_step_samples
        cursor, end = start_sample, start_sample+frames
        while cursor < end:
            if self._last_revision != self.geometry.revision or cursor % step == 0:
                self._update_paths(cursor)
            n = min(end-cursor,step-(cursor % step))
            dry = self.direct.render(cursor,n).astype(np.float64)
            out[cursor-start_sample:cursor-start_sample+n] = dry
            samples = np.arange(cursor,cursor+n,dtype=np.int64)
            for e in self.emitters:
                if not e.enabled:
                    continue
                renderer = self.direct._sources[e.emitter_id]
                for ch in range(channels):
                    for sequence in self._sequences:
                        state = self._states.get((e.emitter_id,ch,sequence))
                        if state is not None:
                            out[cursor-start_sample:cursor-start_sample+n,ch] += (
                                state.render(samples,lambda indices,r=renderer,e=e:
                                             r._emitted(e,indices)))
            cursor += n
        out += self._noise.render(start_sample,frames).astype(np.float64)
        if not np.isfinite(out).all():
            raise FloatingPointError('nonfinite unified PCM')
        peak = float(np.max(np.abs(out)))
        self.max_abs_sample = max(self.max_abs_sample,peak)
        n_over = int(np.count_nonzero(np.abs(out)>1.))
        self.total_overrange_samples += n_over
        if self.config.clip_output:
            np.clip(out,-1.,1.,out=out)
            self.clipped_samples += n_over
        self.next_sample = end
        return np.ascontiguousarray(out,dtype=np.float32)

    def telemetry(self) -> dict:
        return {
            'physics_mode':'unified_geometric_specular_v020_experimental',
            'samples_rendered_per_channel':self.next_sample,
            'geometry_revision':self.geometry.revision,
            'path_states':len(self._states),
            'active_specular_paths_at_last_update':self.active_specular_paths,
            'geometry_updates':self.geometry_updates,
            'max_abs_sample':self.max_abs_sample,
            'overrange_samples':self.total_overrange_samples,
            'clipped_samples':self.clipped_samples,
            'direct':self.direct.telemetry(),
            'assumptions':['geometric_acoustics','order_1_or_2_specular',
                           'quasi_static_moving_boundary','frequency_energy_coefficients',
                           'no_diffraction_or_leakage','no_late_diffuse_reverb'],
        }
