#!/usr/bin/env python3
"""Onda v0.19 explicit-material demo: a door swings closed then open.

Uses geometric intersection with a frequency-dependent measured-or-authored
transmission curve; NOT calibrated against a specific real door. There are no
reflections, leaks, or diffraction around the door in this direct-path demo.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import wave
import numpy as np

from onda_hearing.acoustic_scene import (
    AcousticSceneRenderer, RendererConfig, Pose3, Binding, Schedule,
    Emitter, MicArray, AudioAsset,
)
from onda_hearing.acoustic_scene.physical import (
    AcousticGeometry, AcousticSurface, TransmissionMaterial,
    GeometryAwareAcousticRenderer, PhysicalConfig,
)

SR = 16000
MICROPHONES = ((0., 0., 0.), (0., 0.04, 0.),
               (0., -0.04, 0.), (0.04, 0., 0.))


def door_at_angle(degrees, material):
    # Rotate around z-axis at the hinge of this finite 1.7-m-wide door.
    hinge = np.array((0., -0.85, 0.))
    relative = np.array(((0., 0., 0.), (0., 1.7, 0.),
                         (0., 1.7, 2.1), (0., 0., 2.1)))
    a = np.deg2rad(degrees)
    rotation = np.array(((np.cos(a), -np.sin(a), 0.),
                         (np.sin(a), np.cos(a), 0.), (0., 0., 1.)))
    return AcousticSurface.quad("swinging_door", (rotation @ relative.T).T + hinge,
                               material)


def geometry(door_angle):
    # DEMONSTRATION-only hypothetical TL curves; not a measured construction.
    wood = TransmissionMaterial(
        (125., 500., 1000., 4000., 8000.),
        (6., 10., 16., 29., 35.),
        "DEMO_SYNTHETIC_DOOR_TL_NOT_MEASURED")
    wall = TransmissionMaterial(
        (125., 500., 1000., 4000., 8000.),
        (35., 40., 45., 55., 60.),
        "DEMO_SYNTHETIC_WALL_TL_NOT_MEASURED")
    pieces = [door_at_angle(door_angle, wood)]
    for label, yy in (("wall_left", (-3., -.85)),
                      ("wall_right", (.85, 3.))):
        lo, hi = yy
        pieces.append(AcousticSurface.quad(
            label, ((0,lo,0),(0,hi,0),(0,hi,3),(0,lo,3)), wall))
    pieces.append(AcousticSurface.quad(
        "door_lintel", ((0,-.85,2.1),(0,.85,2.1),
                          (0,.85,3.),(0,-.85,3.)), wall))
    return AcousticGeometry(tuple(pieces))


def create_asset():
    t = np.arange(SR*3) / SR
    # Original synthetic test signal: tonal low+high components for
    # quantifying frequency-selective transmission, plus a slow ring cadence.
    carrier = (np.sin(2*np.pi*440*t) + 0.65*np.sin(2*np.pi*3000*t))
    envelope = np.where((t % 1.2) < .8, 1., .0)
    return AudioAsset("ringer", (0.12*carrier*envelope).astype(np.float32), SR)


def create_renderer(scene):
    array = MicArray(Binding(world=Pose3((1.5, 0., 1.2))), MICROPHONES)
    source = Emitter("phone", "ringer", Binding(world=Pose3((-1.5, 0., 1.2))),
                     Schedule(onset_s=0.))
    return GeometryAwareAcousticRenderer(
        RendererConfig(sample_rate=SR, clip_output=False), array,
        [source], {"ringer": create_asset()}, scene,
        physical_config=PhysicalConfig(
            geometry_step_samples=256, transition_samples=320,
            transmission_taps=129))


def render(renderer, begin, length, block=512):
    output = []
    for cursor in range(begin, begin+length, block):
        output.append(renderer.render(cursor, min(block,begin+length-cursor)))
    return np.concatenate(output)


def export_wav(path: Path, samples: np.ndarray):
    peak = float(np.max(np.abs(samples)))
    if peak > 0.99 or not np.isfinite(samples).all():
        raise ValueError(f"output requires gain adjustment; peak={peak:.5f}")
    pcm = np.round(np.asarray(samples)*32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(samples.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def amplitude(signal, freq_hz, start_s=.10, duration_s=.4):
    start, count = round(start_s*SR), round(duration_s*SR)
    x = signal[start:start+count].astype(np.float64)
    oscillator = np.exp(-2j*np.pi*freq_hz*np.arange(len(x))/SR)
    return float(abs(2*np.dot(x, oscillator)/len(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/v019_door_demo")
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    open_audio = render(create_renderer(geometry(90.)), 0, SR*3)
    closed_audio = render(create_renderer(geometry(0.)), 0, SR*3)
    env = geometry(90.)
    dynamic = create_renderer(env)
    first = render(dynamic, 0, SR)
    env.put(door_at_angle(0., env.surfaces["swinging_door"].material))
    second = render(dynamic, SR, SR)
    env.put(door_at_angle(90., env.surfaces["swinging_door"].material))
    third = render(dynamic, SR*2, SR)
    moving_audio = np.concatenate((first, second, third))
    for filename, values in (("door_open_4ch.wav", open_audio),
                             ("door_closed_4ch.wav", closed_audio),
                             ("door_open_closed_open_4ch.wav", moving_audio)):
        export_wav(output / filename, values)
    losses = {}
    for freq in (440., 3000.):
        opened = amplitude(open_audio[:,0], freq)
        closed = amplitude(closed_audio[:,0], freq)
        losses[str(freq)] = 20*math.log10(opened/max(closed,1e-12))
    report = {
        "experiment": "Onda v0.19 demo door screen transmission",
        "status": "synthetic_example_not_measured_room_fidelity",
        "conditions": ["open", "closed", "open_closed_open"],
        "frequencies_hz": [440, 3000],
        "open_minus_closed_db": losses,
        "scene": "two rooms divided by wall with finite hinged door; direct transmission only",
        "source_distance_m": 3.,
        "microphone_count": 4,
        "material_provenance": "DEMO_SYNTHETIC_UNCALIBRATED",
        "limitations": ["no diffraction", "no realistic seal leakage",
                        "no room-derived reflections", "no physical validation"],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("OUTPUT DIRECTORY:", output)


if __name__ == "__main__":
    main()
