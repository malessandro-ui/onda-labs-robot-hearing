"""Fast, model-free tests for the v0.18 paired acoustic benchmark helpers."""
from __future__ import annotations
import importlib.util
import pathlib
import sys
import numpy as np

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "benchmark_acoustic_semantics_v018.py"
spec = importlib.util.spec_from_file_location("benchmark_acoustic_semantics_v018", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def synthetic():
    t = np.arange(m.WINDOW) / m.SR
    return (0.04*np.sin(2*np.pi*440*t) + 0.02*np.sin(2*np.pi*710*t)).astype(np.float32)


def test_fixed_length_delay_alignment_and_four_mics():
    x = synthetic()
    y, meta = m.render_aligned(x, distance_m=1., channel=0)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert 40 < meta["delay_samples"] < 60
    assert meta["renderer"]["samples_rendered_per_channel"] > len(x)
    assert meta["renderer"]["clipped_samples"] == 0
    assert np.corrcoef(x[1000:-1000], y[1000:-1000])[0,1] > .98


def test_rms_matching_preserves_level():
    x = synthetic()
    y, _ = m.render_aligned(x, distance_m=3.)
    z, gain = m.matched_level(y, x)
    assert gain > 1.
    assert abs(m.rms(z) / m.rms(x) - 1) < 1.e-6


def test_deterministic_noise_and_snr():
    x = synthetic()
    a, meta_a = m.add_condition(x, "snr10_1m", channel=0, seed=2026)
    b, meta_b = m.add_condition(x, "snr10_1m", channel=0, seed=2026)
    assert np.array_equal(a,b)
    assert 9.5 < meta_a["measured_snr_db"] < 10.5
    assert meta_a["measured_snr_db"] == meta_b["measured_snr_db"]


def test_synthetic_echo_is_finite_and_audible():
    x = synthetic()
    y, meta = m.add_condition(x, "echo_1m", channel=0, seed=2026)
    assert y.shape == x.shape and np.isfinite(y).all() and m.rms(y) > 0
    assert meta["rir_kind"] == "synthetic_3_echoes"
