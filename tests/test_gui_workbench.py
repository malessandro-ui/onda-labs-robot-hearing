import base64
from contextlib import closing
from io import BytesIO
import wave

import numpy as np
import pytest

from onda_hearing.gui import backend
from onda_hearing.gui.backend import GuiInputError, build_scene, capabilities, simulate_scenario
from onda_hearing.gui.live import CHUNK_SAMPLES, live_frames
from onda_hearing.semantics import EventScore


@pytest.fixture(autouse=True)
def disable_optional_beats(monkeypatch):
    monkeypatch.delenv("ONDA_GUI_ALLOW_TRUSTED_BEATS", raising=False)


def scenario(**overrides):
    value = {
        "duration_s": 1.0,
        "noise_std": 0.001,
        "mic_preset": "onda_reference_4",
        "room": "direct",
        "processing": "raw",
        "emitters": [
            {
                "id": "speaker",
                "sound": "broadband",
                "bearing_deg": 38,
                "distance_m": 2,
                "gain_db": -3,
                "onset_s": 0,
                "duration_s": None,
                "loop": True,
                "directivity": "omni",
                "enabled": True,
            }
        ],
    }
    value.update(overrides)
    return value


def test_capabilities_are_honest_about_waveformer():
    data = capabilities()
    backends = {item["id"]: item for item in data["enhancement_backends"]}
    assert backends["raw"]["available"] is True
    assert backends["waveformer"]["available"] is False


def test_gui_runs_real_scene_renderer_and_localizer():
    result = simulate_scenario(scenario())
    assert result["renderer"] == "Onda v0.17 AcousticSceneRenderer"
    assert result["samples_per_channel"] == 16_000
    assert result["channels"] == 4
    assert result["spatial"]["valid"] is True
    assert result["spatial"]["evaluation_error_deg"] <= 6
    assert result["policy"]["values"][3] == pytest.approx(1.)
    assert result["semantic"]["status"] == "not_configured"
    assert result["semantic"]["association_valid"] is False
    assert result["telemetry"]["overrange_samples"] == 0
    assert len(result["audio"]["waveform"]) == 4
    assert result["audio"]["wav_base64"].startswith("UklG")


def test_silence_is_invalid_and_finite():
    result = simulate_scenario(scenario(noise_std=0., emitters=[]))
    assert result["spatial"]["valid"] is False
    assert result["spatial"]["bearing_deg"] is None
    assert result["policy"]["values"] == pytest.approx(np.zeros(5))
    assert result["audio"]["peak"] == 0.


def test_multiple_sources_never_claim_truth_association():
    payload = scenario()
    payload["emitters"].append({
        "id": "alarm", "sound": "alarm", "bearing_deg": -55,
        "distance_m": 2.5, "gain_db": -8, "onset_s": 0,
        "duration_s": None, "loop": True, "directivity": "omni", "enabled": True,
    })
    result = simulate_scenario(payload)
    assert result["spatial"]["association_valid"] is False
    assert result["spatial"]["evaluation_truth_deg"] is None
    assert result["spatial"]["evaluation_error_deg"] is None


def test_unavailable_processing_backend_fails_closed():
    with pytest.raises(GuiInputError, match="Waveformer"):
        simulate_scenario(scenario(processing="waveformer"))


def test_configured_sound_brain_events_remain_unassociated(monkeypatch):
    monkeypatch.setattr(backend, "_beats_capability", lambda: {
        "label": "Onda Sound Brain · BEATs", "available": True,
    })

    class Brain:
        def classify(self, mono, sample_rate):
            return (EventScore("Telephone bell ringing", .87),)

    monkeypatch.setattr(backend, "_beats_brain", lambda: Brain())
    result = simulate_scenario(scenario())
    assert result["semantic"]["valid"] is True
    assert result["semantic"]["events"] == [
        {"label": "Telephone bell ringing", "score": pytest.approx(.87)}
    ]
    assert result["semantic"]["window_scope"] == "whole_audio_window"
    assert result["semantic"]["association_valid"] is False


def test_scene_input_bounds_are_enforced():
    with pytest.raises(GuiInputError, match="duration_s"):
        simulate_scenario(scenario(duration_s=100))
    with pytest.raises(GuiInputError, match="duplicate emitter"):
        payload = scenario()
        payload["emitters"].append(dict(payload["emitters"][0]))
        simulate_scenario(payload)


def test_uploaded_sound_is_bound_to_the_object_and_reaches_the_microphones():
    source = BytesIO()
    with wave.open(source, "wb") as wav:
        wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        wav.writeframes(np.full(1600, 8192, dtype="<i2").tobytes())
    payload = scenario(noise_std=0.)
    payload["emitters"][0].update(
        sound="wav", wav_name="fixture.wav", wav_base64=base64.b64encode(source.getvalue()).decode(),
        loop=False,
    )
    normalized, renderer, _ = build_scene(payload)
    emitter = renderer.emitters[0]
    assert emitter.binding.object_path is not None
    assert np.all(renderer.assets[emitter.asset_id].samples == .25)
    assert "wav_base64" not in normalized["emitters"][0]
    pcm = renderer.render(0, 4096)
    assert np.max(pcm) > 0.
    assert not np.any(pcm[-2048:])

    payload["emitters"][0]["wav_base64"] = "invalid"
    with pytest.raises(GuiInputError, match="Invalid WAV"):
        build_scene(payload)


def test_live_stream_advances_and_exposes_monitor_pcm_without_inventing_a_bearing():
    _, renderer, positions = build_scene(scenario(noise_std=0., emitters=[]))
    with closing(live_frames(renderer, positions)) as stream:
        first, second = next(stream), next(stream)
    assert second["time_s"] - first["time_s"] == pytest.approx(CHUNK_SAMPLES / 16_000)
    assert renderer.next_sample == 2 * CHUNK_SAMPLES
    assert second["spatial"]["bearing_deg"] is None
    assert second["semantic"] is None
    assert base64.b64decode(second["audio"]["pcm16_base64"]) == bytes(2 * CHUNK_SAMPLES)
