import numpy as np
import pytest

from onda_hearing.gui.backend import GuiInputError, capabilities, simulate_scenario


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


def test_scene_input_bounds_are_enforced():
    with pytest.raises(GuiInputError, match="duration_s"):
        simulate_scenario(scenario(duration_s=100))
    with pytest.raises(GuiInputError, match="duplicate emitter"):
        payload = scenario()
        payload["emitters"].append(dict(payload["emitters"][0]))
        simulate_scenario(payload)
