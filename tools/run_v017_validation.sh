#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q tests/test_acoustic_scene_v017.py tests/test_semantics_v017.py
python -m pytest -q tests/test_streaming_pcm.py tests/test_streaming_soak.py
echo 'Onda v0.17 Python tests passed; native simulator and BEATs validation still required.'
