#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p docs/validation/streaming_v016_soak

echo '=== STREAMING SOAK REGRESSION TESTS ==='
python -m pytest -q tests/test_streaming_pcm.py tests/test_streaming_soak.py

echo '=== 30 MINUTES ACTUAL WALL-CLOCK PCM SOAK ==='
python tools/soak_streaming_v016.py \
  --duration-s "${ONDA_SOAK_DURATION_S:-1800}" \
  --checkpoint-s "${ONDA_SOAK_CHECKPOINT_S:-60}" \
  --output docs/validation/streaming_v016_soak/wall_clock_soak.json

echo '=== WALL-CLOCK SOAK COMPLETE ==='
python - <<'PY'
import json
from pathlib import Path
p=Path('docs/validation/streaming_v016_soak/wall_clock_soak.json')
x=json.loads(p.read_text())
print('result:',x['result'])
print('duration_s:',x.get('audio_duration_s'))
print('capture-to-publication P99:',x.get('timing',{}).get('capture_to_publication',{}).get('p99_ms'))
print('unique valid:',x.get('reads',{}).get('unique_valid'))
print('failures:',x.get('failures'))
PY
