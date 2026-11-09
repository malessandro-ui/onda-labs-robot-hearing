#!/usr/bin/env bash
set -euo pipefail
ROOT="${ONDA_REPO:-/ephemeral/onda-nvidia/onda-v015}"
ISAAC_ROOT="${ISAAC_LAB_ROOT:-/ephemeral/onda-nvidia/IsaacLab}"
VENV="${ONDA_VENV:-/ephemeral/onda-nvidia/env_isaaclab}"
cd "$ROOT"
source "$VENV/bin/activate"
mkdir -p docs/validation/streaming_v016
printf '\n=== OFFLINE STREAMING TESTS ===\n'
python -m pytest -q tests/test_streaming_pcm.py
printf '\n=== ESTIMATOR-ONLY PERFORMANCE ===\n'
python tools/benchmark_streaming.py --out docs/validation/streaming_v016/cpu_benchmark.json
printf '\n=== NATIVE ISAAC LAB PCM STREAMING ===\n'
cd "$ISAAC_ROOT"
set +e
timeout --signal=TERM --kill-after=10s 240s \
  ./isaaclab.sh -p "$ROOT/tools/native_isaaclab_streaming.py" \
  --out "$ROOT/docs/validation/streaming_v016/native_results.json" \
  2>&1 | tee "$ROOT/docs/validation/streaming_v016/native.log"
status=${PIPESTATUS[0]}
set -e
printf 'Native streaming exit status: %s\n' "$status"
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
python - "$ROOT/docs/validation/streaming_v016/native_results.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print('FINAL STREAMING VERDICT:',x.get('result'))
print('Native valid reads:',x.get('valid_read_counts'))
print('Native angular P75:',x.get('p75_angular_error_deg'))
if x.get('result')!='PASS':raise SystemExit(1)
PY
