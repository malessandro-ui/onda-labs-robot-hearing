#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ONDA_VENV:-/ephemeral/onda-nvidia/env_isaaclab}"
ISAACLAB="${ISAACLAB_ROOT:-/ephemeral/onda-nvidia/IsaacLab}"
OUT="$REPO/docs/validation/native_isaaclab_realtime_v015"
mkdir -p "$OUT"
source "$VENV/bin/activate"
cd "$REPO"
echo '=== ONDA SDK COMMIT ==='
git log -1 --oneline
echo '=== ASYNC WORKER REGRESSION ==='
python -m pytest -q tests/test_async_isaac_lab_worker.py
echo '=== CPU ACOUSTIC PROFILE COMPARISON ==='
python tools/native_isaaclab_realtime/benchmark_acoustic.py \
  --steps "${ONDA_PROFILE_STEPS:-100}" \
  --output "$OUT/profiles.json" 2>&1 | tee "$OUT/profile.log"
echo '=== NATIVE ISAAC LAB ASYNC SMOKE ==='
cd "$ISAACLAB"
set +e
timeout --signal=TERM --kill-after=10s "${ONDA_NATIVE_TIMEOUT_S:-240}" \
  ./isaaclab.sh -p "$REPO/tools/native_isaaclab_realtime/native_async_smoke.py" \
  --repo "$REPO" --out "$OUT/async_smoke.json" --shutdown hard \
  2>&1 | tee "$OUT/native_async.log"
status=${PIPESTATUS[0]}
set -e
echo "Native async exit status: $status"
if [[ "$status" -ne 0 ]]; then
  echo 'FINAL REALTIME VALIDATION: FAIL'
  exit "$status"
fi
printf '\n=== BENCHMARK ASSESSMENT ===\n'
python - "$OUT/profiles.json" "$OUT/async_smoke.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
a=json.load(open(sys.argv[2]))
for name,v in p['profiles'].items():
 print(f"{name}: P50={v['timing']['p50_ms']:.2f} ms, P95={v['timing']['p95_ms']:.2f} ms; angle P75={v['p75_angular_error_deg']}; accuracy={v['accuracy_pass']}")
for name,v in p['candidate_assessment'].items():
 print(f"{name}: speedup={v['relative_p50_speedup']:.2f}x; eligible for native revalidation={v['eligible_for_further_validation']}")
print('async native:',a['result'], 'dropped:',a['telemetry']['dropped_frames'], 'resyncs:',a['telemetry']['resynchronizations'])
PY
echo 'FINAL REALTIME VALIDATION: PASS (safety/integration only; NOT 60Hz throughput certification)'
