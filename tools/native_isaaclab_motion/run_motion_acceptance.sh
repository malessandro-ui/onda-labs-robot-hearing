#!/usr/bin/env bash
# Onda v0.15 / native Isaac Lab rotating array regression on Shadeform RTX4090.
set -Eeuo pipefail
ROOT="${ONDA_WORKSPACE:-/ephemeral/onda-nvidia}"
SDK="${ONDA_REPO:-$ROOT/onda-v015}"
LAB="${ISAAC_LAB_ROOT:-$ROOT/IsaacLab}"
VENV="${ISAAC_LAB_VENV:-$ROOT/env_isaaclab}"
RUNNER="$SDK/tools/native_isaaclab_motion/native_motion_acceptance.py"
RESULTS="$SDK/docs/validation/native_isaaclab_motion_v015"
for required in "$LAB/isaaclab.sh" "$VENV/bin/python" "$RUNNER" "$SDK/examples/v015_isaac_lab_task_integration.py"; do
  if [[ ! -f "$required" ]]; then printf 'ERROR: missing %s\n' "$required" >&2; exit 2; fi
done
current_branch="$(git -C "$SDK" branch --show-current)"
if [[ "$current_branch" != "develop/v0.15-unified-hearing" ]]; then
  echo "ERROR: Expected develop/v0.15-unified-hearing; currently $current_branch" >&2
  exit 2
fi
mkdir -p "$RESULTS"
source "$VENV/bin/activate"
echo "=== SDK VERSION ==="
git -C "$SDK" log -1 --oneline
printf '%s\n' '=== CPU QUATERNION REGRESSION ==='
cd "$SDK"
"$VENV/bin/python" -m pytest -q tests/test_native_motion_math.py
printf '%s\n' '=== NATIVE ISAAC LAB MOTION ACCEPTANCE ==='
# Do not accidentally pass due to an old results.json when the run times out.
rm -f -- "$RESULTS/results.json"
cd "$LAB"
set +e
timeout --signal=TERM --kill-after=10s 240s \
  "$LAB/isaaclab.sh" -p "$RUNNER" \
  --repo "$SDK" --out "$RESULTS" --shutdown hard "$@" \
  2>&1 | tee "$RESULTS/motion.log"
status=${PIPESTATUS[0]}
set -e
printf 'Native motion process exit status: %s\n' "$status"
if [[ "$status" -ne 0 ]]; then
  echo "FINAL MOTION VERDICT: FAIL (runner exit code $status)" >&2
  exit "$status"
fi
"$VENV/bin/python" - "$RESULTS/results.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    sys.exit('FINAL MOTION VERDICT: FAIL (missing results.json)')
report = json.loads(path.read_text())
if report.get('result') != 'PASS' or len(report.get('phases', [])) != 4:
    sys.exit('FINAL MOTION VERDICT: FAIL (incomplete/non-PASS report)')
print('=== MOTION PHASE RESULTS ===')
for phase in report['phases']:
    print(f"Rig yaw {phase['target_yaw_deg']:+.0f}°:", ', '.join(
        f"env{x['env_id']} bearing={x['inferred_local_bearing_last_deg']:.1f}° "
        f"ref={x['expected_local_bearing_last_deg']:.1f}° "
        f"P75error={x['p75_error_deg']:.2f}°"
        for x in phase['environments']))
print('=== LATENCY ===')
for key, value in report.get('timing', {}).items():
    if 'p50' in key or 'p95' in key:
        print(f'{key}: {value:.2f}')
print('FINAL MOTION VERDICT: PASS')
PY
