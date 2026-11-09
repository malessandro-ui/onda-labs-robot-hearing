#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/ephemeral/onda-nvidia/IsaacLab}"
: "${ONDA_V017_TEST_WAV:?Set ONDA_V017_TEST_WAV to an absolute path to a 16kHz mono WAV file}"
: "${ONDA_BEATS_CHECKPOINT:?Set ONDA_BEATS_CHECKPOINT}"
: "${ONDA_BEATS_LABELS:?Set ONDA_BEATS_LABELS}"
: "${ONDA_BEATS_MODULE_DIR:?Set ONDA_BEATS_MODULE_DIR}"
for path in "$ONDA_V017_TEST_WAV" "$ONDA_BEATS_CHECKPOINT" "$ONDA_BEATS_LABELS" "$ONDA_BEATS_MODULE_DIR/BEATs.py"; do
  if [[ ! -f "$path" ]]; then echo "Missing required file: $path" >&2;exit 2;fi
done
mkdir -p "$ROOT/docs/validation/v017_native"
export PYTHONPATH="$ROOT/src:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
cd "$ISAACLAB_ROOT"
set +e
"$ISAACLAB_ROOT/isaaclab.sh" -p "$ROOT/tools/validate_v017_isaaclab_scene.py" \
  --wav "$ONDA_V017_TEST_WAV" \
  --checkpoint "$ONDA_BEATS_CHECKPOINT" \
  --labels "$ONDA_BEATS_LABELS" \
  --beats-dir "$ONDA_BEATS_MODULE_DIR" \
  --device "${ONDA_V017_DEVICE:-cuda}" \
  --duration-sim-s "${ONDA_V017_ISAAC_DURATION_S:-8}" \
  --out "$ROOT/docs/validation/v017_native/isaaclab_unified.json" \
  2>&1 | tee "$ROOT/docs/validation/v017_native/isaaclab_unified.log"
status=${PIPESTATUS[0]}
set -e
printf 'ONDA V017 ISAAC: process exit %s\n' "$status"
exit "$status"
