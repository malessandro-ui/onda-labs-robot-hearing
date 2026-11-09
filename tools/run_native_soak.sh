#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/ephemeral/onda-nvidia/IsaacLab}"
mkdir -p "$ROOT/docs/validation/streaming_v016_soak"
export PYTHONPATH="$ROOT/src:$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
cd "$ISAACLAB_ROOT"
set +e
# Isaac Kit's known shutdown stall is handled by the finite native harness.
"$ISAACLAB_ROOT/isaaclab.sh" -p "$ROOT/tools/native_isaaclab_soak_v016.py" \
  --duration-sim-s "${ONDA_NATIVE_SOAK_DURATION_S:-180}" \
  --overload-ms "${ONDA_NATIVE_SOAK_OVERLOAD_MS:-0}" \
  --out "$ROOT/docs/validation/streaming_v016_soak/native_motion_soak.json" \
  2>&1 | tee "$ROOT/docs/validation/streaming_v016_soak/native_motion_soak.log"
status=${PIPESTATUS[0]}
set -e
printf 'ONDA NATIVE SOAK: process exit %s\n' "$status"
exit "$status"
