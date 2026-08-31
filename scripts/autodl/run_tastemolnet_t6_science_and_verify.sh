#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${AUTODL_PYTHON:?AUTODL_PYTHON is required}"
: "${TASTEMOLNET_T6_OUTPUT:?TASTEMOLNET_T6_OUTPUT is required}"
: "${TASTEMOLNET_T6_VERIFICATION_OUTPUT:?TASTEMOLNET_T6_VERIFICATION_OUTPUT is required}"

"$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/train_tastemolnet_gnn_ppo.py" "$@"

# This is a new process after the trainer has closed all writers and committed
# PASS. It never mutates the scientific root.
exec "$AUTODL_PYTHON" -B \
  "$PROJECT_ROOT/scripts/autodl/verify_tastemolnet_ours_ppo_smoke.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  --science-root "$TASTEMOLNET_T6_OUTPUT" \
  --verification-root "$TASTEMOLNET_T6_VERIFICATION_OUTPUT"
