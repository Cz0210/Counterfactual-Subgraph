#!/usr/bin/env bash
# One-shot, low-priority, read-only T14 convergence audit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${T14_CHECKPOINT_ROOT:?absolute T14 checkpoint root required}"
: "${T14_AUDIT_OUTPUT_ROOT:?fresh external audit root required}"
: "${T14_AUDITOR_EXECUTION_COMMIT:?immutable auditor commit required}"

exec nice -n 10 ionice -c 2 -n 7 \
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_t14_external_convergence_auditor_v1.py" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --checkpoint-root "$T14_CHECKPOINT_ROOT" \
  --output-root "$T14_AUDIT_OUTPUT_ROOT" \
  --execution-commit "$T14_AUDITOR_EXECUTION_COMMIT"
