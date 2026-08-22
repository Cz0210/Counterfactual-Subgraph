#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${ACTION:?Set ACTION to adopt-mut-ours, reexport-mut-ours-matched, freeze-mut-gcf-candidates, verify-mut-ours-adoption, or audit-inventory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh persistent output root}"

AUTODL_PYTHON="${AUTODL_PYTHON:-python}"
PROC_ROOT="${PROC_ROOT:-/proc}"

case "$ACTION" in
  adopt-mut-ours)
    : "${SOURCE_SPEC:?Set SOURCE_SPEC to the persistent legacy source specification}"
    exec "$AUTODL_PYTHON" scripts/autodl/run_am_legacy_standardization.py \
      --config configs/hpc.yaml \
      adopt-mut-ours \
      --source-spec "$SOURCE_SPEC" \
      --output-root "$OUTPUT_ROOT" \
      --proc-root "$PROC_ROOT"
    ;;
  freeze-mut-gcf-candidates)
    : "${SOURCE_SPEC:?Set SOURCE_SPEC to the persistent legacy source specification}"
    : "${MATCHED_THRESHOLD_CONTRACT:?Set MATCHED_THRESHOLD_CONTRACT to the matrix-audit Mutagenicity threshold contract}"
    exec "$AUTODL_PYTHON" scripts/autodl/run_am_legacy_standardization.py \
      --config configs/hpc.yaml \
      freeze-mut-gcf-candidates \
      --source-spec "$SOURCE_SPEC" \
      --matched-threshold-contract "$MATCHED_THRESHOLD_CONTRACT" \
      --output-root "$OUTPUT_ROOT" \
      --proc-root "$PROC_ROOT"
    ;;
  verify-mut-ours-adoption)
    : "${ADOPTED_MUT_OURS_ROOT:?Set ADOPTED_MUT_OURS_ROOT for verification}"
    exec "$AUTODL_PYTHON" scripts/autodl/run_am_legacy_standardization.py \
      --config configs/hpc.yaml \
      verify-mut-ours-adoption \
      --adopted-root "$ADOPTED_MUT_OURS_ROOT" \
      --output-root "$OUTPUT_ROOT"
    ;;
  reexport-mut-ours-matched)
    : "${ADOPTED_MUT_OURS_ROOT:?Set ADOPTED_MUT_OURS_ROOT to the strict original-protocol adoption}"
    : "${MATCHED_PROTOCOL:?Set MATCHED_PROTOCOL to the tracked 601-point protocol}"
    exec "$AUTODL_PYTHON" scripts/autodl/run_am_legacy_standardization.py \
      --config configs/hpc.yaml \
      reexport-mut-ours-matched \
      --adopted-root "$ADOPTED_MUT_OURS_ROOT" \
      --matched-protocol "$MATCHED_PROTOCOL" \
      --output-root "$OUTPUT_ROOT" \
      --proc-root "$PROC_ROOT"
    ;;
  audit-inventory)
    : "${SOURCE_SPEC:?Set SOURCE_SPEC to the persistent legacy source specification}"
    : "${ADOPTED_MUT_OURS_ROOT:?Set ADOPTED_MUT_OURS_ROOT for audit-inventory}"
    exec "$AUTODL_PYTHON" scripts/autodl/run_am_legacy_standardization.py \
      --config configs/hpc.yaml \
      audit-inventory \
      --source-spec "$SOURCE_SPEC" \
      --output-root "$OUTPUT_ROOT" \
      --adopted-mut-ours-root "$ADOPTED_MUT_OURS_ROOT"
    ;;
  *)
    echo "Unsupported ACTION=$ACTION" >&2
    exit 2
    ;;
esac
