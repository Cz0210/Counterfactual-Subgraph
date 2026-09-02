#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_PYTHON:=/root/miniconda3/envs/smiles_pip118/bin/python}"
: "${TASTE_GLOBALGCE_ZERO_COMMAND:=finalize}"
: "${TASTE_GLOBALGCE_SOURCE_ROOT:?set TASTE_GLOBALGCE_SOURCE_ROOT}"
: "${TASTE_GLOBALGCE_ATTEMPT_RECEIPT:?set TASTE_GLOBALGCE_ATTEMPT_RECEIPT}"
: "${TASTE_GLOBALGCE_ZERO_AUTHORIZATION:?set TASTE_GLOBALGCE_ZERO_AUTHORIZATION}"
: "${TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT:?set TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT}"

COMMON=(
  --source-root "$TASTE_GLOBALGCE_SOURCE_ROOT"
  --attempt-receipt "$TASTE_GLOBALGCE_ATTEMPT_RECEIPT"
  --authorization-receipt "$TASTE_GLOBALGCE_ZERO_AUTHORIZATION"
  --execution-commit "$TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT"
)

if [[ "$TASTE_GLOBALGCE_ZERO_COMMAND" == authorize ]]; then
  exec "$AUTODL_PYTHON" -I -B scripts/autodl/run_tastemolnet_globalgce_valid_zero_finalizer_v1.py \
    authorize "${COMMON[@]}"
fi
if [[ "$TASTE_GLOBALGCE_ZERO_COMMAND" != finalize ]]; then
  echo "TASTE_GLOBALGCE_ZERO_COMMAND must be authorize or finalize" >&2
  exit 2
fi

: "${TASTE_GLOBALGCE_ZERO_OBSERVATION:?set TASTE_GLOBALGCE_ZERO_OBSERVATION}"
: "${TASTE_GLOBALGCE_TEST_CSV:?set TASTE_GLOBALGCE_TEST_CSV}"
: "${TASTE_GLOBALGCE_THRESHOLD_CONTRACT:?set TASTE_GLOBALGCE_THRESHOLD_CONTRACT}"
: "${TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT:?set TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT}"

exec "$AUTODL_PYTHON" -I -B scripts/autodl/run_tastemolnet_globalgce_valid_zero_finalizer_v1.py \
  finalize "${COMMON[@]}" \
  --recovery-observation "$TASTE_GLOBALGCE_ZERO_OBSERVATION" \
  --test-csv "$TASTE_GLOBALGCE_TEST_CSV" \
  --threshold-contract "$TASTE_GLOBALGCE_THRESHOLD_CONTRACT" \
  --output-root "$TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT"
