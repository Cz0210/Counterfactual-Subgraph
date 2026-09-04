#!/usr/bin/env bash
set -euo pipefail

: "${T12_ACCELERATED_TASK_SPEC:?set the sealed accelerated task spec}"
: "${ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW:?set the user authorization to 1}"

if [[ "$ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW" != "1" ]]; then
  echo "T12 accelerated dispatch is not explicitly enabled" >&2
  exit 64
fi

PYTHON_BIN=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

exec "$PYTHON_BIN" -I -B \
  "$PROJECT_ROOT/scripts/autodl/run_t12_accelerated_from250_v1.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  owner --task-spec "$T12_ACCELERATED_TASK_SPEC"
