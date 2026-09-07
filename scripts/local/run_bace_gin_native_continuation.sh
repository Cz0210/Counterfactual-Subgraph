#!/usr/bin/env bash
# Mac-only finite continuation. CPU Slurm science uses the existing paired driver.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /Users/cz0210/miniconda3/envs/smiles_local/bin/python -I -B \
  "$SCRIPT_ROOT/scripts/local/run_bace_gin_native_continuation.py" "$@"
