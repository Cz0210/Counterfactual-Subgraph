#!/usr/bin/env bash
# Mac control only; never launch through HPC or without the external volume.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
exec "${LOCAL_RELAY_PYTHON:-python3}" -B "$SCRIPT_ROOT/scripts/local/run_gnn_seed7_corrective_relay.py" "$@"
