#!/usr/bin/env bash
# Mac control only: never submit this relay with Slurm or run it on HPC.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
exec "${LOCAL_RELAY_PYTHON:-python3}" "$SCRIPT_ROOT/scripts/local/run_gnn_seed7_verified_relay.py" "$@"
