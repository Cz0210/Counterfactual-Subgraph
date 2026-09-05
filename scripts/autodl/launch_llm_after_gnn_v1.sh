#!/usr/bin/env bash
set -euo pipefail
gnn_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$gnn_project_root"
# Forward the explicit native successor CLI. It reopens the corrected archive,
# then execs under the existing owner's held FD; no new GPU lease is created.
# --plan-only reports the next L1/L2/L3 task without taking resources.
exec "${AUTODL_PYTHON:-python}" -I -B scripts/ablations/llm/run_bace_llm_successor.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
