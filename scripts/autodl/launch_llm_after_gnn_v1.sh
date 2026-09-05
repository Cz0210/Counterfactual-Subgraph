#!/usr/bin/env bash
set -euo pipefail
gnn_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$gnn_project_root"
# Existing outer queues can dispatch one next variant through the repaired
# AutoDL owner. No new scheduler, GPU borrowing, or reservation authority.
if [[ "${1:-}" == "--owner-dispatch" ]]; then
  shift
  exec "${AUTODL_PYTHON:-python}" -I -B scripts/autodl/gpu_lock.py \
    --project-root "$gnn_project_root" --config configs/hpc.yaml run "$@"
fi
# Forward the native successor CLI. Accepted corrective proof can be reused
# without reopening the archive, while actual resource sources stay fresh.
# --plan-only reports the next L1/L2/L3 task without taking resources.
exec "${AUTODL_PYTHON:-python}" -I -B scripts/ablations/llm/run_bace_llm_successor.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
