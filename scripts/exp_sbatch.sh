#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Slurm opens relative stdout/stderr paths before the batch script starts.
# Fresh worktrees do not carry the ignored logs directory, so create and
# validate it at the registered submission boundary.
mkdir -p "$PROJECT_ROOT/logs"
if [[ ! -d "$PROJECT_ROOT/logs" || ! -w "$PROJECT_ROOT/logs" ]]; then
  echo "[EXP_SUBMIT_ERROR] Slurm log directory is not writable: $PROJECT_ROOT/logs" >&2
  exit 2
fi

python scripts/exp_sbatch.py "$@"
