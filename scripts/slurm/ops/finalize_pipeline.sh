#!/usr/bin/env bash
# Final report renderer. Submit this only through scripts/exp_sbatch.sh.
#SBATCH --job-name=automation_finalize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${AUTOMATION_CONDA_ENV:-smiles_pip118}"
set -u

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${AUTOMATION_RUN_DIR:?AUTOMATION_RUN_DIR is required}"
: "${AUTOMATION_SPEC:?AUTOMATION_SPEC is required}"
: "${AUTOMATION_RUN_ID:?AUTOMATION_RUN_ID is required}"
if [[ ! -s "$AUTOMATION_RUN_DIR/state.json" ]]; then
  python scripts/ops/experimentctl.py initialize-run \
    --spec "$AUTOMATION_SPEC" \
    --run-dir "$AUTOMATION_RUN_DIR" \
    --run-id "$AUTOMATION_RUN_ID" \
    --project-root "$PROJECT_ROOT"
fi
python scripts/ops/experimentctl.py finalize --run-dir "$AUTOMATION_RUN_DIR"
