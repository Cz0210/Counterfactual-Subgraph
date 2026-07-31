#!/usr/bin/env bash
# Generic HPC-side stage runner. Submit this only through scripts/exp_sbatch.sh.
#SBATCH --job-name=automation_stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
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
: "${AUTOMATION_STAGE_ID:?AUTOMATION_STAGE_ID is required}"
: "${AUTOMATION_SPEC:?AUTOMATION_SPEC is required}"
: "${AUTOMATION_RUN_ID:?AUTOMATION_RUN_ID is required}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "AUTOMATION_RUN_DIR=$AUTOMATION_RUN_DIR"
echo "AUTOMATION_STAGE_ID=$AUTOMATION_STAGE_ID"
echo "git_commit=$(git rev-parse HEAD)"
if [[ ! -s "$AUTOMATION_RUN_DIR/state.json" ]]; then
  python scripts/ops/experimentctl.py initialize-run \
    --spec "$AUTOMATION_SPEC" \
    --run-dir "$AUTOMATION_RUN_DIR" \
    --run-id "$AUTOMATION_RUN_ID" \
    --project-root "$PROJECT_ROOT"
fi
python scripts/ops/experimentctl.py execute-stage \
  --run-dir "$AUTOMATION_RUN_DIR" \
  --stage "$AUTOMATION_STAGE_ID" \
  --project-root "$PROJECT_ROOT"
