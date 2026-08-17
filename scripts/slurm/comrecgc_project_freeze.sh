#!/bin/bash
#SBATCH --job-name=comrecgc_project_freeze
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE_ROOT="${BASE_ROOT:-}"
DATASET="${DATASET:-}"
STANDARDIZED_ROOT="${STANDARDIZED_ROOT:-}"
AUTOMATION_STATE="${AUTOMATION_STATE:-}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || { echo "[COMRECGC_CONFIG_ERROR] dataset=$DATASET" >&2; exit 2; }
[[ -n "$BASE_ROOT" && -n "$STANDARDIZED_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] explicit BASE_ROOT and STANDARDIZED_ROOT required" >&2; exit 2; }
[[ ! -e "$STANDARDIZED_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] standardized root exists=$STANDARDIZED_ROOT" >&2; exit 2; }
ARGS=()
[[ -n "$AUTOMATION_STATE" ]] && ARGS=(--automation-state "$AUTOMATION_STATE")
python scripts/baselines/comrecgc/freeze_recovery_result.py \
  --dataset "$DATASET" \
  --source-dir "$BASE_ROOT/unified_eval" \
  --gate-dir "$BASE_ROOT/full_gate" \
  --output-dir "$STANDARDIZED_ROOT" \
  "${ARGS[@]}"
test -s "$STANDARDIZED_ROOT/_FINALIZED.json"
echo "[COMRECGC_PROJECT_FREEZE_SUCCESS]"
