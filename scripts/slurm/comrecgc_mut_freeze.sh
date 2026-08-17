#!/bin/bash
#SBATCH --job-name=comrecgc_mut_freeze
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
[[ -n "${BASE_ROOT:-}" ]] || { echo "[COMRECGC_CONFIG_ERROR] freeze requires explicit BASE_ROOT" >&2; exit 2; }
[[ -n "${STANDARDIZED_ROOT:-}" ]] || { echo "[COMRECGC_CONFIG_ERROR] freeze requires explicit STANDARDIZED_ROOT" >&2; exit 2; }
SOURCE_DIR="${SOURCE_DIR:-$BASE_ROOT/unified_eval}"
GATE_DIR="${GATE_DIR:-$BASE_ROOT/full_gate}"
AUTOMATION_STATE="${AUTOMATION_STATE:-}"
[[ ! -e "$STANDARDIZED_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] standardized root exists: $STANDARDIZED_ROOT" >&2; exit 2; }
ARGS=()
[[ -n "$AUTOMATION_STATE" ]] && ARGS=(--automation-state "$AUTOMATION_STATE")
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_freeze source=$SOURCE_DIR output=$STANDARDIZED_ROOT"
python scripts/baselines/comrecgc/freeze_recovery_result.py \
  --source-dir "$SOURCE_DIR" \
  --gate-dir "$GATE_DIR" \
  --output-dir "$STANDARDIZED_ROOT" \
  "${ARGS[@]}"
test -s "$STANDARDIZED_ROOT/_FINALIZED.json"
echo "[COMRECGC_MUT_FREEZE_SUCCESS]"
