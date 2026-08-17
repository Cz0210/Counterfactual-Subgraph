#!/bin/bash
#SBATCH --job-name=comrecgc_project_gate
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
DATASET="${DATASET:-}"
BASE_ROOT="${BASE_ROOT:-}"
EXPECTED_PROJECT_COMMIT="${EXPECTED_PROJECT_COMMIT:-}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || { echo "[COMRECGC_CONFIG_ERROR] dataset=$DATASET" >&2; exit 2; }
[[ -n "$BASE_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] explicit BASE_ROOT required" >&2; exit 2; }
[[ -n "$EXPECTED_PROJECT_COMMIT" ]] || { echo "[COMRECGC_CONFIG_ERROR] expected project commit required" >&2; exit 2; }
if [[ "$DATASET" == "aids" ]]; then
  EXPECTED_PARENT_COUNT=1283
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
else
  EXPECTED_PARENT_COUNT=217
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
fi
EXPECTED_TEACHER_SHA256="$(sha256sum "$TEACHER_PATH" | awk '{print $1}')"
INPUT_DIR="${INPUT_DIR:-$BASE_ROOT/unified_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/full_gate}"
python scripts/baselines/comrecgc/gate_recovery.py \
  --stage project-full \
  --dataset "$DATASET" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --expected-teacher-sha256 "$EXPECTED_TEACHER_SHA256" \
  --expected-project-commit "$EXPECTED_PROJECT_COMMIT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_PROJECT_FULL_GATE_PASS]"
