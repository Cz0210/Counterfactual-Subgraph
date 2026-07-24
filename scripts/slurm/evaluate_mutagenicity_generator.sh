#!/bin/bash
# Unified task-level generator evaluation on all 260 Mutagenicity val parents.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=mut_gen_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"; fi
[[ -n "$PROJECT_ROOT" ]] || { echo "[ERROR] Could not determine PROJECT_ROOT" >&2; exit 2; }
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

VAL_CSV="${VAL_CSV:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v2/mutagenicity_ppo_prompts_val_label1_v2.csv}"
TEACHER_PATH="${TEACHER_PATH:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/hpc/mutagenicity/eval/generator_fresh_sft_checkpoints}"
MODELS="${MODELS:-Pure-ChemLLM=PURE_BASE}"
CANDIDATE_COUNTS="${CANDIDATE_COUNTS:-1,4,8}"
SEEDS="${SEEDS:-7,17,27,37,47,57,67,77}"
mkdir -p "$PROJECT_ROOT/logs"

MODEL_ARGS=()
IFS=';' read -r -a MODEL_SPECS <<< "$MODELS"
for spec in "${MODEL_SPECS[@]}"; do
  [[ -n "$spec" ]] && MODEL_ARGS+=(--model "$spec")
done

echo "===== MUTAGENICITY GENERATOR TASK EVAL ====="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VAL_CSV=$VAL_CSV"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "MODELS=$MODELS"
echo "CANDIDATE_COUNTS=$CANDIDATE_COUNTS"
echo "SEEDS=$SEEDS"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "git_commit=$(git rev-parse HEAD || true)"
nvidia-smi || true

python scripts/evaluate_mutagenicity_generator.py \
  --config configs/hpc.yaml \
  "${MODEL_ARGS[@]}" \
  --val-csv "$VAL_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --base-model-path "$BASE_MODEL_PATH" \
  --candidate-counts "$CANDIDATE_COUNTS" \
  --seeds "$SEEDS" \
  --output-dir "$OUTPUT_DIR"

test -s "$OUTPUT_DIR/best_task_checkpoint.json"
echo "[MUTAGENICITY_GENERATOR_EVAL_OK]"
