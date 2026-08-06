#!/bin/bash
#SBATCH --job-name=comrecgc_project_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
MODE="${MODE:-full}"
BASE_ROOT="${BASE_ROOT:-}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || { echo "[COMRECGC_CONFIG_ERROR] dataset=$DATASET" >&2; exit 2; }
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || { echo "[COMRECGC_CONFIG_ERROR] mode=$MODE" >&2; exit 2; }
[[ -n "$BASE_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] explicit BASE_ROOT required" >&2; exit 2; }
CHEMISTRY_DIR="${CHEMISTRY_DIR:-$BASE_ROOT/chemistry}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/unified_eval}"
MOLCLR_ROOT="${MOLCLR_ROOT:-pretrained_models/MolCLR}"
MOLCLR_CHECKPOINT="${MOLCLR_CHECKPOINT:-pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}"
if [[ "$DATASET" == "aids" ]]; then
  DATASET_CSV="${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
  THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10/wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv}"
  THRESHOLD_ARGS=(--theta-star 0.05 --cost-cap 0.0535)
  [[ "$MODE" == "full" ]] && EXPECTED_PARENT_COUNT=1283 || EXPECTED_PARENT_COUNT=16
else
  if [[ "$MODE" == "full" ]]; then
    DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/test_source_label1_teacher_correct.csv}"
    EXPECTED_PARENT_COUNT=217
  else
    DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/train_source_label1_teacher_correct.csv}"
    EXPECTED_PARENT_COUNT=16
  fi
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
  THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json}"
  THRESHOLD_ARGS=()
fi
for input in "$CHEMISTRY_DIR/run_manifest.json" "$CHEMISTRY_DIR/medoid_validity.csv" "$DATASET_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$THRESHOLDS_JSON"; do
  [[ -s "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "[COMRECGC_CONFIG_ERROR] output exists=$OUTPUT_DIR" >&2; exit 2; }
echo "[COMRECGC_STAGE_CONFIG] stage=project_slot_eval dataset=$DATASET mode=$MODE parents=$EXPECTED_PARENT_COUNT output=$OUTPUT_DIR"
python scripts/baselines/comrecgc/run_slot_unified_eval.py \
  --dataset "$DATASET" \
  --mode "$MODE" \
  --chemistry-dir "$CHEMISTRY_DIR" \
  --dataset-csv "$DATASET_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --thresholds-json "$THRESHOLDS_JSON" \
  "${THRESHOLD_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --max-k 20 \
  --device cuda
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_PROJECT_SLOT_EVAL_SUCCESS]"
