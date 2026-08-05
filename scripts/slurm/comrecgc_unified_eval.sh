#!/bin/bash
#SBATCH --job-name=comrecgc_eval
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
mkdir -p logs

DATASET="${DATASET:-}"
MODE="${MODE:-smoke}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 2
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
CANDIDATES_CSV="${CANDIDATES_CSV:-$BASE_ROOT/export/selected_top20.csv}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-$BASE_ROOT/export/frozen_candidate_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/eval}"
if [[ "$DATASET" == "aids" ]]; then
  DATASET_CSV="${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
  THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_ours_top20_final/run_config.json}"
  EXPECTED_PARENT_COUNT=1283
else
  DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/test_source_label1_teacher_correct.csv}"
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
  THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json}"
  EXPECTED_PARENT_COUNT=217
fi
echo "[COMRECGC_STAGE_CONFIG] stage=unified_eval dataset=$DATASET mode=$MODE output_dir=$OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/run_unified_eval.py \
  --dataset "$DATASET" --mode "$MODE" \
  --candidates-csv "$CANDIDATES_CSV" --candidate-manifest "$CANDIDATE_MANIFEST" \
  --dataset-csv "$DATASET_CSV" --teacher-path "$TEACHER_PATH" \
  --molclr-root pretrained_models/MolCLR \
  --molclr-checkpoint pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth \
  --thresholds-json "$THRESHOLDS_JSON" --output-dir "$OUTPUT_DIR" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT"
test -s "$OUTPUT_DIR/_COMRECGC_EVAL_COMPLETE.json"
echo "[COMRECGC_UNIFIED_EVAL_SUCCESS]"
