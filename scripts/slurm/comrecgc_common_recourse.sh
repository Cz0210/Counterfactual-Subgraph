#!/bin/bash
#SBATCH --job-name=comrecgc_recourse
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
export PYTHONHASHSEED=0
mkdir -p logs

DATASET="${DATASET:-}"
MODE="${MODE:-smoke}"
RESUME="${RESUME:-false}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 2
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || exit 2
if [[ "$MODE" == "smoke" ]]; then PARENT_LIMIT_EXPECTED=64; elif [[ "$DATASET" == "aids" ]]; then PARENT_LIMIT_EXPECTED=1283; else PARENT_LIMIT_EXPECTED=1448; fi
PARENT_LIMIT="${PARENT_LIMIT:-$PARENT_LIMIT_EXPECTED}"
[[ "$PARENT_LIMIT" == "$PARENT_LIMIT_EXPECTED" ]] || { echo "[COMRECGC_CONFIG_ERROR] parent_limit=$PARENT_LIMIT expected=$PARENT_LIMIT_EXPECTED" >&2; exit 2; }
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
GENERATION_DIR="${GENERATION_DIR:-$BASE_ROOT/generation}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/common_recourse}"
COMRECGC_EXPECTED_COMMIT="${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}"
COMRECGC_ROOT="${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}"
if [[ "$RESUME" != "true" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[COMRECGC_CONFIG_ERROR] non-empty output with RESUME=false: $OUTPUT_DIR" >&2; exit 2
fi
if [[ "$DATASET" == "aids" ]]; then
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}"
  SOURCE_CSV="${SOURCE_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}"
  SOURCE_ARGS=(--source-csv "$SOURCE_CSV")
else
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
  DISTANCE_CHECKPOINT="${DISTANCE_CHECKPOINT:-outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
  SOURCE_ARGS=()
fi
[[ -s "$GENERATION_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] generation incomplete" >&2; exit 2; }
python scripts/verify_comrecgc_checkout.py --config configs/hpc.yaml \
  --root "$COMRECGC_ROOT" --expected-commit "$COMRECGC_EXPECTED_COMMIT" \
  --validate-imports
echo "[COMRECGC_STAGE_CONFIG] stage=common_recourse dataset=$DATASET mode=$MODE output_dir=$OUTPUT_DIR"
RESUME_ARGS=(); [[ "$RESUME" == "true" ]] && RESUME_ARGS=(--resume)
python scripts/baselines/comrecgc/run_common_recourse.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --dataset "$DATASET" --mode "$MODE" --upstream-root "$COMRECGC_ROOT" \
  --dataset-dir "$DATASET_DIR" "${SOURCE_ARGS[@]}" --generation-dir "$GENERATION_DIR" \
  --distance-checkpoint "$DISTANCE_CHECKPOINT" --output-dir "$OUTPUT_DIR" \
  --parent-limit "$PARENT_LIMIT" --device cuda:0 "${RESUME_ARGS[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_COMMON_RECOURSE_SUCCESS]"
