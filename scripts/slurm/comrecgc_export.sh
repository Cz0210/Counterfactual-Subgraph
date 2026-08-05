#!/bin/bash
#SBATCH --job-name=comrecgc_export
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
REQUIRE_TOP_K="${REQUIRE_TOP_K:-false}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || exit 2
BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/$DATASET/${MODE}_v1}"
COMMON_RECOURSE_DIR="${COMMON_RECOURSE_DIR:-$BASE_ROOT/common_recourse}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/export}"
if [[ "$DATASET" == "aids" ]]; then
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
  VOCAB_ARGS=(--atom-vocabulary-json outputs/hpc/gcfexplainer_hiv_csv/dataset/atom_vocab.json)
else
  TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
  VOCAB_ARGS=(--dataset-summary-json outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset/dataset_summary.json)
fi
REQUIRE_ARGS=(); [[ "$REQUIRE_TOP_K" == "true" ]] && REQUIRE_ARGS=(--require-top-k)
echo "[COMRECGC_STAGE_CONFIG] stage=export dataset=$DATASET mode=$MODE output_dir=$OUTPUT_DIR"
python scripts/baselines/comrecgc/export_candidates.py \
  --dataset "$DATASET" --common-recourse-dir "$COMMON_RECOURSE_DIR" \
  --teacher-path "$TEACHER_PATH" "${VOCAB_ARGS[@]}" --output-dir "$OUTPUT_DIR" \
  --top-k 20 "${REQUIRE_ARGS[@]}"
test -s "$OUTPUT_DIR/candidate_filter_audit.jsonl"
echo "[COMRECGC_EXPORT_SUCCESS]"
