#!/bin/bash
#SBATCH --job-name=comrecgc_standardize
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper required by the repository workflow.  The current
# campaign is AutoDL-only; this file is syntax-checked but is not submitted.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

: "${DATASET:?DATASET is required}"
: "${SOURCE_GENERATION_ROOT:?SOURCE_GENERATION_ROOT is required}"
: "${COMRECGC_UPSTREAM_ROOT:?COMRECGC_UPSTREAM_ROOT is required}"
: "${DATASET_DIR:?DATASET_DIR is required}"
: "${DISTANCE_CHECKPOINT:?DISTANCE_CHECKPOINT is required}"
: "${DATASET_CSV:?DATASET_CSV is required}"
: "${TEACHER_PATH:?TEACHER_PATH is required}"
: "${MOLCLR_ROOT:?MOLCLR_ROOT is required}"
: "${MOLCLR_CHECKPOINT:?MOLCLR_CHECKPOINT is required}"
: "${THRESHOLDS_PATH:?THRESHOLDS_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset "$DATASET"
  --source-generation-root "$SOURCE_GENERATION_ROOT"
  --upstream-root "$COMRECGC_UPSTREAM_ROOT"
  --dataset-dir "$DATASET_DIR"
  --distance-checkpoint "$DISTANCE_CHECKPOINT"
  --dataset-csv "$DATASET_CSV"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --thresholds-path "$THRESHOLDS_PATH"
  --output-root "$OUTPUT_ROOT"
  --device cuda:0
)
if [[ "$DATASET" == "aids" ]]; then
  : "${SOURCE_CSV:?SOURCE_CSV is required for AIDS}"
  args+=(--source-csv "$SOURCE_CSV" --theta-star "${THETA_STAR:-0.05}" --cost-cap "${COST_CAP:-0.0535}")
fi

python scripts/autodl/run_comrecgc_standardized_continuation.py "${args[@]}"
