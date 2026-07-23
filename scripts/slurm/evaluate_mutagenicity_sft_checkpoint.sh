#!/bin/bash
# Evaluate one Mutagenicity continued-SFT checkpoint on validation data.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=mut_sft_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] Could not determine PROJECT_ROOT" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
PWD_BEFORE_CD="$PWD"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a PEFT adapter checkpoint}"
FINAL_DATA_CSV="$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v1/mutagenicity_sft_val.csv"
FALLBACK_DATA_CSV="$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_ppo_data_v1/mutagenicity_sft_val.csv"
if [[ -z "${DATA_CSV:-}" ]]; then
  if [[ -f "$FINAL_DATA_CSV" ]]; then
    DATA_CSV="$FINAL_DATA_CSV"
  else
    DATA_CSV="$FALLBACK_DATA_CSV"
  fi
fi
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new evaluation directory}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-7}"

mkdir -p "$PROJECT_ROOT/logs"
echo "===== MUTAGENICITY SFT CHECKPOINT EVAL ====="
echo "hostname=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD before cd=$PWD_BEFORE_CD"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PWD after cd=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python=$(which python)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CHECKPOINT=$CHECKPOINT"
echo "DATA_CSV=$DATA_CSV"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_SAMPLES=$MAX_SAMPLES"
python scripts/evaluate_mutagenicity_sft_checkpoint.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --checkpoint "$CHECKPOINT" \
  --data-csv "$DATA_CSV" \
  --base-model-path "$BASE_MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --seed "$SEED"
