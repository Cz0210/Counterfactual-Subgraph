#!/bin/bash
# Main Fresh-LoRA SFT route using strict-only Mutagenicity v2 targets.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mut_fresh_strict
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

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v2}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
TOKENIZER_FALLBACK_PATH="${TOKENIZER_FALLBACK_PATH:-$PROJECT_ROOT/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/sft_fresh_strict_v2}"
mkdir -p "$PROJECT_ROOT/logs"
echo "===== MUTAGENICITY FRESH STRICT-V2 SFT FULL ====="
echo "PROJECT_ROOT=$PROJECT_ROOT DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH TOKENIZER_FALLBACK_PATH=$TOKENIZER_FALLBACK_PATH"
echo "git_commit=$(git rev-parse HEAD || true)"
nvidia-smi || true

python scripts/train_mutagenicity_sft_fresh.py \
  --config configs/hpc.yaml \
  --config configs/train/mutagenicity_fresh_sft.yaml \
  --mode full \
  --dataset-variant strict_v2 \
  --data-root "$DATA_ROOT" \
  --base-model-path "$BASE_MODEL_PATH" \
  --tokenizer-fallback-path "$TOKENIZER_FALLBACK_PATH" \
  --output-root "$OUTPUT_ROOT"

python scripts/audit_mutagenicity_sft_fresh.py \
  --config configs/hpc.yaml \
  --output-root "$OUTPUT_ROOT" \
  --forbidden-adapter-checkpoint "$TOKENIZER_FALLBACK_PATH"
echo "[MUTAGENICITY_FRESH_SFT_FULL_OK]"
