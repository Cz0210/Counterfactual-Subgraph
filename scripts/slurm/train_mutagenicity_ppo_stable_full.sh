#!/bin/bash
#SBATCH --job-name=mut_ppo_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118
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
if [[ ! -f "$PROJECT_ROOT/scripts/train_mutagenicity_ppo_stable.py" ]]; then
  echo "[ERROR] Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi

echo "PWD before cd=$PWD"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$PROJECT_ROOT/logs"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v1/mutagenicity_ppo_prompts_train_label1.csv}"
VAL_CSV="${VAL_CSV:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_ppo_data_v1/mutagenicity_ppo_prompts_val_label1.csv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
POLICY_ADAPTER_CHECKPOINT="${POLICY_ADAPTER_CHECKPOINT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_continued_v1_best}"
TOKENIZER_FALLBACK_PATH="${TOKENIZER_FALLBACK_PATH:-$PROJECT_ROOT/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500}"
TEACHER_PATH="${TEACHER_PATH:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/ppo_stable_v1}"
SOURCE_LABEL="${SOURCE_LABEL:-1}"
TARGET_LABEL="${TARGET_LABEL:-0}"
MAX_PARENTS="${MAX_PARENTS:-1448}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
SAMPLES_PER_UPDATE="$ROLLOUT_BATCH_SIZE"
UPDATES_PER_EPOCH=$(( (MAX_PARENTS + SAMPLES_PER_UPDATE - 1) / SAMPLES_PER_UPDATE ))
MAX_UPDATES="${MAX_UPDATES:-$UPDATES_PER_EPOCH}"
SEED="${SEED:-7}"
SAVE_STEPS="${SAVE_STEPS:-5}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-5}"

mkdir -p "$LOG_DIR"
for path in "$TRAIN_CSV" "$VAL_CSV" "$TEACHER_PATH"; do
  if [[ ! -s "$path" ]]; then
    echo "[ERROR] Required file is missing or empty: $path" >&2
    exit 2
  fi
done
for path in "$BASE_MODEL_PATH" "$POLICY_ADAPTER_CHECKPOINT"; do
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Required directory is missing: $path" >&2
    exit 2
  fi
done

echo "===== MUTAGENICITY STABLE PPO FULL ====="
echo "hostname=$(hostname)"
echo "date=$(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PWD after cd=$PWD"
echo "PYTHONPATH=$PYTHONPATH"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "POLICY_ADAPTER_CHECKPOINT=$POLICY_ADAPTER_CHECKPOINT"
echo "TOKENIZER_PATH=auto"
echo "TOKENIZER_FALLBACK_PATH=$TOKENIZER_FALLBACK_PATH"
echo "TEACHER_PATH=$TEACHER_PATH"
echo "SOURCE_LABEL=$SOURCE_LABEL"
echo "TARGET_LABEL=$TARGET_LABEL"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MAX_PARENTS=$MAX_PARENTS"
echo "MAX_UPDATES=$MAX_UPDATES"
echo "samples_per_update=$SAMPLES_PER_UPDATE"
echo "updates_per_epoch=$UPDATES_PER_EPOCH"
echo "git commit=$(git rev-parse HEAD)"
echo "python path=$(which python)"
echo "conda env=${CONDA_DEFAULT_ENV:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

python scripts/train_mutagenicity_ppo_stable.py \
  --config configs/hpc.yaml \
  --mode full \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --base-model-path "$BASE_MODEL_PATH" \
  --policy-adapter-checkpoint "$POLICY_ADAPTER_CHECKPOINT" \
  --tokenizer-fallback-path "$TOKENIZER_FALLBACK_PATH" \
  --teacher-path "$TEACHER_PATH" \
  --oracle-path "$TEACHER_PATH" \
  --output-dir "$OUTPUT_ROOT" \
  --source-label "$SOURCE_LABEL" \
  --target-label "$TARGET_LABEL" \
  --max-parents "$MAX_PARENTS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --max-updates "$MAX_UPDATES" \
  --save-steps "$SAVE_STEPS" \
  --eval-every-steps "$EVAL_EVERY_STEPS" \
  --logging-steps 1 \
  --ppo-learning-rate 1e-6 \
  --ppo-clip-range 0.05 \
  --stable-ppo-epochs 1 \
  --max-grad-norm 0.5 \
  --target-kl 0.30 \
  --hard-kl 0.80 \
  --enable-adaptive-kl \
  --kl-penalty-init 0.05 \
  --kl-penalty-multiplier 1.5 \
  --reward-clip-min -5.0 \
  --reward-clip-max 5.0 \
  --normalize-reward \
  --normalize-advantage \
  --enable-teacher-confidence-gate \
  --min-teacher-p-before 0.5 \
  --low-conf-cf-weight 0.3 \
  --save-best-checkpoint \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight 0.3 \
  --projection-penalty 1.0 \
  --enable-minimal-syntax-repair \
  --enable-component-salvage \
  --require-chemistry-reward-path \
  --require-teacher-sem \
  --log-unified-ppo-samples

python scripts/audit_mutagenicity_ppo_run.py \
  --run-dir "$OUTPUT_ROOT" \
  --require-full-coverage

