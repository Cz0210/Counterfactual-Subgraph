#!/bin/bash
# Main one-epoch Fresh Mutagenicity PPO route (1448 parents, 91 updates).

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=mut_fresh_ppo_full
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
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/mutagenicity_ppo_prompts_train_label1_v2.csv}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/mutagenicity_ppo_prompts_val_label1_v2.csv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PROJECT_ROOT/pretrained_models/ChemLLM-7B-Chat}"
POLICY_ADAPTER_CHECKPOINT="${POLICY_ADAPTER_CHECKPOINT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/final/sft_fresh_strict_v2_best}"
TEACHER_PATH="${TEACHER_PATH:-$PROJECT_ROOT/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
REWARD_CONFIG_JSON="${REWARD_CONFIG_JSON:-$PROJECT_ROOT/outputs/hpc/mutagenicity/audits/ppo_reward_components_v1/recommended_reward_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/hpc/mutagenicity/ppo_fresh_v2}"
MAX_PARENTS="${MAX_PARENTS:-1448}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
UPDATES_PER_EPOCH=$(( (MAX_PARENTS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
MAX_UPDATES="${MAX_UPDATES:-$UPDATES_PER_EPOCH}"
mkdir -p "$PROJECT_ROOT/logs"

echo "===== MUTAGENICITY FRESH PPO FULL ====="
echo "PROJECT_ROOT=$PROJECT_ROOT TRAIN_CSV=$TRAIN_CSV VAL_CSV=$VAL_CSV"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH POLICY_ADAPTER_CHECKPOINT=$POLICY_ADAPTER_CHECKPOINT"
echo "TEACHER_PATH=$TEACHER_PATH REWARD_CONFIG_JSON=$REWARD_CONFIG_JSON OUTPUT_ROOT=$OUTPUT_ROOT"
echo "MAX_PARENTS=$MAX_PARENTS samples_per_update=$ROLLOUT_BATCH_SIZE updates_per_epoch=$UPDATES_PER_EPOCH MAX_UPDATES=$MAX_UPDATES"
echo "git_commit=$(git rev-parse HEAD || true)"
nvidia-smi || true

python scripts/train_mutagenicity_ppo_fresh.py \
  --reward-config-json "$REWARD_CONFIG_JSON" \
  --config configs/hpc.yaml \
  --mode full \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --base-model-path "$BASE_MODEL_PATH" \
  --policy-adapter-checkpoint "$POLICY_ADAPTER_CHECKPOINT" \
  --tokenizer-path "$POLICY_ADAPTER_CHECKPOINT" \
  --teacher-path "$TEACHER_PATH" \
  --oracle-path "$TEACHER_PATH" \
  --output-dir "$OUTPUT_ROOT" \
  --max-parents "$MAX_PARENTS" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --max-updates "$MAX_UPDATES" \
  --eval-every-steps 10 \
  --save-steps 10 \
  --logging-steps 1 \
  --ppo-learning-rate 1e-6 \
  --ppo-clip-range 0.05 \
  --stable-ppo-epochs 1 \
  --max-grad-norm 0.5 \
  --target-kl 0.30 \
  --hard-kl 0.80 \
  --enable-adaptive-kl \
  --normalize-reward \
  --normalize-advantage \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight 0.3 \
  --require-chemistry-reward-path \
  --require-teacher-sem \
  --save-best-checkpoint

python scripts/audit_mutagenicity_ppo_fresh.py \
  --config configs/hpc.yaml \
  --run-dir "$OUTPUT_ROOT" \
  --expected-mode full
