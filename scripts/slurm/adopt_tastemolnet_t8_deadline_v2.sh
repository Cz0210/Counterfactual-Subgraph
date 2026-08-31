#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=t8-adopt-v2

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONNOUSERSITE=1
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] \
  || { echo "T8 managed adoption forbids GNN ablation" >&2; exit 64; }

: "${T8_DEADLINE_OUTPUT_ROOT:?fresh 25-epoch deadline PASS root is required}"
: "${T8_DEADLINE_STATE_ROOT:?deadline private state root is required}"
: "${T8_DEADLINE_ATTEMPT_ID:?fresh deadline UUIDv4 is required}"
: "${T8_RECOVERY_SOURCE_ATTEMPT_ID:?failed source attempt UUIDv4 is required}"
: "${TASTEMOLNET_T3_OUTPUT:?T3 typed PASS root is required}"
: "${TASTEMOLNET_T4_OUTPUT:?T4 typed PASS root is required}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?frozen GINE checkpoint is required}"
: "${TASTEMOLNET_TRAIN_CSV:?frozen train split is required}"
: "${TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT:?pinned official source is required}"
: "${T8_ADOPTION_STAGE_ROOT:?existing managed stage parent is required}"
: "${T8_ADOPTION_FINAL_PATH:?fresh managed-v2 final path is required}"
: "${T8_ADOPTION_MANAGED_ATTEMPT_ID:?fresh managed UUIDv4 is required}"
: "${T8_ADOPTION_RUN_ID:?controller run ID is required}"

python -I -B scripts/autodl/adopt_tastemolnet_t8_deadline_v2.py \
  --mode run \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --deadline-output-root "$T8_DEADLINE_OUTPUT_ROOT" \
  --deadline-state-root "$T8_DEADLINE_STATE_ROOT" \
  --deadline-attempt-id "$T8_DEADLINE_ATTEMPT_ID" \
  --recovery-source-attempt-id "$T8_RECOVERY_SOURCE_ATTEMPT_ID" \
  --t3-output "$TASTEMOLNET_T3_OUTPUT" \
  --t4-output "$TASTEMOLNET_T4_OUTPUT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
  --stage-root "$T8_ADOPTION_STAGE_ROOT" \
  --final-path "$T8_ADOPTION_FINAL_PATH" \
  --managed-attempt-id "$T8_ADOPTION_MANAGED_ATTEMPT_ID" \
  --run-id "$T8_ADOPTION_RUN_ID"
