#!/usr/bin/env bash
#SBATCH --job-name=taste_t11_ppo
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'
nvidia-smi

: "${T6_OUTPUT_ROOT:?set T6_OUTPUT_ROOT}"
: "${T11_PPO_OUTPUT_ROOT:?set T11_PPO_OUTPUT_ROOT to a fresh root}"
: "${TASTEMOLNET_BASE_MODEL:?set TASTEMOLNET_BASE_MODEL}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set TASTEMOLNET_GNN_CHECKPOINT}"
: "${TASTEMOLNET_TRAIN_CSV:?set TASTEMOLNET_TRAIN_CSV}"

resume_args=()
if [[ -n "${T11_PPO_RESUME_CHECKPOINT:-}" ]]; then
  resume_args+=(--resume-from-checkpoint "$T11_PPO_RESUME_CHECKPOINT")
fi

python scripts/train_tastemolnet_ours_full.py \
  --config configs/hpc.yaml \
  --model-path "$TASTEMOLNET_BASE_MODEL" \
  --dataset-path "$TASTEMOLNET_TRAIN_CSV" \
  --output-dir "$T11_PPO_OUTPUT_ROOT" \
  --t6-output "$T6_OUTPUT_ROOT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --batch-size "${T11_PPO_BATCH_SIZE:-8}" \
  "${resume_args[@]}"
