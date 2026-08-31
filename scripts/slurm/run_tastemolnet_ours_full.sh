#!/usr/bin/env bash
#SBATCH --job-name=taste_t11_eval
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
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

: "${T11_PPO_OUTPUT_ROOT:?set the passing T11 PPO root}"
: "${T11_SCIENCE_ROOT:?set the fresh/resumable T11 science root}"
: "${T11_FINAL_ROOT:?set the fresh verifier publication root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set TASTEMOLNET_GNN_CHECKPOINT}"
: "${TASTEMOLNET_TRAIN_CSV:?set TASTEMOLNET_TRAIN_CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set TASTEMOLNET_CALIBRATION_CSV}"
: "${TASTEMOLNET_TEST_CSV:?set TASTEMOLNET_TEST_CSV}"
: "${MOLCLR_ROOT:?set MOLCLR_ROOT}"
: "${MOLCLR_CHECKPOINT:?set MOLCLR_CHECKPOINT}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set the existing shared frozen threshold contract}"

WNODE_CACHE_DB=${WNODE_CACHE_DB:-outputs/hpc/cache/distance_cache/tastemolnet_ours_full.sqlite}
NODE_EMBEDDING_CACHE_DIR=${NODE_EMBEDDING_CACHE_DIR:-outputs/hpc/cache/molclr_node_embeddings}
resume_args=()
if [[ -f "$T11_SCIENCE_ROOT/checkpoint.json" ]]; then
  resume_args+=(--resume)
fi

python scripts/run_tastemolnet_ours_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --science-root "$T11_SCIENCE_ROOT" \
  --ppo-root "$T11_PPO_OUTPUT_ROOT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV" \
  --test-csv "$TASTEMOLNET_TEST_CSV" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
  "${resume_args[@]}"

python scripts/run_tastemolnet_ours_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --science-root "$T11_SCIENCE_ROOT" \
  --final-root "$T11_FINAL_ROOT" \
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT" \
  --verify-only
