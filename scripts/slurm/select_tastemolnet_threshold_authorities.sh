#!/usr/bin/env bash
#SBATCH --job-name=taste_thresholds
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

: "${TASTE_T3_ROOT:?set the frozen managed T3 root}"
: "${TASTE_T4_ROOT:?set the frozen managed T4 root}"
: "${TASTE_GRAPH_CACHE_ROOT:?set the frozen Taste graph-cache root}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the managed NeuroSED PASS root}"
: "${MOLCLR_ROOT:?set the pinned MolCLR checkout}"
: "${MOLCLR_CHECKPOINT:?set the pinned MolCLR GIN checkpoint}"
: "${TASTE_THRESHOLD_OUTPUT_ROOT:?set one fresh output root}"

OFFICIAL_GCF_ROOT=${OFFICIAL_GCF_ROOT:-$PWD/baselines/gcfexplainer_official}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-/autodl-fs/data/counterfactual-subgraph-runtime/cache/tastemolnet/threshold_authorities/wnode.sqlite}
NODE_EMBEDDING_CACHE_DIR=${NODE_EMBEDDING_CACHE_DIR:-/autodl-fs/data/counterfactual-subgraph-runtime/cache/tastemolnet/threshold_authorities/molclr_nodes}
mkdir -p "$(dirname "$WNODE_CACHE_DB")" "$NODE_EMBEDDING_CACHE_DIR" logs

python -B scripts/select_tastemolnet_threshold_authorities.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t3-root "$TASTE_T3_ROOT" \
  --t4-root "$TASTE_T4_ROOT" \
  --graph-cache-root "$TASTE_GRAPH_CACHE_ROOT" \
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT" \
  --official-gcf-root "$OFFICIAL_GCF_ROOT" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --output-root "$TASTE_THRESHOLD_OUTPUT_ROOT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
  --device cuda:0 \
  --batch-size 64
