#!/usr/bin/env bash
#SBATCH --job-name=taste_t13_full
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'
nvidia-smi

: "${T8_PASS_ROOT:?set T8_PASS_ROOT to the published managed-v2 T8 final directory}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set TASTEMOLNET_GNN_CHECKPOINT}"
: "${TASTEMOLNET_TRAIN_CSV:?set TASTEMOLNET_TRAIN_CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set TASTEMOLNET_CALIBRATION_CSV}"
: "${TASTEMOLNET_TEST_CSV:?set TASTEMOLNET_TEST_CSV}"
: "${GLOBALGCE_OFFICIAL_ROOT:?set GLOBALGCE_OFFICIAL_ROOT}"
: "${MOLCLR_ROOT:?set MOLCLR_ROOT}"
: "${MOLCLR_CHECKPOINT:?set MOLCLR_CHECKPOINT}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set TASTEMOLNET_THRESHOLD_CONTRACT}"
: "${T13_OUTPUT_DIR:?set T13_OUTPUT_DIR to one fresh persistent root}"

WNODE_CACHE_DB=${WNODE_CACHE_DB:-outputs/hpc/cache/distance_cache/tastemolnet_globalgce_full.sqlite}
NODE_EMBEDDING_CACHE_DIR=${NODE_EMBEDDING_CACHE_DIR:-outputs/hpc/cache/molclr_node_embeddings}
T13_EPOCHS=${T13_EPOCHS:-100}

resume_args=()
if [[ -f "$T13_OUTPUT_DIR/checkpoint.json" ]]; then
  resume_args+=(--resume)
fi

python -I -B scripts/run_tastemolnet_globalgce_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t8-pass-root "$T8_PASS_ROOT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV" \
  --test-csv "$TASTEMOLNET_TEST_CSV" \
  --official-root "$GLOBALGCE_OFFICIAL_ROOT" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT" \
  --output-dir "$T13_OUTPUT_DIR" \
  --epochs "$T13_EPOCHS" \
  "${resume_args[@]}"

# A distinct process reopens the sealed bytes and is the only invocation that
# can publish the terminal PASS marker.
python -I -B scripts/run_tastemolnet_globalgce_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --output-dir "$T13_OUTPUT_DIR" \
  --verify-only
