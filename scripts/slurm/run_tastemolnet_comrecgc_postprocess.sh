#!/usr/bin/env bash
#SBATCH --job-name=taste-t14-post
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "T14 retained-authority postprocess is AutoDL-only; direct Slurm execution is disabled" >&2
exit 64

# CLI parity (documentation only):
# python scripts/run_tastemolnet_comrecgc_postprocess.py --mode postprocess \
#   --config configs/hpc.yaml --generation-root "$TASTEMOLNET_T14_GENERATION_ROOT" \
#   --science-root "$TASTEMOLNET_T14_POSTPROCESS_ROOT" \
#   --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV" \
#   --test-csv "$TASTEMOLNET_TEST_CSV" \
#   --gnn-checkpoint "$TASTEMOLNET_T3_OUTPUT_ROOT/artifacts/checkpoint" \
#   --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
#   --threshold-contract "$TASTEMOLNET_WNODE_THRESHOLD_JSON" \
#   --wnode-cache-db "$WNODE_CACHE_DB" \
#   --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
#   --set inference.fallback_to_heuristic=false
