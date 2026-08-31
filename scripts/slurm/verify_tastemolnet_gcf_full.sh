#!/usr/bin/env bash
#SBATCH --job-name=taste-t12-verify
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7

: "${T12_GENERATION_ROOT:?set T12_GENERATION_ROOT}"
: "${T12_TRAIN_CSV:?set T12_TRAIN_CSV}"
: "${T12_CALIBRATION_CSV:?set T12_CALIBRATION_CSV}"
: "${T12_TEST_CSV:?set T12_TEST_CSV}"
: "${T12_GNN_CHECKPOINT:?set T12_GNN_CHECKPOINT}"
: "${T12_MOLCLR_ROOT:?set T12_MOLCLR_ROOT}"
: "${T12_MOLCLR_CHECKPOINT:?set T12_MOLCLR_CHECKPOINT}"
: "${T12_WNODE_THRESHOLD_CONTRACT:?set T12_WNODE_THRESHOLD_CONTRACT}"
: "${T12_PAPER_ROOT:?set T12_PAPER_ROOT}"
: "${T12_TERMINAL_VERIFICATION_ROOT:?set T12_TERMINAL_VERIFICATION_ROOT}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'

python scripts/verify_tastemolnet_gcf_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --generation-root "$T12_GENERATION_ROOT" \
  --generation-verification-root "$T12_GENERATION_ROOT/generation_verification" \
  --train-csv "$T12_TRAIN_CSV" \
  --calibration-csv "$T12_CALIBRATION_CSV" \
  --test-csv "$T12_TEST_CSV" \
  --gnn-checkpoint "$T12_GNN_CHECKPOINT" \
  --molclr-root "$T12_MOLCLR_ROOT" \
  --molclr-checkpoint "$T12_MOLCLR_CHECKPOINT" \
  --threshold-contract "$T12_WNODE_THRESHOLD_CONTRACT" \
  --output-root "$T12_PAPER_ROOT" \
  --verification-root "$T12_TERMINAL_VERIFICATION_ROOT"
