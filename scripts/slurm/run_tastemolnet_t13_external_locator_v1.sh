#!/usr/bin/env bash
#SBATCH --job-name=t13_locator
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

: "${T8_DUAL_CONTROLLER_ROOT:?set the exact running T8 dual controller root}"
: "${T13_LOCATOR_CONTROLLER_ROOT:?set the fresh locator controller root}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
OUTPUT_BASE=${T13_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/globalgce/t13-full}

mkdir -p logs "$T13_LOCATOR_CONTROLLER_ROOT"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/run_tastemolnet_t13_external_locator_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t8-dual-controller-root "$T8_DUAL_CONTROLLER_ROOT" \
  --control-root "$CONTROL" \
  --t13-output-base "$OUTPUT_BASE" \
  --locator-path "$T13_LOCATOR_CONTROLLER_ROOT/cell_root_locator.json" \
  --heartbeat-path "$T13_LOCATOR_CONTROLLER_ROOT/heartbeat.json" \
  --poll-seconds "${T13_LOCATOR_POLL_SECONDS:-60}"
