#!/bin/bash
#SBATCH --job-name=resolve_frozen_thresholds
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
cd "${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
export PYTHONPATH=$PWD
mkdir -p logs
THRESHOLDS_JSON=${THRESHOLDS_JSON:?THRESHOLDS_JSON is required}
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
python scripts/resolve_frozen_wnode_thresholds.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --thresholds-json "$THRESHOLDS_JSON" --validate-only
