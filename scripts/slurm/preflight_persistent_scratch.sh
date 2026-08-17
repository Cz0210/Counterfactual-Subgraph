#!/bin/bash
#SBATCH --job-name=preflight_scratch
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
SCRATCH_ROOT=${SCRATCH_ROOT:?SCRATCH_ROOT is required}
OUTPUT_JSON=${OUTPUT_JSON:?OUTPUT_JSON is required}
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
python scripts/ops/preflight_persistent_scratch.py \
  --root "$SCRATCH_ROOT" --output "$OUTPUT_JSON" \
  --min-free-gib 50 --min-free-inodes 100000
