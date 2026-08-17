#!/bin/bash
#SBATCH --job-name=audit_project_gpu_lanes
#SBATCH --partition=intel
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
OUTPUT_JSON=${OUTPUT_JSON:?OUTPUT_JSON is required}
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
python scripts/ops/audit_project_gpu_lanes.py \
  --emit-json "$OUTPUT_JSON"
