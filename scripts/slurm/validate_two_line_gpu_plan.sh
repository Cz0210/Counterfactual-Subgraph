#!/bin/bash
#SBATCH --job-name=validate_two_gpu_plan
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
PLAN_JSON=${PLAN_JSON:?PLAN_JSON is required}
CURRENT_USAGE_JSON=${CURRENT_USAGE_JSON:?CURRENT_USAGE_JSON is required}
echo "hostname=$(hostname) python=$(which python) commit=$(git rev-parse HEAD)"
python --version
python scripts/ops/validate_two_line_gpu_plan.py \
  --plan-json "$PLAN_JSON" --include-current-squeue \
  --current-usage-json "$CURRENT_USAGE_JSON" \
  --mut-lane-limit 1 --bace-lane-limit 1 --total-limit 2
