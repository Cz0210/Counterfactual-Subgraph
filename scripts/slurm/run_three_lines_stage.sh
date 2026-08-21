#!/bin/bash
#SBATCH --job-name=three_lines_stage
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
STEP0_PROJECT_ROOT=${STEP0_PROJECT_ROOT:?STEP0_PROJECT_ROOT is required}
PERSISTENT_ROOT=${PERSISTENT_ROOT:?PERSISTENT_ROOT is required}
FAST_ROOT=${FAST_ROOT:?FAST_ROOT is required}
EXTERNAL_ROOT=${EXTERNAL_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/122f9341a360e9f06bb58a2f5823bb596021f6bf}
STAGE=${STAGE:?STAGE is required}
RESUME=${RESUME:-0}

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONHASHSEED=0
mkdir -p logs

case "$STAGE" in
  mut-*|aids-*) export DISALLOW_GENERATION=1 ;;
esac

echo "hostname=$(hostname)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

args=(
  "$STAGE"
  --project-root "$PROJECT_ROOT"
  --step0-project-root "$STEP0_PROJECT_ROOT"
  --external-root "$EXTERNAL_ROOT"
  --persistent-root "$PERSISTENT_ROOT"
  --fast-root "$FAST_ROOT"
  --python "$(command -v python)"
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
)
[[ "$RESUME" == 1 ]] && args+=(--resume)

python scripts/autodl/run_three_lines_stage.py "${args[@]}"
