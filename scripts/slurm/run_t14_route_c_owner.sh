#!/usr/bin/env bash
#SBATCH --job-name=t14-route-c-owner
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
echo "route_c_formal_canary_promotes_step500_without_replaying_steps_1_500"

: "${T14_ROUTE_C_TASK_SPEC:?required}"
: "${T14_ROUTE_C_CONTINUATION_SPEC:?required}"
python scripts/autodl/run_t14_route_c_owner.py \
  --config configs/hpc.yaml \
  --task-spec "$T14_ROUTE_C_TASK_SPEC" \
  --continuation-spec "$T14_ROUTE_C_CONTINUATION_SPEC"
