#!/usr/bin/env bash
#SBATCH --job-name=t14-route-c-cont
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
echo "T14 Route C continuation consumes AutoDL authorities; direct Slurm execution is disabled" >&2
exit 64

# AutoDL documentation only:
# python scripts/autodl/run_t14_route_c_continuation.py \
#   --config configs/hpc.yaml \
#   --set inference.fallback_to_heuristic=false \
#   --continuation-spec /absolute/T14_ROUTE_C_CONTINUATION_SPEC.json
