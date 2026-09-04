#!/bin/bash
#SBATCH --job-name=mut-ab-spec
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
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

: "${MUT_SAME_CONTRACT_AB_TEMPLATE:?MUT_SAME_CONTRACT_AB_TEMPLATE is required}"
: "${MUT_SAME_CONTRACT_AB_SPEC:?MUT_SAME_CONTRACT_AB_SPEC is required}"
python scripts/autodl/build_mut_same_contract_ab_task_spec_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  build --template "$MUT_SAME_CONTRACT_AB_TEMPLATE" \
  --output "$MUT_SAME_CONTRACT_AB_SPEC"
