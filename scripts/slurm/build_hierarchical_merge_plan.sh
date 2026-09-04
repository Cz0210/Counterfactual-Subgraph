#!/usr/bin/env bash
# CPU-only planning job; the T8 scientific shards are already complete.
#SBATCH --job-name=t8-hier-plan
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'
exec python scripts/hpc/t8/build_hierarchical_merge_plan.py --config configs/hpc.yaml "$@"
