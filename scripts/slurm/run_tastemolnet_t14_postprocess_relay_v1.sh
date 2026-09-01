#!/usr/bin/env bash
#SBATCH --job-name=taste-t14-relay
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
echo "T14 retained-AutoDL relay is AutoDL-only; direct Slurm execution is disabled" >&2
exit 64

# AutoDL launcher (documentation only):
# RUN_GNN_ABLATION=0 T14_RELAY_REPO_ROOT="$PWD" \
#   bash scripts/autodl/launch_tastemolnet_t14_postprocess_relay_v1.sh
