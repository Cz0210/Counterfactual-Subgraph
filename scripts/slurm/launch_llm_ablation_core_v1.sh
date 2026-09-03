#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=launch-llm-core-ablation
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

# The paired launcher performs its own evidence status decision, then execs
# run_llm_ablation_variant.py --config configs/hpc.yaml.  It is not status-only.
exec bash scripts/autodl/launch_llm_ablation_core_v1.sh
