#!/usr/bin/env bash
#SBATCH --job-name=main-ablation-sidecar
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
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
echo "AutoDL process-adoption scheduler only; Slurm launch is disabled" >&2
exit 64
# python scripts/autodl/run_main_and_ablations_v1.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --state-root /runtime/control/main-and-ablations-v1 --matrix-authority /runtime/control/fast16_matrix_authority/state.json
