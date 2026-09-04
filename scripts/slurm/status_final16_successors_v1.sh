#!/usr/bin/env bash
# Static AutoDL-only guard for the final16 successors status view.
#SBATCH --job-name=final16-successors-status-static-refusal
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
python scripts/autodl/status_final16_successors_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --help >/dev/null
echo "REFUSING_HPC_EXECUTION: final16 status must read the AutoDL control root." >&2
exit 78
