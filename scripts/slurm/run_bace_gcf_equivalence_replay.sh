#!/bin/bash
#SBATCH --job-name=bace_gcf_replay
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export AUTODL_PYTHON="$(command -v python)"
export AUTODL_PHYSICAL_GPU_UUID="${AUTODL_PHYSICAL_GPU_UUID:-SLURM_VISIBLE_GPU}"
echo "python=$AUTODL_PYTHON"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

# The called runner explicitly supplies configs/hpc.yaml and disables the
# heuristic fallback for both replay implementations.
bash scripts/autodl/run_bace_gcf_equivalence_replay.sh
