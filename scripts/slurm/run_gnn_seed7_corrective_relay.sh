#!/usr/bin/env bash
# Mac external-disk transport is not executable on a Slurm node.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
echo 'REFUSING_HPC_EXECUTION: corrective relay requires the Mac external volume.' >&2
exit 78
