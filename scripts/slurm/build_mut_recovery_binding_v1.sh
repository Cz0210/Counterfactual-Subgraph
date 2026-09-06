#!/bin/bash
#SBATCH --job-name=mut-recovery-binding-gate
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
echo "AutoDL-only binding. Do not submit this refusal wrapper."
python scripts/autodl/build_mut_recovery_binding_v1.py --config configs/hpc.yaml --help
exit 2
