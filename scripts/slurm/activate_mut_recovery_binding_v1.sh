#!/bin/bash
#SBATCH --job-name=mut-recovery-activation-gate
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
echo "AutoDL-only one-shot; this Slurm wrapper is a refusal, not a submission."
python scripts/autodl/activate_mut_recovery_binding_v1.py --config configs/hpc.yaml --help
exit 2
