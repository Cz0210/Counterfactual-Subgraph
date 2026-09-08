#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Compatibility wrapper only: preparation/pilot/handoff belong on AutoDL, not HPC.
# This guard exits before environment activation or any task action.
echo "AutoDL-only one-shot resource binding: do not submit this wrapper to HPC."
exit 2
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "Python: $(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
python scripts/autodl/prepare_global_cpu_resource_successor.py --config configs/hpc.yaml "$@"
