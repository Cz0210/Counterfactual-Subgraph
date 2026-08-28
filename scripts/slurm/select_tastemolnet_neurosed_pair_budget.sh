#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED fixed-budget work is AutoDL-only.
#SBATCH --job-name=taste-pair-budget-refuse
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
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "REFUSING_HPC_EXECUTION: Taste NeuroSED pair planning is AutoDL-only." >&2
echo "WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED" >&2
echo "WORKER_RESOURCE_EVIDENCE_PRODUCER_NOT_IMPLEMENTED" >&2
exit 78
