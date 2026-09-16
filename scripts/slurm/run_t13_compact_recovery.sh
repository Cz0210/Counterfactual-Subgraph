#!/bin/bash
# --action continue is a same-run AutoDL owner stage, never a fresh Slurm start.
# AutoDL-only GPU1 owner/FD probe. Never sbatch on a mismatched HPC GPU.
# V5 user override: HPC CPU only. This AutoDL-only launcher is not submitted.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "T13 compact recovery python=$(command -v python)"
python -V
echo 'AutoDL-specific absolute inputs and inherited GPU1 lease required; use deployed owner command.' >&2
exit 2
