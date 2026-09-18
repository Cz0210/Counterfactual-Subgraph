#!/bin/bash
# --action continue is a same-run AutoDL owner stage, never a fresh Slurm start.
# V10 latest_checkpoint_recovery adopts a bound complete native checkpoint;
# without that overlay, the original epoch29 branch remains unchanged.
# AutoDL-only GPU1 owner/FD probe. Never sbatch on a mismatched HPC GPU.
# V6 user override: HPC CPU only. This AutoDL-only launcher is not submitted.
# Optional final_evaluation_binding executes the existing evaluator/publisher
# only after both native branches and its independent resource gate succeed.
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
