#!/bin/bash
# Submission is a lightweight login-side operation, not another heavy allocation.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
: "${CM_EXECUTION_ROOT:?immutable CM worktree required}"
cd "$CM_EXECUTION_ROOT"
export PYTHONPATH=$PWD
echo "CM submission wrapper Python=$(command -v python)"
python -V
python -I -B scripts/submit_cm_crem_stage.py --config configs/hpc.yaml "$@"
