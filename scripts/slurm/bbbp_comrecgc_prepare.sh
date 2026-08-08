#!/bin/bash
#SBATCH --job-name=bbbp_cr_prepare
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Framework only: this file is syntax/dry-run tested but is not submitted here.
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
export BBBP_PLAN=common4
export BBBP_STAGE=comrec_prepare
exec bash "$PROJECT_ROOT/scripts/slurm/bbbp_stage_common.sh"
