#!/usr/bin/env bash
#SBATCH --job-name=t14-ledger-review
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Task-specific exception: this saved-JSON audit must not reserve a GPU or run
# inference. No fallback_to_heuristic flag exists because there is no inference.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
echo "python=$(command -v python)"
python --version
echo "saved-ledger-only; cuda/not-requested; no transitions"
python -I -B scripts/autodl/review_t14_route_c_parity.py --config configs/hpc.yaml "$@"
