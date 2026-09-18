#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
# V10 explicitly overrides the repository's generic GPU template: HPC CPU only.
# This paired entry is ONLY the synthetic capture serialization preflight.
# Actual replay remains AutoDL under the original GPU2 lease and V8 ledger.
python -I -B scripts/autodl/diagnose_t14_causal_170.py --config configs/hpc.yaml --action follower-preflight "$@"
