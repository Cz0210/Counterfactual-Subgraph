#!/usr/bin/env bash
# This entrypoint is an AutoDL import audit, never an HPC science job.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
python -I -B scripts/ablations/llm/import_corrected_gnn_core.py --config configs/hpc.yaml --help
echo 'REFUSING_HPC_EXECUTION: scoped importer must run on AutoDL.' >&2
exit 78
