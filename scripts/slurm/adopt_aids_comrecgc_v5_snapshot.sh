#!/bin/bash
#SBATCH --job-name=aids-snapshot-adopt
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only.  Production is CPU-only and owned by the fresh
# persistent AutoDL controller; this job must never duplicate that authority.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "AutoDL-only snapshot adoption; do not submit this Slurm wrapper." >&2
echo "reference: python scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py --config configs/hpc.yaml --help" >&2
exit 78
