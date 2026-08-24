#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids-greed-scan-supervisor

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "This wrapper is syntax/CLI parity evidence only; AutoDL owns the production route."
echo "Do not submit this CPU-only continuation on HPC."

# The immutable AutoDL manifest supplies all provenance-bound arguments and the
# reviewed child argv after `--`.  An ad-hoc Slurm launch is intentionally
# blocked so it cannot create a competing writer.
python scripts/autodl/run_aids_greed_full_scan_supervisor.py \
  --config configs/hpc.yaml \
  --help
exit 78
