#!/usr/bin/env bash
# Static refusal: T13 chemistry/GINE/calibration/test is explicitly AutoDL-only.
#SBATCH --job-name=t13-from-hpc-autodl-only
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/run_t13_from_hpc_import_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --help >/dev/null
echo "REFUSING_HPC_EXECUTION: HPC is not authorized for GINE/calibration/test." >&2
exit 78
