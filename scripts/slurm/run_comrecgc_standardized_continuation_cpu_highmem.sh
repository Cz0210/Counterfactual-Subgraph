#!/bin/bash
# Static CLI-parity wrapper required by repository policy.  The guarded repair
# is AutoDL-only and depends on AutoDL cgroup-v1 evidence; do not submit it.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids-comrecgc-cpu-highmem

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

echo "AutoDL-only guard: no scientific command is executed from Slurm."
exit 2
