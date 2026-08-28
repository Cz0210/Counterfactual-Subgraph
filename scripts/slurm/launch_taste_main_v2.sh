#!/usr/bin/env bash
#SBATCH --job-name=taste-main-v2-launch
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "Taste main-v2 foreground launcher is AutoDL-only; Slurm is refused." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python -I -B scripts/autodl/run_taste_main_v2.py --config configs/hpc.yaml launch \
  --control-root /absolute/runtime/control \
  --controller-root /absolute/runtime/control/taste-main-v2/controllers/UUID \
  --launcher-root /absolute/runtime/control/taste-main-v2/launches/UUID \
  --controller-id taste-main-v2-UUID --controller-uuid UUID \
  --project-root "$PWD" --persistent-storage-root /absolute/runtime \
  --expected-git-commit COMMIT --expected-git-tree TREE \
  --controller-log /absolute/runtime/logs/taste-main-v2/UUID/controller.log
