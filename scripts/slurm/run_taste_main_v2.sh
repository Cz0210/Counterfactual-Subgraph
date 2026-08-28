#!/usr/bin/env bash
#SBATCH --job-name=taste-main-v2-run
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
echo "Taste main-v2 controller authority is AutoDL-only; Slurm is refused." >&2
exit 64

# Unreachable controller-child protocol parity. The external launcher owns
# descriptors 3/4; this command must never be invoked manually.
python -I -B scripts/autodl/run_taste_main_v2.py --config configs/hpc.yaml run \
  --controller-root /absolute/runtime/control/taste-main-v2/controllers/UUID \
  --controller-id taste-main-v2-UUID --controller-uuid UUID \
  --project-root "$PWD" --persistent-storage-root /absolute/runtime \
  --expected-git-commit COMMIT --expected-git-tree TREE \
  --launcher-receipt /absolute/runtime/control/taste-main-v2/launches/UUID/launcher_receipt.json \
  --launcher-handshake-fd 3 --launcher-registration-fd 4
