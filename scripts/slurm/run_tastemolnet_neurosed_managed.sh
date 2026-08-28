#!/usr/bin/env bash
# Static CLI parity only. Taste NeuroSED managed launch is AutoDL-only.
#SBATCH --job-name=taste-neurosed-managed-refuse
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
echo "REFUSING_HPC_EXECUTION: TasteMolNet NeuroSED is AutoDL-only and pair semantics remain review-blocked." >&2
exit 78

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/run_tastemolnet_neurosed_managed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --python /absolute/autodl/python \
  --neurosed-config configs/autodl/tastemolnet_neurosed_v1.yaml \
  --train-csv /absolute/private/splits/train.csv \
  --validation-csv /absolute/private/splits/validation.csv \
  --t2-receipt-root /absolute/t2/receipt \
  --t2-source-bundle-root /absolute/t2/source \
  --t3-final-root /absolute/t3/final \
  --controller-receipt /absolute/controller/receipt.json \
  --controller-heartbeat /absolute/controller/heartbeat.json \
  --expected-controller-id controller-id \
  --stage-root /absolute/managed/stage \
  --final-root /absolute/fresh/final-root \
  --execution-git-commit 0000000000000000000000000000000000000000 \
  --execution-git-tree 0000000000000000000000000000000000000000 \
  --device cuda:0 \
  --require-cuda-tolerance
