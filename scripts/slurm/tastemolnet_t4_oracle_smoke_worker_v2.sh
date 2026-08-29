#!/usr/bin/env bash
#SBATCH --job-name=taste-t4-worker-v2
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
echo "TasteMolNet T4 adaptive calibration-only managed-v2 science is AutoDL-only; this Slurm wrapper is static CLI parity." >&2
echo "Gate: 16x8->64x16->128x32, strict_flips>=16, distinct_parents>=8; destination diversity is nonblocking." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python -I -B scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t3-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed \
  --graph-cache-root /absolute/private/graph-cache \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --controller-launcher-receipt /absolute/launcher/launcher_receipt.json \
  --controller-receipt /absolute/controller/controller_receipt.json \
  --controller-anchor-heartbeat /absolute/controller/heartbeats/00000000000000000001-00000000-0000-4000-8000-000000000001.json \
  --expected-controller-id taste-main-v2-UUID \
  --expected-git-commit COMMIT --expected-git-tree TREE \
  --expected-controller-launcher-receipt-sha256 SHA256 \
  --expected-controller-receipt-sha256 SHA256 \
  --expected-controller-anchor-heartbeat-sha256 SHA256 \
  --expected-gpu-lease-uuid UUID --expected-gpu-lease-sha256 SHA256
