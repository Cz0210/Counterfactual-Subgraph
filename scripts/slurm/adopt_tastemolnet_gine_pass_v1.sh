#!/usr/bin/env bash
#SBATCH --job-name=taste-t2-pass-adoption-v1-static-refusal
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "[TASTE_T2_PASS_ADOPTION_AUTODL_ONLY] refusing HPC execution" >&2
echo "Use the reviewed AutoDL CLI in preflight/status mode; publish remains release-frozen." >&2
exit 78

# Static CLI parity only. This line is unreachable by design and must not be
# changed into a submission route.
python -I -B scripts/autodl/adopt_tastemolnet_gine_pass_v1.py \
  --config configs/hpc.yaml \
  preflight \
  --control-root /autodl-fs/data/counterfactual-subgraph-runtime/control \
  --controller-root /autodl-fs/data/counterfactual-subgraph-runtime/control/tastemolnet-gine-training-v2/tastemolnet_gine_v2_20260827T160626Z_583bf668 \
  --scientific-output-root /autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/tastemolnet/gine/seed7/full-20260827T160626Z \
  --training-state-root /path/to/exact-reviewed-training-state-root \
  --execution-project-root /path/to/deployed/583bf668-worktree \
  --identity-fix-project-root /path/to/deployed/3a90fd86-worktree
