#!/usr/bin/env bash
# CPU-only durable route selection after the bounded Mut same-contract A/B.

#SBATCH --job-name=mut-post-ab-v1
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${MUT_AB_TASK_SPEC:?absolute sealed A/B task spec required}"
: "${MUT_POST_AB_OUTPUT:?absolute fresh continuation output required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'

python scripts/autodl/run_mut_post_ab_continuation_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ab-task-spec "$MUT_AB_TASK_SPEC" \
  --output-root "$MUT_POST_AB_OUTPUT" \
  --poll-seconds 60
