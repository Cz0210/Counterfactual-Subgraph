#!/usr/bin/env bash
#SBATCH --job-name=t14-resume-owner
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=450G
#SBATCH --time=48:00:00
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

: "${T14_RESUME_SPEC:?required}"
: "${T14_CANARY_RECEIPT:?required}"
: "${T14_OWNER_ROOT:?required}"
: "${T14_AUDITOR_PID:?required}"
: "${T14_AUDITOR_START_TICKS:?required}"

python scripts/autodl/run_t14_low_memory_resume_owner.py \
  --config configs/hpc.yaml \
  --resume-spec "$T14_RESUME_SPEC" \
  --canary-receipt "$T14_CANARY_RECEIPT" \
  --owner-root "$T14_OWNER_ROOT" \
  --science-wrapper "$PWD/scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh" \
  --cgroup-limit-file /sys/fs/cgroup/memory/memory.limit_in_bytes \
  --cgroup-current-file /sys/fs/cgroup/memory/memory.usage_in_bytes \
  --auditor-pid "$T14_AUDITOR_PID" \
  --auditor-start-ticks "$T14_AUDITOR_START_TICKS"
