#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: process-set verification is an AutoDL v5 supervisor preflight" >&2
echo "reference: python scripts/autodl/verify_aids_comrecgc_v5_process_set.py --config configs/hpc.yaml --proc-root /proc --allowed-pid PID --allowed-start-ticks TICKS --allowed-cmdline-sha256 SHA --allowed-output-root /old/output --allowed-project-root /old/worktree" >&2
echo "mid-run adds: --allowed-route-root-pid PID --allowed-route-root-start-ticks TICKS --allowed-route-output-root /fresh/common_recourse --allowed-route-project-root /immutable/worktree" >&2
exit 78
