#!/bin/bash
#SBATCH --job-name=mut-trace-launch
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS=900
export MUT_TRACE_SEMANTIC_FINALIZER_PROJECT_ROOT=/root/autodl-tmp/worktrees/final-five-closeout-582bc4b-20260902T040000Z
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "AutoDL-only persistent launcher; do not submit this Slurm guard."
bash scripts/autodl/launch_mut_trace_on_adoption_worker.sh --help || true
exit 2
