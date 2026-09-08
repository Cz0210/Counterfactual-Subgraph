#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
# Read-only lock inventory. This wrapper never starts an AutoDL process.
# AutoDL LLM dispatch uses gpu_lock.py run --llm-dispatch-spec <sealed.json>
# --llm-dispatch-spec-sha256 <SHA> --owner-output-root <fresh> on AutoDL only;
# a Slurm allocation cannot supply main-priority reservation evidence.
# --llm-complete-chain is the bounded existing AutoDL L1/L2/L3 continuation;
# it releases each GPU lease before its CPU evaluator. This wrapper stays read-only.
# The explicit parser-correction binding is CPU-evaluation-only, pinned to one
# queue root, and cannot enter generation or obtain a GPU lease.
# --t13-performance-spec defaults to read-only preflight. On AutoDL only,
# --t13-performance-execute uses the existing UUID/slot owner, inherited FD
# and terminal-aware provider. Missing source/resource bindings stay BLOCKED.
# This HPC wrapper never launches the diagnostic or changes a main registry.
python scripts/autodl/gpu_lock.py --config configs/hpc.yaml list
