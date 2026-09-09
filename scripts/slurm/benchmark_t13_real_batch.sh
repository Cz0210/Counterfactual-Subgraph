#!/bin/bash
# The interceptor requires plan.committed_compact_payload and never falls back
# to full expansion. Historical masks may be reconstructed only after full
# index/input/mask/split/RNG digest proof; this wrapper does not reconstruct.
# The physical Taste canary is AutoDL-owner controlled; this paired HPC wrapper
# is an input-only inspection entry and is not an authorization to run it on HPC.
# Actual AutoDL child receives the existing owner pipe/FD; manual FD integers
# do not constitute an authorization, and the inspection wrapper supplies none.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
python -I -B scripts/benchmarks/benchmark_t13_real_batch.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --plan "${T13_PERFORMANCE_PLAN:?absolute audited plan required}" --action inspect
