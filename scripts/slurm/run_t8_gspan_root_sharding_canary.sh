#!/usr/bin/env bash
#SBATCH --job-name=t8-gspan-root-shard-canary
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

: "${T8_GSPAN_GRAPHS_JSONL:?set a frozen real graph JSONL}"
: "${T8_GSPAN_OFFICIAL_SRC:?set the pinned official GlobalGCE src directory}"
: "${T8_GSPAN_CANARY_ROOT:?set a fresh persistent output root}"
: "${T8_GSPAN_ROOT_INDICES:?set comma-separated small and large root indices}"
: "${T8_GSPAN_MIN_SUPPORT:?set the frozen production min_support}"

scratch_args=()
if [[ -n "${T8_GSPAN_SCRATCH_ROOT:-}" ]]; then
  scratch_args=(--scratch-root "$T8_GSPAN_SCRATCH_ROOT")
fi

python -B scripts/autodl/run_t8_gspan_root_sharding_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --graphs-jsonl "$T8_GSPAN_GRAPHS_JSONL" \
  --official-src "$T8_GSPAN_OFFICIAL_SRC" \
  --output-root "$T8_GSPAN_CANARY_ROOT" \
  "${scratch_args[@]}" \
  --root-indices "$T8_GSPAN_ROOT_INDICES" \
  --shard-count "${T8_GSPAN_SHARD_COUNT:-2}" \
  --min-support "$T8_GSPAN_MIN_SUPPORT" \
  --min-vertices "${T8_GSPAN_MIN_VERTICES:-3}" \
  --max-vertices "${T8_GSPAN_MAX_VERTICES:-20}" \
  --top-k "${T8_GSPAN_TOP_K:-20}"
