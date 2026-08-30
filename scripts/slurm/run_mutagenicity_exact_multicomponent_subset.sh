#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=4

: "${MUT_PAIR_STORE_MANIFEST:?absolute completed pair-store manifest is required}"
: "${MUT_EXACT_SUBSET_OUTPUT:?absolute fresh subset output is required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/run_mutagenicity_exact_multicomponent_subset.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pair-store-manifest "$MUT_PAIR_STORE_MANIFEST" \
  --output-dir "$MUT_EXACT_SUBSET_OUTPUT" \
  --subset-count "${MUT_EXACT_SUBSET_COUNT:-3}" \
  --subset-size "${MUT_EXACT_SUBSET_SIZE:-2048}" \
  --expected-sklearn-version "${COMRECGC_EXPECTED_SKLEARN_VERSION:-1.7.2}"
