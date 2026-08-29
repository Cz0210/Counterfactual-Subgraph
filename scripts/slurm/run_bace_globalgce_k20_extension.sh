#!/usr/bin/env bash
#SBATCH --job-name=bace-globalgce-k20
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

# This recovery route owns physical AutoDL GPU index 2.  Slurm allocates an
# arbitrary visible GPU and therefore cannot prove that physical identity.
# The AutoDL thin CLI installs its deferred signal mask before science imports.
echo "BLOCKED_STATIC_REFUSAL: use the reviewed AutoDL physical-GPU2 controller route" >&2
exit 75

# Documentation-only CLI parity; deliberately unreachable under Slurm.
python scripts/autodl/run_bace_globalgce_k20_extension.py \
  --config "$PWD/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  controller \
  "$@"
