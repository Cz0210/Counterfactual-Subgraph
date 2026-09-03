#!/usr/bin/env bash
# CPU-only override: deterministic parity verification uses no GPU.
#SBATCH --job-name=t8-exact-parity
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
: "${T8_EXECUTION_WORKTREE:?T8_EXECUTION_WORKTREE is required}"
: "${T8_EXPECTED_COMMIT:?T8_EXPECTED_COMMIT is required}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
[[ "$(git rev-parse HEAD)" == "$T8_EXPECTED_COMMIT" ]] || { echo "execution commit mismatch" >&2; exit 65; }
echo "python=$(command -v python)"
python --version
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
python scripts/hpc/t8/verify_exact_parity.py --config configs/hpc.yaml "$@"
