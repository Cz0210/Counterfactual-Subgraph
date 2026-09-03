#!/usr/bin/env bash
# Paired wrapper for scripts/hpc/t8/build_input_bundle.py.
# CPU-only is an intentional task-specific override: packaging performs no GPU work.
#SBATCH --job-name=t8-input-bundle
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
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
python scripts/hpc/t8/build_input_bundle.py --config configs/hpc.yaml "$@"
