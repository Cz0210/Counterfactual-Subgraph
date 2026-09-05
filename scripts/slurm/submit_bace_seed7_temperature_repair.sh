#!/usr/bin/env bash
# CPU-only submission utility, intentionally no GPU request.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable correction worktree required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/hpc/gnn/submit_bace_seed7_temperature_repair.py --config configs/hpc.yaml "$@"
