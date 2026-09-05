#!/usr/bin/env bash
# Explicit CPU-only temperature/inference correction overrides generic A800 defaults.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?fresh immutable correction worktree required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
echo "BACE CPU correction: no weight training, no GINE change, no OT solver"
exec nice -n 10 python -I -B scripts/hpc/gnn/repair_bace_seed7_temperature_contract.py --config configs/hpc.yaml "$@"
