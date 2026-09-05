#!/bin/bash
# This authorized read-only audit is CPU-only; it must not reserve a main GPU.
#SBATCH --job-name=gnn-temperature-audit
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
if [[ -n "${GNN_EXECUTION_WORKTREE:-}" ]]; then
  cd "$GNN_EXECUTION_WORKTREE"
  export PYTHONPATH=$PWD
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
: "${GNN_VERIFIED_ARCHIVE:?required}"
: "${GNN_VERIFIED_SHA256:?required}"
: "${GNN_TEMPERATURE_CORRECTION_RECEIPT:?fresh receipt required}"
echo "Python: $(command -v python)"
python --version
echo "GPU science disabled; offline archive and temperature-receipt verification only"
exec python scripts/hpc/gnn/audit_bace_gnn_temperature_promotion.py \
  --config configs/hpc.yaml --archive "$GNN_VERIFIED_ARCHIVE" --sha256 "$GNN_VERIFIED_SHA256" \
  --output "$GNN_TEMPERATURE_CORRECTION_RECEIPT"
