#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=audit-gnn-oracle

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python - <<'PY'
try:
    import torch
except ImportError:
    print("torch=unavailable cuda_available=false")
else:
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

: "${ORACLE_ARTIFACT:?Set ORACLE_ARTIFACT to a frozen GNN checkpoint directory or provenance JSON}"
: "${DATASET:?Set DATASET to bace or tastemolnet}"
ORACLE_AUDIT_JSON="${ORACLE_AUDIT_JSON:-outputs/hpc/oracle_audits/${DATASET}_oracle_provenance.json}"

python scripts/audit_oracle_provenance.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --artifact "$ORACLE_ARTIFACT" \
  --dataset "$DATASET" \
  --output-json "$ORACLE_AUDIT_JSON"
