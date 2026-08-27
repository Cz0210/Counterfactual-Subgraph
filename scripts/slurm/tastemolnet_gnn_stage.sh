#!/bin/bash
# Static CLI parity only. The scoped TasteMolNet downstream route is AutoDL-only.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "TasteMolNet T3/T4 research execution is authorized only on the reviewed AutoDL GPU1 route; HPC remains forbidden." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python -B scripts/autodl/tastemolnet_gnn_stage.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  t3-adopt \
  --checkpoint-dir /absolute/immutable/t2-bundle \
  --t2-adoption-root /absolute/runtime/control/tastemolnet-t2-gine-pass-adoption-v1/source-cid \
  --t2-adoption-gate-sha256 REVIEWED_64_HEX_GATE_SHA256 \
  --t2-adoption-receipt-sha256 REVIEWED_64_HEX_RECEIPT_SHA256 \
  --t2-source-evidence-sha256 REVIEWED_64_HEX_SOURCE_EVIDENCE_SHA256 \
  --graph-cache-root /absolute/private/graph-cache \
  --artifact-root /absolute/artifact-root \
  --output-dir /absolute/artifact-root/gnn_oracles/tastemolnet/gine/seed7/calibrated-documentation \
  --downstream-policy configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json \
  --base-policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml

python -B scripts/autodl/tastemolnet_gnn_stage.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  t4-oracle-smoke \
  --checkpoint-dir /absolute/immutable/t2-bundle \
  --t2-adoption-root /absolute/runtime/control/tastemolnet-t2-gine-pass-adoption-v1/source-cid \
  --t2-adoption-gate-sha256 REVIEWED_64_HEX_GATE_SHA256 \
  --t2-adoption-receipt-sha256 REVIEWED_64_HEX_RECEIPT_SHA256 \
  --t2-source-evidence-sha256 REVIEWED_64_HEX_SOURCE_EVIDENCE_SHA256 \
  --t3-gate /absolute/artifact-root/gnn_oracles/tastemolnet/gine/seed7/calibrated-documentation/gate.json \
  --graph-cache-root /absolute/private/graph-cache \
  --artifact-root /absolute/artifact-root \
  --output-dir /absolute/artifact-root/gnn_oracles/tastemolnet/gine/seed7/t4-oracle-smoke-documentation \
  --downstream-policy configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json \
  --base-policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --physical-gpu-index 1 \
  --gpu-uuid GPU-REQUIRED \
  --device cuda:0 \
  --batch-size 32 \
  --source-count 16 \
  --max-deletions-per-parent 4
