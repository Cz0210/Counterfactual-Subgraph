#!/usr/bin/env bash
# Static CLI-parity wrapper. The active four-by-four campaign is AutoDL-only.
# GlobalGCE and ComRecGC callers must pass an explicit audited --official-root;
# GlobalGCE exact-top-k is exposed only through `globalgce-train-rules`, which
# requires the frozen calibrated GINE checkpoint, frozen train source manifest,
# and complete processed train CSV; the adapter maps the exact frozen 869 IDs.
# The historical RF build_bace_train_pool wrapper is not an exact-route CLI.
# GINE provenance rejects the standalone `rf` token, not the letters inside
# legitimate `counterfactual*` generation fields used by candidate export.
# `globalgce-bridge-smoke` also exercises the pinned official decoder's
# negative affine edge-score domain; an identity one-hot smoke is insufficient.
# V6 may pass either a deep --gspan-adoption-proof or the controller-owned
# --gspan-mining-decision; the latter resolves to adoption or fresh exact-v2
# mining and rejects any attempt to combine both routes.
# GCFExplainer `select` and its task-fragment require an explicit
# --thresholds-json from the immutable BACE Ours B12 selector; the Python
# entrypoint validates its B11/B12 provenance and never re-fits that grid.
# This wrapper never assumes a submodule is populated;
# generic "$@" forwarding preserves exact AutoDL/Slurm CLI parity.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
exec python scripts/autodl/run_bace_baseline_gnn_route.py \
  --config configs/hpc.yaml "$@"
