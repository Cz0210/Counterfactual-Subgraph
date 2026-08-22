#!/usr/bin/env bash
# Static CLI-parity wrapper. The active four-by-four campaign is AutoDL-only.
# GlobalGCE and ComRecGC preflight callers must pass an explicit audited
# --official-root; this wrapper never assumes an upstream checkout/submodule is
# populated.  The existing generic "$@" forwarding needs no CLI shape change.
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
