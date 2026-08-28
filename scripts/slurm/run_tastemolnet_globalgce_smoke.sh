#!/usr/bin/env bash
# Static CLI parity only. TasteMolNet policy-v2 science is AutoDL-only.
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
echo "TasteMolNet T8 GlobalGCE is authorized only by the managed AutoDL controller." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python -I -B scripts/run_tastemolnet_globalgce_smoke.py \
  --config configs/hpc.yaml \
  --stage T8_GLOBALGCE_SMOKE \
  --t2-adoption /absolute/private/t2-adoption \
  --t3-output /absolute/private/T3_GINE_CALIBRATED \
  --t4-output /absolute/private/T4_ORACLE_SMOKE \
  --gnn-checkpoint /absolute/private/frozen-gine \
  --train-csv /absolute/private/train.csv \
  --official-root /absolute/private/GlobalGCE \
  --downstream-policy configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json \
  --base-policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --state-dir /absolute/fresh/private/t8-globalgce-state \
  --output-dir /absolute/fresh/private/t8-globalgce-smoke \
  --set inference.fallback_to_heuristic=false
