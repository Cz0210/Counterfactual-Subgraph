#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=prepare-tastemolnet

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

: "${TASTEMOLNET_CSV:?Set TASTEMOLNET_CSV to an existing, local source CSV}"
TASTEMOLNET_SOURCE_MODE="${TASTEMOLNET_SOURCE_MODE:-upstream_processed}"
TASTEMOLNET_OUTPUT_DIR="${TASTEMOLNET_OUTPUT_DIR:-data/processed/tastemolnet}"
TASTEMOLNET_SOURCE_URL="${TASTEMOLNET_SOURCE_URL:-https://github.com/MujeebOnawole/Taste_Prediction_RGCN}"
TASTEMOLNET_UPSTREAM_COMMIT="${TASTEMOLNET_UPSTREAM_COMMIT:-16af8ead8a17b6bd3941d9eb5879c5be75c14114}"

args=(
  python scripts/prepare_tastemolnet.py
  --config configs/hpc.yaml
  --input-csv "$TASTEMOLNET_CSV"
  --source-mode "$TASTEMOLNET_SOURCE_MODE"
  --output-dir "$TASTEMOLNET_OUTPUT_DIR"
)

if [[ -n "${TASTEMOLNET_SOURCE_URL:-}" ]]; then
  args+=(--source-url "$TASTEMOLNET_SOURCE_URL")
fi
if [[ -n "${TASTEMOLNET_UPSTREAM_COMMIT:-}" ]]; then
  args+=(--upstream-commit "$TASTEMOLNET_UPSTREAM_COMMIT")
fi
if [[ -n "${TASTEMOLNET_LICENSE_ID:-}" ]]; then
  args+=(--license-id "$TASTEMOLNET_LICENSE_ID")
fi
if [[ "${TASTEMOLNET_LICENSE_REVIEWED:-0}" == "1" ]]; then
  args+=(--license-reviewed)
fi

"${args[@]}"
