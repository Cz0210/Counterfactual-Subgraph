#!/bin/bash
#SBATCH -J globalgce_export
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
mkdir -p logs

set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

export PYTHONPATH=$PWD
GLOBALGCE_ROOT=${GLOBALGCE_ROOT:-baselines/globalgce_official}
RUN_ROOT=${RUN_ROOT:-outputs/hpc/globalgce/aids_official_top30}
OUT_DIR=${OUT_DIR:-outputs/hpc/globalgce/aids_official_top30_exported}

echo "===== GLOBALGCE AIDS EXPORT ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
python - <<'PY'
import importlib.util
print("torch_geometric available:", importlib.util.find_spec("torch_geometric") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "GLOBALGCE_ROOT=${GLOBALGCE_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "=========================================="

python scripts/baselines/globalgce/export_globalgce_outputs.py \
  --config configs/hpc.yaml \
  --run-root "$RUN_ROOT" \
  --dataset AIDS \
  --out-dir "$OUT_DIR"

echo "===== GLOBALGCE AIDS EXPORT DONE ====="
find "$OUT_DIR" -maxdepth 2 -type f | sort
