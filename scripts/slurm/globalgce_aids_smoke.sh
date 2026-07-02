#!/bin/bash
#SBATCH -J globalgce_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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
RUN_ROOT=${RUN_ROOT:-outputs/hpc/globalgce/aids_smoke}
DEVICE=${DEVICE:-0}
SMOKE_TRAIN_GNN_EPOCHS=10

echo "===== GLOBALGCE AIDS SMOKE ENV CHECK ====="
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
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "GLOBALGCE_ROOT=${GLOBALGCE_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SMOKE_TRAIN_GNN_EPOCHS=${SMOKE_TRAIN_GNN_EPOCHS}"
echo "=========================================="

python scripts/baselines/globalgce/check_globalgce_layout.py \
  --config configs/hpc.yaml \
  --globalgce-root "$GLOBALGCE_ROOT"

python scripts/baselines/globalgce/run_globalgce_wrapper.py \
  --config configs/hpc.yaml \
  --globalgce-root "$GLOBALGCE_ROOT" \
  --run-root "$RUN_ROOT" \
  --dataset AIDS \
  --epochs 1 \
  --train-gnn-epochs "$SMOKE_TRAIN_GNN_EPOCHS" \
  --topk 2 \
  --exp-num 1 \
  --batch-size 64 \
  --device "$DEVICE" \
  --overwrite-run-src

echo "===== GLOBALGCE AIDS SMOKE DONE ====="
echo "RUN_ROOT=${RUN_ROOT}"
find "$RUN_ROOT" -maxdepth 3 -type f | sort
