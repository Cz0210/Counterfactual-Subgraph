#!/bin/bash
#SBATCH -J greed_train_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

PAIRS_DIR=${PAIRS_DIR:-outputs/hpc/greed_hiv/pairs_smoke}
CHECKPOINT=${CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged_smoke.pt}
TRAIN_METRICS=${TRAIN_METRICS:-outputs/hpc/greed_hiv/reports/train_metrics_smoke.json}
TEST_METRICS=${TEST_METRICS:-outputs/hpc/greed_hiv/reports/test_metrics_smoke.csv}

echo "===== GREED HIV TRAIN SMOKE ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

python scripts/train_hiv_greed_distance.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pairs-dir "${PAIRS_DIR}" \
  --checkpoint-path "${CHECKPOINT}" \
  --train-metrics-json "${TRAIN_METRICS}" \
  --test-metrics-csv "${TEST_METRICS}" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --patience "${PATIENCE:-2}" \
  --device "${DEVICE:-cuda}"

echo "===== GREED HIV TRAIN SMOKE DONE ====="
ls -lh "${CHECKPOINT}" "${TRAIN_METRICS}" "${TEST_METRICS}"
