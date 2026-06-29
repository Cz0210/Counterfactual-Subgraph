#!/bin/bash
#SBATCH -J greed_train_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
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
  --pairs-dir "${PAIRS_DIR:-outputs/hpc/greed_hiv/pairs}" \
  --checkpoint-path "${CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}" \
  --train-metrics-json "${TRAIN_METRICS:-outputs/hpc/greed_hiv/reports/train_metrics.json}" \
  --test-metrics-csv "${TEST_METRICS:-outputs/hpc/greed_hiv/reports/test_metrics.csv}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-layers "${NUM_LAYERS:-8}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --lr "${LR:-0.001}" \
  --patience "${PATIENCE:-10}" \
  --device "${DEVICE:-cuda}"

echo "===== GREED HIV TRAIN FULL DONE ====="
