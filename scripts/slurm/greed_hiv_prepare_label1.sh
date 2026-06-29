#!/bin/bash
#SBATCH -J greed_prepare_l1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
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

DATASET_CSV=${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
LABEL=${LABEL:-1}
OUT_JSONL=${OUT_JSONL:-outputs/hpc/greed_hiv/dataset/graphs.jsonl}
MAX_ROWS=${MAX_ROWS:-}

echo "===== GREED HIV PREPARE ENV ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import importlib.util
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("torch available:", importlib.util.find_spec("torch") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
echo "DATASET_CSV=${DATASET_CSV}"
echo "OUT_JSONL=${OUT_JSONL}"

CMD=(python scripts/prepare_hiv_greed_dataset.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --dataset-csv "${DATASET_CSV}" --output-jsonl "${OUT_JSONL}" --smiles-col "${SMILES_COL}" --label-col "${LABEL_COL}" --label "${LABEL}")
if [ -n "${MAX_ROWS}" ]; then
  CMD+=(--max-rows "${MAX_ROWS}")
fi
"${CMD[@]}"

echo "===== GREED HIV PREPARE DONE ====="
ls -lh "${OUT_JSONL}"
