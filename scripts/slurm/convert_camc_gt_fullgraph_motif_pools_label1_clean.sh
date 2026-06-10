#!/bin/bash
#SBATCH -J conv_gt_camc_l1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

mkdir -p logs

SCRIPT=scripts/convert_camc_motif_pool_to_candidate_pool.py
METHOD=gt_fullgraph_greedy_proxy
LABEL=1

echo "===== ENV CHECK ====="
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
echo "SCRIPT=${SCRIPT}"
echo "METHOD=${METHOD}"
echo "LABEL=${LABEL}"
echo "Using clean multi-seed CAMC pools: 1594411, 1594412, 1594413"
echo "Not using old run: 1593189"
echo "====================="

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERROR] missing converter: ${SCRIPT}"
  exit 1
fi

RUN_IDS=(1594411 1594412 1594413)
for RUN_ID in "${RUN_IDS[@]}"; do
  RUN_DIR="outputs/hpc/comparison/hiv_quick/label1_${RUN_ID}"
  INPUT_CSV="${RUN_DIR}/camc_gt_fullgraph_motif_pool.csv"
  OUTPUT_JSONL="${RUN_DIR}/gt_fullgraph_candidate_pool.jsonl"
  SUMMARY_JSON="${RUN_DIR}/candidate_pool_conversion_summary.json"
  FAILED_JSONL="${RUN_DIR}/failed_rows.jsonl"

  echo "===== CONVERT label1_${RUN_ID} ====="
  if [ ! -f "${INPUT_CSV}" ]; then
    echo "[ERROR] missing input CSV: ${INPUT_CSV}"
    exit 1
  fi

  python "${SCRIPT}" \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --input-csv "${INPUT_CSV}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --label "${LABEL}" \
    --method "${METHOD}" \
    --summary-json "${SUMMARY_JSON}" \
    --failed-rows-jsonl "${FAILED_JSONL}"

  echo "===== CONVERSION SUMMARY label1_${RUN_ID} ====="
  cat "${SUMMARY_JSON}"
done

echo "===== ALL GT CAMC MOTIF POOLS CONVERTED ====="
