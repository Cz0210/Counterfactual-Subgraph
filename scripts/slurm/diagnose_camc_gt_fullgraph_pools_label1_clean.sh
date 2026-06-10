#!/bin/bash
#SBATCH -J diag_gt_pool_l1
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

SCRIPT=scripts/diagnose_candidate_pool_for_selector.py
OUT_ROOT=outputs/hpc/diagnostics/camc_gt_fullgraph_label1
RUN_IDS=(1594411 1594412 1594413)

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
echo "OUT_ROOT=${OUT_ROOT}"
echo "RUN_IDS=${RUN_IDS[*]}"
echo "====================="

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERROR] missing diagnosis script: ${SCRIPT}"
  exit 1
fi

for RUN_ID in "${RUN_IDS[@]}"; do
  RUN_DIR="outputs/hpc/comparison/hiv_quick/label1_${RUN_ID}"
  DIAG_DIR="${OUT_ROOT}/label1_${RUN_ID}"
  mkdir -p "${DIAG_DIR}"

  POOL="${RUN_DIR}/gt_fullgraph_candidate_pool.jsonl"
  if [ -f "${POOL}" ]; then
    echo "===== DIAGNOSE ${POOL} ====="
    python "${SCRIPT}" \
      --pool-jsonl "${POOL}" \
      --label 1 \
      --embedding-field final_fragment_embedding \
      --out-json "${DIAG_DIR}/candidate_pool_diagnosis.json" \
      --out-txt "${DIAG_DIR}/candidate_pool_diagnosis.txt"
  else
    echo "MISSING: ${POOL}"
  fi

  EMB_POOL="${RUN_DIR}/gt_fullgraph_candidate_pool_with_embeddings.jsonl"
  if [ -f "${EMB_POOL}" ]; then
    echo "===== DIAGNOSE ${EMB_POOL} ====="
    python "${SCRIPT}" \
      --pool-jsonl "${EMB_POOL}" \
      --label 1 \
      --embedding-field final_fragment_embedding \
      --out-json "${DIAG_DIR}/candidate_pool_with_embeddings_diagnosis.json" \
      --out-txt "${DIAG_DIR}/candidate_pool_with_embeddings_diagnosis.txt"
  else
    echo "MISSING: ${EMB_POOL}"
  fi
done

echo "===== GT FULLGRAPH POOL DIAGNOSIS DONE ====="
find "${OUT_ROOT}" -maxdepth 3 -type f | sort
