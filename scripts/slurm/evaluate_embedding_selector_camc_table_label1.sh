#!/bin/bash
#SBATCH -J eval_emb_camc_l1
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

SCRIPT=scripts/evaluate_embedding_selector_camc_table.py
OURS_SELECTOR_DIR=${OURS_SELECTOR_DIR:-outputs/hpc/selectors/widegrid_ours_embedding_label1/__SET_FINAL_WIDEGRID_DIR__}
GT_SELECTOR_ROOT=${GT_SELECTOR_ROOT:-outputs/hpc/selectors/widegrid_gt_fullgraph_embedding_label1_relaxed}
OUT_DIR=${OUT_DIR:-outputs/hpc/comparison/hiv_quick/embedding_selector_final_label1}
DATASET_PATH=${DATASET_PATH:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
GT_SEEDS=${GT_SEEDS:-label1_1594411,label1_1594412,label1_1594413}
THETA=${THETA:-0.20}

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
echo "OURS_SELECTOR_DIR=${OURS_SELECTOR_DIR}"
echo "GT_SELECTOR_ROOT=${GT_SELECTOR_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "GT_SEEDS=${GT_SEEDS}"
echo "THETA=${THETA}"
echo "====================="

if [[ "${OURS_SELECTOR_DIR}" == *"__SET_FINAL_WIDEGRID_DIR__"* ]]; then
  echo "[ERROR] OURS_SELECTOR_DIR is still the placeholder."
  echo "[HINT] Set it to a final widegrid directory, for example:"
  echo "  OURS_SELECTOR_DIR=outputs/hpc/selectors/widegrid_ours_embedding_label1/beta_10p0_gamma_15p0 sbatch scripts/slurm/evaluate_embedding_selector_camc_table_label1.sh"
  exit 1
fi

for path in "${SCRIPT}" "${OURS_SELECTOR_DIR}" "${GT_SELECTOR_ROOT}" "${DATASET_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

python "${SCRIPT}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ours-selector-dir "${OURS_SELECTOR_DIR}" \
  --gt-selector-root "${GT_SELECTOR_ROOT}" \
  --gt-seeds "${GT_SEEDS}" \
  --dataset-path "${DATASET_PATH}" \
  --label 1 \
  --theta "${THETA}" \
  --out-dir "${OUT_DIR}"

echo "===== FINAL EMBEDDING SELECTOR CAMC TABLE DONE ====="
ls -lh "${OUT_DIR}"
echo "===== FINAL EMBEDDING SELECTOR CAMC REPORT ====="
cat "${OUT_DIR}/final_camc_embedding_selector_report.txt"
