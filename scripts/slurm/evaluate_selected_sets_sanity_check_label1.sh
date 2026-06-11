#!/bin/bash
#SBATCH -J eval_sel_sanity_l1
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

SCRIPT=scripts/evaluate_selected_sets_sanity_check.py
DATASET_PATH=${DATASET_PATH:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
OUT_DIR=${OUT_DIR:-outputs/hpc/comparison/hiv_quick/embedding_selector_sanity_check_label1}
CANDIDATE_POOL=${CANDIDATE_POOL:-outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_embeddings.jsonl}

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
try:
    from rdkit import Chem
    print("rdkit available: true")
except Exception as exc:
    print("rdkit available: false", repr(exc))
PY
echo "SCRIPT=${SCRIPT}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "CANDIDATE_POOL=${CANDIDATE_POOL}"
echo "====================="

for path in "${SCRIPT}" "${DATASET_PATH}" "${CANDIDATE_POOL}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

python "${SCRIPT}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-path "${DATASET_PATH}" \
  --label 1 \
  --theta 0.20 \
  --out-dir "${OUT_DIR}" \
  --candidate-pool-jsonl "${CANDIDATE_POOL}" \
  --selected-set old_morgan=outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20 \
  --selected-set embedding_conservative=outputs/hpc/selectors/widegrid_ours_embedding_label1/beta_20p0_gamma_5p0 \
  --selected-set embedding_lowred=outputs/hpc/selectors/widegrid_ours_embedding_label1/beta_10p0_gamma_8p0

echo "===== SELECTED SETS SANITY CHECK DONE ====="
ls -lh "${OUT_DIR}"
echo "===== SANITY CHECK REPORT ====="
cat "${OUT_DIR}/sanity_check_report.txt"
