#!/bin/bash
#SBATCH -J merge_s300_b_t07
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

BASE_POOL=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/stable300_label1_n4/candidate_pool.jsonl
TEMP_POOL=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/stable300_label1_n4_temp07_topp09/candidate_pool.jsonl
OUT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07
OUT_POOL=${OUT_DIR}/candidate_pool.jsonl
MERGE_SUMMARY=${OUT_DIR}/merge_summary.json
AUDIT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/audits/stable300_label1_merged_base_temp07_audit
MERGE_SCRIPT=scripts/merge_candidate_pools.py

mkdir -p "${OUT_DIR}"
mkdir -p "${AUDIT_DIR}"

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
echo "BASE_POOL=${BASE_POOL}"
echo "TEMP_POOL=${TEMP_POOL}"
echo "OUT_POOL=${OUT_POOL}"
echo "MERGE_SUMMARY=${MERGE_SUMMARY}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "MERGE_SCRIPT=${MERGE_SCRIPT}"
echo "====================="

for p in "${BASE_POOL}" "${TEMP_POOL}" "${MERGE_SCRIPT}" scripts/audit_candidate_pool.py; do
  if [ ! -e "${p}" ]; then
    echo "[ERROR] missing path: ${p}"
    exit 1
  fi
done

python "${MERGE_SCRIPT}" --help || true

python "${MERGE_SCRIPT}" \
  --config configs/hpc.yaml \
  --pool-jsonl "${BASE_POOL}" \
  --pool-jsonl "${TEMP_POOL}" \
  --out-jsonl "${OUT_POOL}" \
  --out-summary-json "${MERGE_SUMMARY}" \
  --dedup-key final_fragment,parent_smiles \
  --keep-best-by reward_total

echo "===== MERGE CHECK ====="
ls -lh "${OUT_POOL}" "${MERGE_SUMMARY}"
wc -l "${OUT_POOL}"
cat "${MERGE_SUMMARY}"

python scripts/audit_candidate_pool.py --help || true

python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "${OUT_POOL}" \
  --out_json "${AUDIT_DIR}/audit_summary.json" \
  --out_txt "${AUDIT_DIR}/audit_report.txt" \
  --group_by_label \
  --sim_sample_size 10000 \
  --topk_show 30

echo "===== AUDIT DONE ====="
ls -lh "${AUDIT_DIR}"
echo "===== AUDIT REPORT ====="
cat "${AUDIT_DIR}/audit_report.txt"
