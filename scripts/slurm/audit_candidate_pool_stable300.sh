#!/bin/bash
#SBATCH -J audit_stable300_pool
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

RUN_NAME=decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500
POOL=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/rl_checkpoints/${RUN_NAME}/candidate_pool.jsonl
OUT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/audits/${RUN_NAME}_candidate_pool_audit
AUDIT_SCRIPT=scripts/audit_candidate_pool.py

mkdir -p logs
mkdir -p "${OUT_DIR}"

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
echo "POOL=${POOL}"
echo "OUT_DIR=${OUT_DIR}"
echo "AUDIT_SCRIPT=${AUDIT_SCRIPT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "====================="

if [ ! -f "${POOL}" ]; then
  echo "[ERROR] candidate_pool not found: ${POOL}"
  exit 1
fi

if [ ! -f "${AUDIT_SCRIPT}" ]; then
  echo "[ERROR] audit script not found: ${AUDIT_SCRIPT}"
  find scripts -name "*audit*candidate*pool*.py" -o -name "*candidate*pool*audit*.py" || true
  exit 1
fi

echo "===== AUDIT SCRIPT HELP ====="
python "${AUDIT_SCRIPT}" --help || true
echo "============================="

echo "===== RUNNING STABLE300 CANDIDATE POOL AUDIT ====="

python "${AUDIT_SCRIPT}" \
  --config configs/hpc.yaml \
  --pool_jsonl "${POOL}" \
  --out_json "${OUT_DIR}/audit_summary.json" \
  --out_txt "${OUT_DIR}/audit_report.txt" \
  --group_by_label \
  --sim_sample_size 5000 \
  --topk_show 20

echo "===== AUDIT DONE ====="
ls -lh "${OUT_DIR}"
echo "===== AUDIT REPORT ====="
cat "${OUT_DIR}/audit_report.txt"
