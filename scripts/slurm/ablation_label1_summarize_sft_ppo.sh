#!/bin/bash
#SBATCH -J abl_l1_summary
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

ROOT=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/ablations/label1_sft_ppo
OUT_CSV=${ROOT}/ablation_summary.csv
OUT_MD=${ROOT}/ablation_summary.md
SUMMARY_SCRIPT=scripts/summarize_label1_sft_ppo_ablation.py

mkdir -p "${ROOT}"

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
echo "ROOT=${ROOT}"
echo "OUT_CSV=${OUT_CSV}"
echo "OUT_MD=${OUT_MD}"
echo "SUMMARY_SCRIPT=${SUMMARY_SCRIPT}"
echo "====================="

if [ ! -f "${SUMMARY_SCRIPT}" ]; then
  echo "[ERROR] summary script not found: ${SUMMARY_SCRIPT}"
  exit 1
fi

python "${SUMMARY_SCRIPT}" \
  --config configs/hpc.yaml \
  --root-dir "${ROOT}" \
  --out-csv "${OUT_CSV}" \
  --out-md "${OUT_MD}"

echo "===== SUMMARY CSV ====="
ls -lh "${OUT_CSV}" "${OUT_MD}"
echo "===== SUMMARY MARKDOWN ====="
cat "${OUT_MD}"
