#!/bin/bash
#SBATCH -J abl_l1_base_sel
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

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
ROOT=${PROJECT_DIR}/outputs/hpc/ablations/label1_sft_ppo
POOL=${ROOT}/base_chemmllm_n4/candidate_pool.jsonl
OUT_DIR=${ROOT}/selectors/base_chemmllm_n4_cov20
SELECTOR=scripts/select_class_counterfactual_subgraphs.py

mkdir -p "${OUT_DIR}"

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
echo "POOL=${POOL}"
echo "OUT_DIR=${OUT_DIR}"
echo "SELECTOR=${SELECTOR}"
echo "====================="

if [ ! -f "${POOL}" ]; then
  echo "[ERROR] pool not found: ${POOL}"
  echo "Please run: sbatch scripts/slurm/ablation_label1_generate_a_base_chemmllm_n4.sh"
  exit 1
fi

if [ ! -f "${SELECTOR}" ]; then
  echo "[ERROR] selector not found: ${SELECTOR}"
  exit 1
fi

python "${SELECTOR}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pool-jsonl "${POOL}" \
  --out-dir "${OUT_DIR}" \
  --label 1 \
  --top-k 20 \
  --alpha-cf 0.8 \
  --beta-coverage 20.0 \
  --gamma-redundancy 0.7 \
  --eta-size 0.3 \
  --min-cf-drop 0.2 \
  --require-cf-flip \
  --require-final-substructure \
  --sim-metric morgan \
  --top-candidates-per-fragment 3 \
  --dedup-by-final-fragment

echo "===== SELECTOR DONE ====="
ls -lh "${OUT_DIR}"
echo "===== SELECTOR REPORT ====="
cat "${OUT_DIR}/selector_report.txt"
