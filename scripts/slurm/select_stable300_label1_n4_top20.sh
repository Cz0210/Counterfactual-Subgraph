#!/bin/bash
#SBATCH -J sel_stable300_n4
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

POOL=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/stable300_label1_n4/candidate_pool.jsonl
OUT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/selectors/stable300_label1_n4_top20_mmr
SELECTOR_SCRIPT=scripts/select_class_counterfactual_subgraphs.py

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
echo "SELECTOR_SCRIPT=${SELECTOR_SCRIPT}"
echo "====================="

for p in "${POOL}" "${SELECTOR_SCRIPT}"; do
  if [ ! -e "${p}" ]; then
    echo "[ERROR] missing path: ${p}"
    exit 1
  fi
done

python "${SELECTOR_SCRIPT}" --help || true

python "${SELECTOR_SCRIPT}" \
  --config configs/hpc.yaml \
  --pool-jsonl "${POOL}" \
  --out-dir "${OUT_DIR}" \
  --label 1 \
  --top-k 20 \
  --alpha-cf 1.0 \
  --beta-coverage 1.0 \
  --gamma-redundancy 0.7 \
  --eta-size 0.3 \
  --min-cf-drop 0.2 \
  --require-cf-flip \
  --require-final-substructure \
  --sim-metric morgan \
  --top-candidates-per-fragment 3 \
  --dedup-by-final-fragment

echo "===== SELECTOR OUTPUTS ====="
ls -lh "${OUT_DIR}"
echo "===== SELECTOR REPORT ====="
cat "${OUT_DIR}/selector_report.txt"
