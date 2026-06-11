#!/bin/bash
#SBATCH -J sum_legacy_camc_l1
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

SCRIPT=scripts/summarize_legacy_camc_embedding_selector_eval.py
ROOT=${ROOT:-outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1}

echo "===== LEGACY CAMC SUMMARY ENV CHECK ====="
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
echo "ROOT=${ROOT}"
echo "========================================="

for path in "${SCRIPT}" "${ROOT}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    echo "[HINT] Run scripts/slurm/eval_hiv_quick_embedding_selectors_label1_legacy_camc.sh first."
    exit 1
  fi
done

python "${SCRIPT}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --root "${ROOT}"

echo "===== LEGACY CAMC EMBEDDING SELECTOR SUMMARY DONE ====="
ls -lh "${ROOT}"/legacy_camc_embedding_selector_summary.*
echo "===== LEGACY CAMC EMBEDDING SELECTOR REPORT ====="
cat "${ROOT}/legacy_camc_embedding_selector_report.txt"
