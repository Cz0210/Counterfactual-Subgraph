#!/bin/bash
#SBATCH -J sum_emb_l1_rx
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

SCRIPT=scripts/summarize_embedding_selector_sweeps.py
OURS_ROOT=outputs/hpc/selectors/param_sweep_ours_embedding_cov20
GT_ROOT=outputs/hpc/selectors/param_sweep_gt_fullgraph_embedding_cov20_relaxed
OUT_DIR=outputs/hpc/selectors/embedding_selector_sweep_comparison_label1_relaxed

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
echo "OURS_ROOT=${OURS_ROOT}"
echo "GT_ROOT=${GT_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "====================="

for path in "${SCRIPT}" "${OURS_ROOT}" "${GT_ROOT}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    echo "[HINT] Run ours and relaxed GT embedding selector sweep sbatch jobs first."
    exit 1
  fi
done

python "${SCRIPT}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ours-root "${OURS_ROOT}" \
  --gt-root "${GT_ROOT}" \
  --out-dir "${OUT_DIR}"

echo "===== RELAXED EMBEDDING SELECTOR SWEEP COMPARISON DONE ====="
ls -lh "${OUT_DIR}"
echo "===== RELAXED COMPARISON REPORT ====="
cat "${OUT_DIR}/comparison_report.txt"
