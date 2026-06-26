#!/bin/bash
#SBATCH -J close_cf_plot
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

if [ -z "${SUMMARY_CSVS:-}" ]; then
  echo "[ERROR] SUMMARY_CSVS is required, comma-separated threshold_summary.csv paths."
  exit 2
fi
if [ -z "${LABELS:-}" ]; then
  echo "[ERROR] LABELS is required, for example LABELS=Ours-GED,GCF-GED,Ours-Emb,GCF-Emb."
  exit 2
fi
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval_close_cf_coverage/plots_$(date +%Y%m%d_%H%M%S)}
TITLE_PREFIX=${TITLE_PREFIX:-Close CF Coverage}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "python path: $(which python)"
python --version
python - <<'PY'
import importlib.util
print("matplotlib available:", importlib.util.find_spec("matplotlib") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
echo "SUMMARY_CSVS=${SUMMARY_CSVS}"
echo "LABELS=${LABELS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "====================="

python scripts/plot_close_counterfactual_coverage.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --summary-csvs "${SUMMARY_CSVS}" \
  --labels "${LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --title-prefix "${TITLE_PREFIX}"

find "${OUTPUT_DIR}/figures" -type f | sort
