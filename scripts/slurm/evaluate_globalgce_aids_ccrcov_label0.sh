#!/bin/bash
#SBATCH -J globalgce_l0_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
mkdir -p logs

set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

export PYTHONPATH=$PWD
GLOBALGCE_ROOT=${GLOBALGCE_ROOT:-baselines/globalgce_official}
RUN_ROOT=${RUN_ROOT:-outputs/hpc/globalgce/aids_official_top30}
EXPORT_DIR=${EXPORT_DIR:-outputs/hpc/globalgce/aids_official_top30_exported}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval/globalgce/aids_official_top30/label0}
LABEL=${LABEL:-0}
K=${K:-30}
THRESHOLDS=${THRESHOLDS:-0.05,0.10,0.20}
EVAL_MODE=${EVAL_MODE:-native-cf}

if [ -z "${TEACHER_PATH:-}" ]; then
  for candidate in \
    outputs/hpc/oracle/aids_rf_model.pkl \
    outputs/oracle/aids_rf_model.pkl \
    outputs/hpc/teacher/aids_rf_model.pkl; do
    if [ -f "$candidate" ]; then
      TEACHER_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "${TEACHER_PATH:-}" ] || [ ! -f "$TEACHER_PATH" ]; then
  echo "[ERROR] TEACHER_PATH is required for GlobalGCE CCRCov evaluation." >&2
  exit 2
fi

echo "===== GLOBALGCE AIDS CCRCov LABEL0 ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
python - <<'PY'
import importlib.util
print("torch_geometric available:", importlib.util.find_spec("torch_geometric") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "RUN_ROOT=${RUN_ROOT}"
echo "EXPORT_DIR=${EXPORT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "EVAL_MODE=${EVAL_MODE}"
echo "=================================================="

python scripts/baselines/globalgce/evaluate_globalgce_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --eval-mode "$EVAL_MODE" \
  --run-root "$RUN_ROOT" \
  --export-dir "$EXPORT_DIR" \
  --dataset AIDS \
  --label "$LABEL" \
  --k "$K" \
  --thresholds "$THRESHOLDS" \
  --teacher-path "$TEACHER_PATH" \
  --output-dir "$OUTPUT_DIR"

echo "===== GLOBALGCE AIDS CCRCov LABEL0 DONE ====="
cat "$OUTPUT_DIR/report.txt"
cat "$OUTPUT_DIR/summary.json"
