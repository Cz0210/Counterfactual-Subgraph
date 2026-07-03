#!/bin/bash
#SBATCH -J globalgce_aids_hiv_action_l1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

DEFAULT_AIDS_HIV_CSV=/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv
AIDS_CSV=${AIDS_CSV:-${HIV_CSV:-$DEFAULT_AIDS_HIV_CSV}}
HIV_CSV=${HIV_CSV:-$AIDS_CSV}
SMILES_COLUMN=${SMILES_COLUMN:-smiles}
LABEL_COLUMN=${LABEL_COLUMN:-HIV_active}
TARGET_LABEL=${TARGET_LABEL:-1}
DATASET_DISPLAY_NAME=${DATASET_DISPLAY_NAME:-AIDS/HIV}
DATASET=${DATASET:-aids}
TEACHER_PATH=${TEACHER_PATH:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl}
CFS_JSONL=${CFS_JSONL:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/globalgce/aids_official_top30_exported/globalgce_cfs_graphs.jsonl}
RULES_JSONL=${RULES_JSONL:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/globalgce/aids_official_top30_exported/globalgce_rules.jsonl}
RUN_ROOT=${RUN_ROOT:-outputs/hpc/globalgce/aids_official_top30}
EXPORT_DIR=${EXPORT_DIR:-outputs/hpc/globalgce/aids_official_top30_exported}
EVAL_MODE=${EVAL_MODE:-native-cf-delta-action}
OUT_DIR=${OUT_DIR:-outputs/hpc/eval/globalgce/aids_hiv_${EVAL_MODE}/label1}
CF_MODE=${CF_MODE:-strict_flip}
DISTANCE_MODE=${DISTANCE_MODE:-tanimoto}
THRESHOLDS=${THRESHOLDS:-0.05,0.10,0.20}
TOP_K=${TOP_K:-30}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

echo "===== GLOBALGCE AIDS/HIV RULE/DELTA LABEL1 ENV CHECK ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
python - <<'PY'
import importlib.util
print("torch available:", importlib.util.find_spec("torch") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
try:
    import rdkit
    print("rdkit available:", True)
except Exception as exc:
    print("rdkit available:", False, repr(exc))
PY
echo "DATASET_DISPLAY_NAME=${DATASET_DISPLAY_NAME}"
echo "DATASET=${DATASET}"
echo "AIDS_CSV=${AIDS_CSV}"
echo "HIV_CSV=${HIV_CSV}"
echo "SMILES_COLUMN=${SMILES_COLUMN}"
echo "LABEL_COLUMN=${LABEL_COLUMN}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "CFS_JSONL=${CFS_JSONL}"
echo "RULES_JSONL=${RULES_JSONL}"
echo "EVAL_MODE=${EVAL_MODE}"
echo "CF_MODE=${CF_MODE}"
echo "DISTANCE_MODE=${DISTANCE_MODE}"
echo "OUT_DIR=${OUT_DIR}"
echo "==========================================================="

python scripts/baselines/globalgce/evaluate_globalgce_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --eval-mode "$EVAL_MODE" \
  --run-root "$RUN_ROOT" \
  --export-dir "$EXPORT_DIR" \
  --dataset-display-name "$DATASET_DISPLAY_NAME" \
  --dataset-key "$DATASET" \
  --dataset-csv "$AIDS_CSV" \
  --smiles-col "$SMILES_COLUMN" \
  --label-col "$LABEL_COLUMN" \
  --target-label "$TARGET_LABEL" \
  --teacher-path "$TEACHER_PATH" \
  --cfs-jsonl "$CFS_JSONL" \
  --rules-jsonl "$RULES_JSONL" \
  --distance-mode "$DISTANCE_MODE" \
  --cf-mode "$CF_MODE" \
  --min-cf-drop "$MIN_CF_DROP" \
  --thresholds "$THRESHOLDS" \
  --top-k "$TOP_K" \
  --output-dir "$OUT_DIR"

echo "===== GLOBALGCE AIDS/HIV RULE/DELTA LABEL1 DONE ====="
cat "$OUT_DIR/report.txt"
cat "$OUT_DIR/summary.json"
