#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=clear_eval_pool

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

DATASET="${DATASET:-ogbg_molhiv}"
CANDIDATE_POOL="${CANDIDATE_POOL:-outputs/hpc/baselines/clear/${DATASET}/candidate_pool/clear_${DATASET}_candidate_pool.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/hpc/baselines/clear/${DATASET}/eval}"
TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}"
CF_MODE="${CF_MODE:-strict_flip}"
MIN_CF_DROP="${MIN_CF_DROP:-0.0}"
TOP_K="${TOP_K:-1,5,10,20}"
THRESHOLDS="${THRESHOLDS:-5,10,20,50,100,200}"
DISTANCE_METHOD="${DISTANCE_METHOD:-action}"
DEDUPLICATE_BY="${DEDUPLICATE_BY:-none}"
RANK_BY="${RANK_BY:-total_cost}"
MAX_CANDIDATES="${MAX_CANDIDATES:-}"
ALLOW_ACTION_ONLY="${ALLOW_ACTION_ONLY:-0}"

mkdir -p logs "${OUT_DIR}"

echo "===== CLEAR UNIFIED EVAL ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATASET=${DATASET}"
echo "CANDIDATE_POOL=${CANDIDATE_POOL}"
echo "OUT_DIR=${OUT_DIR}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "CF_MODE=${CF_MODE}"
echo "MIN_CF_DROP=${MIN_CF_DROP}"
echo "TOP_K=${TOP_K}"
echo "THRESHOLDS=${THRESHOLDS}"
echo "DISTANCE_METHOD=${DISTANCE_METHOD}"
echo "DEDUPLICATE_BY=${DEDUPLICATE_BY}"
echo "RANK_BY=${RANK_BY}"
echo "MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "ALLOW_ACTION_ONLY=${ALLOW_ACTION_ONLY}"
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("device name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "=============================="

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --candidate-pool "${CANDIDATE_POOL}"
  --dataset "${DATASET}"
  --teacher-path "${TEACHER_PATH}"
  --out-dir "${OUT_DIR}"
  --cf-mode "${CF_MODE}"
  --min-cf-drop "${MIN_CF_DROP}"
  --top-k "${TOP_K}"
  --thresholds "${THRESHOLDS}"
  --distance-method "${DISTANCE_METHOD}"
  --deduplicate-by "${DEDUPLICATE_BY}"
  --rank-by "${RANK_BY}"
)

if [ -n "${MAX_CANDIDATES}" ]; then
  args+=(--max-candidates "${MAX_CANDIDATES}")
fi
if [ "${ALLOW_ACTION_ONLY}" = "1" ]; then
  args+=(--allow-action-only)
fi

python scripts/baselines/clear/evaluate_clear_candidate_pool.py "${args[@]}"
