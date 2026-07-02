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
#SBATCH --job-name=clear_convert_pool

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

DATASET="${DATASET:-ogbg_molhiv}"
EXPORT_DIR="${EXPORT_DIR:-outputs/hpc/baselines/clear/${DATASET}/test_exports}"
OUT_JSONL="${OUT_JSONL:-outputs/hpc/baselines/clear/${DATASET}/candidate_pool/clear_${DATASET}_candidate_pool.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-outputs/hpc/baselines/clear/${DATASET}/candidate_pool/clear_${DATASET}_candidate_pool_summary.json}"
PREFER_EXP="${PREFER_EXP:-all}"
DEDUPLICATE_BY="${DEDUPLICATE_BY:-none}"
MAX_RECORDS="${MAX_RECORDS:-}"
INCLUDE_FULL_GRAPHS="${INCLUDE_FULL_GRAPHS:-0}"
FILTER_OFFICIAL_FLIP="${FILTER_OFFICIAL_FLIP:-0}"

mkdir -p logs outputs/hpc/baselines/clear/"${DATASET}"/candidate_pool

echo "===== CLEAR CANDIDATE POOL CONVERT ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "DATASET=${DATASET}"
echo "EXPORT_DIR=${EXPORT_DIR}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "OUT_SUMMARY=${OUT_SUMMARY}"
echo "PREFER_EXP=${PREFER_EXP}"
echo "DEDUPLICATE_BY=${DEDUPLICATE_BY}"
echo "MAX_RECORDS=${MAX_RECORDS}"
echo "INCLUDE_FULL_GRAPHS=${INCLUDE_FULL_GRAPHS}"
echo "FILTER_OFFICIAL_FLIP=${FILTER_OFFICIAL_FLIP}"
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
PY
echo "========================================"

args=(
  --config configs/hpc.yaml
  --export-dir "${EXPORT_DIR}"
  --dataset "${DATASET}"
  --out-jsonl "${OUT_JSONL}"
  --out-summary "${OUT_SUMMARY}"
  --prefer-exp "${PREFER_EXP}"
  --deduplicate-by "${DEDUPLICATE_BY}"
)

if [ -n "${MAX_RECORDS}" ]; then
  args+=(--max-records "${MAX_RECORDS}")
fi
if [ "${INCLUDE_FULL_GRAPHS}" = "1" ]; then
  args+=(--include-full-graphs)
fi
if [ "${FILTER_OFFICIAL_FLIP}" = "1" ]; then
  args+=(--filter-official-flip)
fi

python scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py "${args[@]}"
