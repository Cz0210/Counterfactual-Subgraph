#!/bin/bash
#SBATCH --job-name=clear_top20_mmr
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CLEAR_CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

CANDIDATE_CSV=${CANDIDATE_CSV:-${PROJECT_ROOT}/outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv}
PARENT_CSV=${PARENT_CSV:-${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/outputs/hpc/baselines/clear/aids/selected}
REFERENCE_SELECTOR_SUMMARY=${REFERENCE_SELECTOR_SUMMARY:-${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20/selector_summary.json}
SELECTOR_CACHE_DIR=${SELECTOR_CACHE_DIR:-${PROJECT_ROOT}/outputs/hpc/cache/clear_selector_morgan}
CANDIDATE_SMILES_COL=${CANDIDATE_SMILES_COL:-candidate_smiles}
PARENT_SMILES_COL=${PARENT_SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
TARGET_LABEL=${TARGET_LABEL:-1}
TOP_K=${TOP_K:-20}
COVERAGE_METRIC=${COVERAGE_METRIC:-morgan_tanimoto}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-}
FINGERPRINT_RADIUS=${FINGERPRINT_RADIUS:-2}
FINGERPRINT_BITS=${FINGERPRINT_BITS:-2048}
SELECTOR_SEED=${SELECTOR_SEED:-13}

echo "[CLEAR_GLOBAL_SELECTOR_CONFIG]"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CANDIDATE_CSV=${CANDIDATE_CSV}"
echo "PARENT_CSV=${PARENT_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "REFERENCE_SELECTOR_SUMMARY=${REFERENCE_SELECTOR_SUMMARY}"
echo "SELECTOR_CACHE_DIR=${SELECTOR_CACHE_DIR}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "TOP_K=${TOP_K}"
echo "COVERAGE_METRIC=${COVERAGE_METRIC}"
echo "COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-required}"
echo "SELECTOR_SEED=${SELECTOR_SEED}"
echo "CF_MODE=strict_flip"

if [ ! -f "${CANDIDATE_CSV}" ]; then
  echo "[ERROR] CLEAR candidate CSV not found: ${CANDIDATE_CSV}"
  exit 2
fi
if [ ! -f "${PARENT_CSV}" ]; then
  echo "[ERROR] Parent CSV not found: ${PARENT_CSV}"
  exit 2
fi
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] RF teacher not found: ${TEACHER_PATH}"
  exit 2
fi
if [ -z "${COVERAGE_THRESHOLD}" ]; then
  echo "[ERROR] COVERAGE_THRESHOLD must be explicitly set because the Ours selector uses exact fragment support and does not define a full-molecule Tanimoto threshold."
  echo "Submit with: COVERAGE_THRESHOLD=<value_in_0_to_1> sbatch scripts/slurm/clear_aids_select_top20_greedy_mmr.sh"
  exit 2
fi

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --candidate-csv "${CANDIDATE_CSV}"
  --parent-csv "${PARENT_CSV}"
  --teacher-path "${TEACHER_PATH}"
  --out-dir "${OUT_DIR}"
  --candidate-smiles-col "${CANDIDATE_SMILES_COL}"
  --parent-smiles-col "${PARENT_SMILES_COL}"
  --label-col "${LABEL_COL}"
  --target-label "${TARGET_LABEL}"
  --top-k "${TOP_K}"
  --coverage-metric "${COVERAGE_METRIC}"
  --coverage-threshold "${COVERAGE_THRESHOLD}"
  --fingerprint-radius "${FINGERPRINT_RADIUS}"
  --fingerprint-bits "${FINGERPRINT_BITS}"
  --cache-dir "${SELECTOR_CACHE_DIR}"
  --reference-selector-summary "${REFERENCE_SELECTOR_SUMMARY}"
  --seed "${SELECTOR_SEED}"
)
if [ -n "${W_COV:-}" ]; then args+=(--w-cov "${W_COV}"); fi
if [ -n "${W_CF:-}" ]; then args+=(--w-cf "${W_CF}"); fi
if [ -n "${W_COST:-}" ]; then args+=(--w-cost "${W_COST}"); fi
if [ -n "${W_RED:-}" ]; then args+=(--w-red "${W_RED}"); fi

python scripts/baselines/clear/select_clear_global_topk.py "${args[@]}"

echo "[CLEAR_GLOBAL_SELECTOR_OUTPUTS]"
find "${OUT_DIR}" -maxdepth 1 -type f | sort
