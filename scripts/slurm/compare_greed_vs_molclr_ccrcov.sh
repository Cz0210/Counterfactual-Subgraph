#!/bin/bash
#SBATCH -J cmp_greed_molclr
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

COMPARE_MODE=${COMPARE_MODE:-smoke}
case "${COMPARE_MODE}" in
  smoke)
    DEFAULT_GREED_SUMMARY="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_greed_hiv_smoke/combined/combined_threshold_summary.csv"
    DEFAULT_MOLCLR_SUMMARY="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_molclr_hiv_smoke/combined/combined_threshold_summary.csv"
    DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_compare_smoke"
    ;;
  full)
    DEFAULT_GREED_SUMMARY="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_greed_hiv_full/combined/combined_threshold_summary.csv"
    DEFAULT_MOLCLR_SUMMARY="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_molclr_hiv_full/combined/combined_threshold_summary.csv"
    DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_compare_full"
    ;;
  *)
    echo "[ERROR] COMPARE_MODE must be smoke or full, got: ${COMPARE_MODE}"
    exit 2
    ;;
esac

GREED_SUMMARY=${GREED_SUMMARY:-${DEFAULT_GREED_SUMMARY}}
MOLCLR_SUMMARY=${MOLCLR_SUMMARY:-${DEFAULT_MOLCLR_SUMMARY}}
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}

echo "[COMPARE_CONFIG]"
echo "COMPARE_MODE=${COMPARE_MODE}"
echo "GREED_SUMMARY=${GREED_SUMMARY}"
echo "MOLCLR_SUMMARY=${MOLCLR_SUMMARY}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if [ ! -f "${GREED_SUMMARY}" ] || [ ! -f "${MOLCLR_SUMMARY}" ]; then
  echo "[ERROR] Missing compare summary file."
  echo "GREED_SUMMARY exists: $([ -f "${GREED_SUMMARY}" ] && echo yes || echo no)"
  echo "MOLCLR_SUMMARY exists: $([ -f "${MOLCLR_SUMMARY}" ] && echo yes || echo no)"
  echo "Hint:"
  echo 'find outputs/hpc -type f \( -name "combined_threshold_summary.csv" -o -name "threshold_summary.csv" \) | grep -Ei "greed|molclr|ccrcov"'
  exit 2
fi

echo "===== COMPARE GREED VS MOLCLR ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version

python scripts/compare_greed_vs_molclr_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --greed-summary "${GREED_SUMMARY}" \
  --molclr-summary "${MOLCLR_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}"

echo "===== COMPARE GREED VS MOLCLR DONE ====="
find "${OUTPUT_DIR}" -type f | sort
