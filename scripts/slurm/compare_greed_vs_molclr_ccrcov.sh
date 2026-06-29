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

GREED_SUMMARY=${GREED_SUMMARY:-outputs/hpc/eval/ccrcov_greed_hiv_full/combined/combined_threshold_summary.csv}
MOLCLR_SUMMARY=${MOLCLR_SUMMARY:-outputs/hpc/eval/ccrcov_molclr_hiv_full/combined/combined_threshold_summary.csv}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval/ccrcov_distance_comparison}

echo "===== COMPARE GREED VS MOLCLR ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
echo "GREED_SUMMARY=${GREED_SUMMARY}"
echo "MOLCLR_SUMMARY=${MOLCLR_SUMMARY}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python scripts/compare_greed_vs_molclr_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --greed-summary "${GREED_SUMMARY}" \
  --molclr-summary "${MOLCLR_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}"

echo "===== COMPARE GREED VS MOLCLR DONE ====="
find "${OUTPUT_DIR}" -type f | sort
