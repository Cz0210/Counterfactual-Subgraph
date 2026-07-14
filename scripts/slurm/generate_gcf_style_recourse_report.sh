#!/bin/bash
#SBATCH --job-name=gcf_style_report
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
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/eval/reports/gcf_style_molclr_node_fgw_final}
DISTANCE_LABEL=${DISTANCE_LABEL:-MolCLR-Node-FGW}
TABLE_PREFIX=${TABLE_PREFIX:-}
K=${K:-10}
THETA_STAR=${THETA_STAR:-0.0545395671276376}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-0}
THRESHOLD_GRID=${THRESHOLD_GRID:-}
INSET_MAX_K=${INSET_MAX_K:-}

echo "===== GCF-STYLE RECOURSE REPORT ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DISTANCE_LABEL=${DISTANCE_LABEL}"
echo "TABLE_PREFIX=${TABLE_PREFIX}"
echo "K=${K}"
echo "THETA_STAR=${THETA_STAR}"
echo "BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES}"
echo "BOOTSTRAP_SEED=${BOOTSTRAP_SEED}"
echo "THRESHOLD_GRID=${THRESHOLD_GRID:-default_101_point_grid}"

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --output-dir "${OUTPUT_DIR}"
  --distance-label "${DISTANCE_LABEL}"
  --table-prefix "${TABLE_PREFIX}"
  --k "${K}"
  --theta-star "${THETA_STAR}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --seed "${BOOTSTRAP_SEED}"
)

if [ -n "${THRESHOLD_GRID}" ]; then
  args+=(--threshold-grid "${THRESHOLD_GRID}")
fi
if [ -n "${INSET_MAX_K}" ]; then
  args+=(--inset-max-k "${INSET_MAX_K}")
fi

python scripts/generate_gcf_style_recourse_report.py "${args[@]}"
