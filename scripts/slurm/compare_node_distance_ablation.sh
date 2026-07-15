#!/usr/bin/env bash
#SBATCH --job-name=node_dist_ablation
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
CONDA_ENV="${CONDA_ENV:-smiles_pip118}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hpc/eval/ablations/wnode_vs_fgw}"

source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD"
mkdir -p logs "${OUTPUT_DIR}"

echo "===== WNODE VS NODE-FGW ABLATION ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python_path=$(which python)"
echo "python_version=$(python --version 2>&1)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

methods=(Ours GlobalGCE CLEAR GCFExplainer)
args=(--output-dir "${OUTPUT_DIR}")
for method in "${methods[@]}"; do
  env_key=$(printf '%s' "${method}" | tr '[:lower:]' '[:upper:]')
  wnode_key="WNODE_${env_key}_RUN"
  fgw_key="FGW_${env_key}_RUN"
  wnode_path="${!wnode_key:-}"
  fgw_path="${!fgw_key:-}"
  if [[ -z "${wnode_path}" || -z "${fgw_path}" ]]; then
    echo "[ERROR] Set ${wnode_key} and ${fgw_key}." >&2
    exit 1
  fi
  args+=(--wnode-run "${method}=${wnode_path}")
  args+=(--fgw-run "${method}=${fgw_path}")
done

if [[ -n "${WNODE_CALIBRATION_JSON:-}" ]]; then
  args+=(--wnode-calibration-json "${WNODE_CALIBRATION_JSON}")
fi
if [[ -n "${FGW_CALIBRATION_JSON:-}" ]]; then
  args+=(--fgw-calibration-json "${FGW_CALIBRATION_JSON}")
fi

python scripts/compare_node_distance_ablation.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "${args[@]}"
