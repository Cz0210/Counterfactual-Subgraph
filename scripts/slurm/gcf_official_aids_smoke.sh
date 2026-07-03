#!/bin/bash
#SBATCH -J gcf_off_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

resolve_gcf_repo() {
  if [ -n "${GCF_OFFICIAL_REPO:-}" ]; then
    echo "${GCF_OFFICIAL_REPO}"
  elif [ -d "${PROJECT_ROOT}/third_party/GCFExplainer" ]; then
    echo "${PROJECT_ROOT}/third_party/GCFExplainer"
  else
    echo "${PROJECT_ROOT}/baselines/gcfexplainer_official"
  fi
}

GCF_OFFICIAL_REPO=$(resolve_gcf_repo)
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/gcfexplainer_official}
GCF_ALPHA=${GCF_ALPHA:-0.5}
GCF_TRAIN_THETA=${GCF_TRAIN_THETA:-0.05}
GCF_EVAL_THETA=${GCF_EVAL_THETA:-0.1}
GCF_MAX_STEPS=${GCF_MAX_STEPS:-200}
GCF_TOP_K_LIST=${GCF_TOP_K_LIST:-1,5,10}
GCF_TELEPORT=${GCF_TELEPORT:-0.1}
GCF_SAMPLE=${GCF_SAMPLE:-0}
GCF_SAMPLE_SIZE=${GCF_SAMPLE_SIZE:-10000}
RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}/smoke/alpha_${GCF_ALPHA}_theta_${GCF_TRAIN_THETA}_steps_${GCF_MAX_STEPS}}
SELECTED_GRAPHS_PATH="${RUN_DIR}/summary_export/selected_counterfactual_graphs.pt"

echo "===== GCF OFFICIAL AIDS SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
echo "GCF_OFFICIAL_REPO=${GCF_OFFICIAL_REPO}"
echo "RUN_DIR=${RUN_DIR}"
echo "GCF_ALPHA=${GCF_ALPHA}"
echo "GCF_TRAIN_THETA=${GCF_TRAIN_THETA}"
echo "GCF_EVAL_THETA=${GCF_EVAL_THETA}"
echo "GCF_TOP_K_LIST=${GCF_TOP_K_LIST}"
echo "CF_MODE=strict_flip"

python scripts/gcf_official_check_assets.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --official-repo "${GCF_OFFICIAL_REPO}" \
  --out-dir "${RUN_DIR}/asset_check"

VRRW_ARGS=()
if [ "${GCF_SAMPLE}" = "1" ]; then
  VRRW_ARGS+=(--sample)
fi

python scripts/gcf_official_run_vrrw.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --official-repo "${GCF_OFFICIAL_REPO}" \
  --dataset aids \
  --alpha "${GCF_ALPHA}" \
  --train-theta "${GCF_TRAIN_THETA}" \
  --max-steps "${GCF_MAX_STEPS}" \
  --teleport "${GCF_TELEPORT}" \
  --sample-size "${GCF_SAMPLE_SIZE}" \
  --device1 "${GCF_DEVICE1:-0}" \
  --device2 "${GCF_DEVICE2:-0}" \
  --run-dir "${RUN_DIR}" \
  "${VRRW_ARGS[@]}"

python scripts/gcf_official_export_summary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --run-dir "${RUN_DIR}" \
  --top-k-list "${GCF_TOP_K_LIST}" \
  --eval-theta "${GCF_EVAL_THETA}"

python scripts/evaluate_gcf_official_native.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --official-repo "${GCF_OFFICIAL_REPO}" \
  --selected-graphs-path "${SELECTED_GRAPHS_PATH}" \
  --out-dir "${RUN_DIR}/native_eval" \
  --dataset aids \
  --thresholds "${GCF_EVAL_THRESHOLDS:-0.05,0.10,0.20}" \
  --top-k-list "${GCF_TOP_K_LIST}" \
  --device "${GCF_NATIVE_DEVICE:-cuda:0}"

echo "===== GCF OFFICIAL AIDS SMOKE DONE ====="
echo "run_dir=${RUN_DIR}"

