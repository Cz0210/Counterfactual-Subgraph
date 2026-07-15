#!/usr/bin/env bash
#SBATCH --job-name=wnode_ccrcov
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
CLEAR_ENV="${CLEAR_CONDA_ENV:-smiles_pip118}"
TEACHER_PATH="${TEACHER_PATH:-${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl}"
DATASET_CSV="${DATASET_CSV:-${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
MOLCLR_ROOT="${MOLCLR_ROOT:-${PROJECT_ROOT}/pretrained_models/MolCLR}"
MOLCLR_CKPT="${MOLCLR_CKPT:-${MOLCLR_ROOT}/ckpt/pretrained_gin/checkpoints/model.pth}"
OURS_SELECTED_PATH="${OURS_SELECTED_PATH:-${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}"
FULLGRAPH_CANDIDATES_PATH="${FULLGRAPH_CANDIDATES_PATH:-}"
FULLGRAPH_METHOD_NAME="${FULLGRAPH_METHOD_NAME:-fullgraph_preselected_top20}"
SELECTION_METHOD="${SELECTION_METHOD:-external_preselected_top20}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_molclr_node_wasserstein_smoke}"
WNODE_CACHE_DB="${WNODE_CACHE_DB:-${PROJECT_ROOT}/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_v1.sqlite}"
NODE_EMB_CACHE_DIR="${NODE_EMB_CACHE_DIR:-${PROJECT_ROOT}/outputs/hpc/cache/molclr_node_embeddings}"
MAX_PARENTS="${MAX_PARENTS:-200}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
RUN_OURS="${RUN_OURS:-1}"
RUN_FULLGRAPH="${RUN_FULLGRAPH:-0}"
CF_MODE="${CF_MODE:-strict_flip}"
FEATURE_COST="${FEATURE_COST:-cosine}"
NODE_MASS="${NODE_MASS:-uniform}"
SIZE_PENALTY_BETA="${SIZE_PENALTY_BETA:-0.0}"
WNODE_THRESHOLDS="${WNODE_THRESHOLDS:-auto_quantile}"
WNODE_QUANTILES="${WNODE_QUANTILES:-0.05,0.10,0.20,0.30,0.50,0.70,0.90}"
SKIP_REDUNDANCY="${SKIP_REDUNDANCY:-1}"
PARTIAL_EVERY="${PARTIAL_EVERY:-500}"
RESUME="${RESUME:-1}"
PRESELECTED_TOPK="${PRESELECTED_TOPK:-20}"
REQUIRE_PRESELECTED_TOPK="${REQUIRE_PRESELECTED_TOPK:-1}"
RUN_DISTANCE_SELF_TEST="${RUN_DISTANCE_SELF_TEST:-0}"
TARGET_LABEL="${TARGET_LABEL:-1}"
SMILES_COL="${SMILES_COL:-smiles}"
LABEL_COL="${LABEL_COL:-label}"
DEVICE="${DEVICE:-cuda}"

source ~/.bashrc
conda activate "${CLEAR_ENV}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$PWD"
mkdir -p logs "${OUTPUT_DIR}" "$(dirname "${WNODE_CACHE_DB}")" "${NODE_EMB_CACHE_DIR}"

echo "===== MOLCLR NODE WASSERSTEIN CCRCOV ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python_path=$(which python)"
echo "python_version=$(python --version 2>&1)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
for name in TEACHER_PATH DATASET_CSV MOLCLR_ROOT MOLCLR_CKPT OURS_SELECTED_PATH FULLGRAPH_CANDIDATES_PATH OUTPUT_DIR WNODE_CACHE_DB NODE_EMB_CACHE_DIR MAX_PARENTS MAX_CANDIDATES RUN_OURS RUN_FULLGRAPH CF_MODE FEATURE_COST NODE_MASS SIZE_PENALTY_BETA WNODE_THRESHOLDS WNODE_QUANTILES SKIP_REDUNDANCY PARTIAL_EVERY RESUME PRESELECTED_TOPK REQUIRE_PRESELECTED_TOPK RUN_DISTANCE_SELF_TEST; do
  echo "${name}=${!name}"
done

python - <<'PY'
import importlib
import torch

for package in ("torch_geometric", "rdkit", "ot"):
    importlib.import_module(package)
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
print("POT/ot=available")
PY

for required in "${TEACHER_PATH}" "${DATASET_CSV}" "${MOLCLR_ROOT}" "${MOLCLR_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[ERROR] required path is missing: ${required}" >&2
    exit 1
  fi
done

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset-csv "${DATASET_CSV}"
  --teacher-path "${TEACHER_PATH}"
  --molclr-root "${MOLCLR_ROOT}"
  --molclr-checkpoint "${MOLCLR_CKPT}"
  --label "${TARGET_LABEL}"
  --smiles-col "${SMILES_COL}"
  --label-col "${LABEL_COL}"
  --cf-mode "${CF_MODE}"
  --output-dir "${OUTPUT_DIR}"
  --max-parents "${MAX_PARENTS}"
  --max-candidates "${MAX_CANDIDATES}"
  --wnode-thresholds "${WNODE_THRESHOLDS}"
  --wnode-quantiles "${WNODE_QUANTILES}"
  --wnode-cache-db "${WNODE_CACHE_DB}"
  --node-emb-cache-dir "${NODE_EMB_CACHE_DIR}"
  --feature-cost "${FEATURE_COST}"
  --node-mass "${NODE_MASS}"
  --size-penalty-beta "${SIZE_PENALTY_BETA}"
  --device "${DEVICE}"
  --skip-redundancy "${SKIP_REDUNDANCY}"
  --partial-every "${PARTIAL_EVERY}"
  --resume "${RESUME}"
  --run-distance-self-test "${RUN_DISTANCE_SELF_TEST}"
  --run-ours "${RUN_OURS}"
  --run-fullgraph "${RUN_FULLGRAPH}"
  --selection-method "${SELECTION_METHOD}"
  --preselected-topk "${PRESELECTED_TOPK}"
  --require-preselected-topk "${REQUIRE_PRESELECTED_TOPK}"
)

if [[ "${RUN_OURS}" == "1" ]]; then
  args+=(--ours-selected-path "${OURS_SELECTED_PATH}")
fi
if [[ "${RUN_FULLGRAPH}" == "1" ]]; then
  if [[ -z "${FULLGRAPH_CANDIDATES_PATH}" || ! -f "${FULLGRAPH_CANDIDATES_PATH}" ]]; then
    echo "[ERROR] RUN_FULLGRAPH=1 requires an existing FULLGRAPH_CANDIDATES_PATH" >&2
    exit 1
  fi
  args+=(--fullgraph-candidates-path "${FULLGRAPH_CANDIDATES_PATH}")
  args+=(--fullgraph-method-name "${FULLGRAPH_METHOD_NAME}")
fi

python scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py "${args[@]}"
