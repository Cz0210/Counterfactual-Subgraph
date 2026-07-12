#!/bin/bash
#SBATCH -J node_fgw_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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

TEACHER_PATH=${TEACHER_PATH:-${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-${PROJECT_ROOT}/pretrained_models/MolCLR}
MOLCLR_CKPT=${MOLCLR_CKPT:-${PROJECT_ROOT}/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-${PROJECT_ROOT}/outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH:-${PROJECT_ROOT}/outputs/hpc/baselines/gt_fullgraph/label1_opposite_fullgraph_candidates_max2000_seed0.csv}
GCF_CANDIDATES_PATH=${GCF_CANDIDATES_PATH:-${GT_FULLGRAPH_CANDIDATES_PATH}}
HIV_CSV=${HIV_CSV:-${PROJECT_ROOT}/data/raw/AIDS/HIV.csv}
CLEAR_PARENT_CSV=${CLEAR_PARENT_CSV:-${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-HIV_active}
TARGET_LABEL=${TARGET_LABEL:-1}
CF_MODE=${CF_MODE:-strict_flip}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/hpc/eval/ccrcov_molclr_node_fgw_smoke}
FGW_LAMBDA=${FGW_LAMBDA:-0.5}
FGW_THRESHOLDS=${FGW_THRESHOLDS:-auto_quantile}
FGW_QUANTILES=${FGW_QUANTILES:-0.05,0.10,0.20,0.30,0.50,0.70,0.90}
FGW_CACHE_DB=${FGW_CACHE_DB:-${PROJECT_ROOT}/outputs/hpc/cache/distance_cache/molclr_node_fgw_v1.sqlite}
NODE_EMB_CACHE_DIR=${NODE_EMB_CACHE_DIR:-${PROJECT_ROOT}/outputs/hpc/cache/molclr_node_embeddings}
STRUCTURE_MODE=${STRUCTURE_MODE:-shortest_path_unweighted}
FEATURE_COST=${FEATURE_COST:-cosine}
ATOM_PENALTY=${ATOM_PENALTY:-0.0}
MAX_PARENTS=${MAX_PARENTS:-50}
MAX_CANDIDATES=${MAX_CANDIDATES:-20}
SKIP_REDUNDANCY=${SKIP_REDUNDANCY:-1}
RUN_OURS=${RUN_OURS:-1}
RUN_FULLGRAPH=${RUN_FULLGRAPH:-1}
RUN_GT_FULLGRAPH=${RUN_GT_FULLGRAPH:-1}
FULLGRAPH_METHOD=${FULLGRAPH_METHOD:-}
PRESELECTED_TOPK=${PRESELECTED_TOPK:-0}
REQUIRE_PRESELECTED_TOPK=${REQUIRE_PRESELECTED_TOPK:-0}

if [ "${FULLGRAPH_METHOD}" = "clear" ]; then
  RUN_GT_FULLGRAPH=0
  HIV_CSV=${CLEAR_PARENT_CSV}
  SMILES_COL=smiles
  LABEL_COL=label
  CLEAR_FULLGRAPH_CANDIDATES_PATH=${CLEAR_FULLGRAPH_CANDIDATES_PATH:-${PROJECT_ROOT}/outputs/hpc/baselines/clear/aids/selected/clear_aids_rf_strict_flip_top20_greedy_mmr.csv}
  FULLGRAPH_METHOD_NAME=${FULLGRAPH_METHOD_NAME:-CLEAR-RF-FullGraph}
  if [ "${REQUIRE_PRESELECTED_TOPK}" = "1" ]; then
    MAX_PARENTS=0
    MAX_CANDIDATES=${PRESELECTED_TOPK}
  fi
fi

echo "===== MOLCLR NODE FGW CCRCov SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "python version: $(python --version)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "MOLCLR_ROOT=${MOLCLR_ROOT}"
echo "MOLCLR_CKPT=${MOLCLR_CKPT}"
echo "OURS_SELECTED_PATH=${OURS_SELECTED_PATH}"
echo "GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH}"
echo "GCF_CANDIDATES_PATH=${GCF_CANDIDATES_PATH}"
echo "HIV_CSV=${HIV_CSV}"
echo "CLEAR_PARENT_CSV=${CLEAR_PARENT_CSV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FGW_LAMBDA=${FGW_LAMBDA}"
echo "FGW_THRESHOLDS=${FGW_THRESHOLDS}"
echo "FGW_QUANTILES=${FGW_QUANTILES}"
echo "MAX_PARENTS=${MAX_PARENTS}"
echo "MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "CF_MODE=${CF_MODE}"
echo "SKIP_REDUNDANCY=${SKIP_REDUNDANCY}"
echo "RUN_OURS=${RUN_OURS}"
echo "RUN_FULLGRAPH=${RUN_FULLGRAPH}"
echo "RUN_GT_FULLGRAPH=${RUN_GT_FULLGRAPH}"
echo "FULLGRAPH_METHOD=${FULLGRAPH_METHOD:-unset}"
echo "PRESELECTED_TOPK=${PRESELECTED_TOPK}"
echo "REQUIRE_PRESELECTED_TOPK=${REQUIRE_PRESELECTED_TOPK}"
echo "CLEAR_FULLGRAPH_CANDIDATES_PATH=${CLEAR_FULLGRAPH_CANDIDATES_PATH:-unset}"

python - <<'PY'
import importlib.util
print("torch available:", importlib.util.find_spec("torch") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("cuda device 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
print("torch_geometric available:", importlib.util.find_spec("torch_geometric") is not None)
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("POT/ot available:", importlib.util.find_spec("ot") is not None)
if importlib.util.find_spec("ot") is None:
    raise SystemExit("[ERROR] POT is required for MolCLR Node-FGW. Install with: pip install POT  # or conda install -c conda-forge pot")
PY

if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] TEACHER_PATH not found: ${TEACHER_PATH}"
  exit 2
fi
if [ ! -f "${HIV_CSV}" ]; then
  echo "[ERROR] Dataset CSV not found: ${HIV_CSV}"
  exit 2
fi
if [ ! -d "${MOLCLR_ROOT}" ]; then
  echo "[ERROR] MOLCLR_ROOT not found: ${MOLCLR_ROOT}"
  exit 2
fi
if [ ! -f "${MOLCLR_CKPT}" ]; then
  echo "[ERROR] MOLCLR_CKPT not found: ${MOLCLR_CKPT}"
  exit 2
fi
if [ "${REQUIRE_PRESELECTED_TOPK}" = "1" ] && [ "${PRESELECTED_TOPK}" -le 0 ]; then
  echo "[ERROR] REQUIRE_PRESELECTED_TOPK=1 requires PRESELECTED_TOPK > 0."
  exit 2
fi
if [ "${REQUIRE_PRESELECTED_TOPK}" = "1" ] && [ "${CF_MODE}" != "strict_flip" ]; then
  echo "[ERROR] Preselected final evaluation requires CF_MODE=strict_flip."
  exit 2
fi
if [ "${FULLGRAPH_METHOD}" = "clear" ] && [ ! -f "${CLEAR_FULLGRAPH_CANDIDATES_PATH}" ]; then
  echo "[ERROR] Preselected CLEAR candidate CSV not found: ${CLEAR_FULLGRAPH_CANDIDATES_PATH}"
  exit 2
fi

args=(
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${HIV_CSV}" \
  --teacher-path "${TEACHER_PATH}" \
  --molclr-root "${MOLCLR_ROOT}" \
  --molclr-checkpoint "${MOLCLR_CKPT}" \
  --label "${TARGET_LABEL}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --cf-mode "${CF_MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-parents "${MAX_PARENTS}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --fgw-lambda "${FGW_LAMBDA}" \
  --fgw-thresholds "${FGW_THRESHOLDS}" \
  --fgw-quantiles "${FGW_QUANTILES}" \
  --fgw-cache-db "${FGW_CACHE_DB}" \
  --node-emb-cache-dir "${NODE_EMB_CACHE_DIR}" \
  --structure-mode "${STRUCTURE_MODE}" \
  --feature-cost "${FEATURE_COST}" \
  --atom-penalty "${ATOM_PENALTY}" \
  --skip-redundancy \
  --partial-every 500
  --run-ours "${RUN_OURS}"
  --run-fullgraph "${RUN_FULLGRAPH}"
  --preselected-topk "${PRESELECTED_TOPK}"
  --require-preselected-topk "${REQUIRE_PRESELECTED_TOPK}"
)
if [ "${RUN_OURS}" = "1" ]; then
  args+=(--ours-selected-path "${OURS_SELECTED_PATH}")
fi
if [ "${RUN_FULLGRAPH}" = "1" ] && [ "${RUN_GT_FULLGRAPH}" = "1" ]; then
  args+=(--gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}")
fi
if [ -n "${CLEAR_FULLGRAPH_CANDIDATES_PATH:-}" ]; then
  args+=(--clear-fullgraph-candidates-path "${CLEAR_FULLGRAPH_CANDIDATES_PATH}")
fi
if [ -n "${FULLGRAPH_METHOD_NAME:-}" ]; then
  args+=(--fullgraph-method-name "${FULLGRAPH_METHOD_NAME}")
fi

python scripts/evaluate_ccrcov_with_molclr_node_fgw.py "${args[@]}"

echo "===== MOLCLR NODE FGW CCRCov SMOKE DONE ====="
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort
