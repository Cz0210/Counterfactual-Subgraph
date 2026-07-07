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
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FGW_LAMBDA=${FGW_LAMBDA}"
echo "FGW_THRESHOLDS=${FGW_THRESHOLDS}"
echo "FGW_QUANTILES=${FGW_QUANTILES}"
echo "MAX_PARENTS=${MAX_PARENTS}"
echo "MAX_CANDIDATES=${MAX_CANDIDATES}"
echo "CF_MODE=${CF_MODE}"
echo "SKIP_REDUNDANCY=${SKIP_REDUNDANCY}"

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
if [ ! -d "${MOLCLR_ROOT}" ]; then
  echo "[ERROR] MOLCLR_ROOT not found: ${MOLCLR_ROOT}"
  exit 2
fi
if [ ! -f "${MOLCLR_CKPT}" ]; then
  echo "[ERROR] MOLCLR_CKPT not found: ${MOLCLR_CKPT}"
  exit 2
fi

python scripts/evaluate_ccrcov_with_molclr_node_fgw.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${HIV_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}" \
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

echo "===== MOLCLR NODE FGW CCRCov SMOKE DONE ====="
find "${OUTPUT_DIR}" -maxdepth 3 -type f | sort
