#!/bin/bash
#SBATCH -J sel_gt_molclr_l1
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

mkdir -p logs

ALPHA_CF=${ALPHA_CF:-0.8}
BETA_COVERAGE=${BETA_COVERAGE:-20.0}
GAMMA_REDUNDANCY=${GAMMA_REDUNDANCY:-5.0}
ETA_SIZE=${ETA_SIZE:-0.3}
TOP_K=${TOP_K:-20}
MIN_CF_DROP=${MIN_CF_DROP:-0.2}
POOL_JSONL=${POOL_JSONL:-outputs/hpc/comparison/hiv_quick/label1_1594411/gt_fullgraph_candidate_pool_with_molclr_gnn_embeddings.jsonl}
OUT_DIR=${OUT_DIR:-outputs/hpc/selectors/molclr_gnn_gt_fullgraph_embedding_label1_relaxed/label1_1594411/beta_20p0_gamma_5p0}
EMBEDDING_FIELD=${EMBEDDING_FIELD:-final_fragment_gnn_embedding}

echo "===== MOLCLR GNN GT SELECTOR ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "ALPHA_CF=${ALPHA_CF}"
echo "BETA_COVERAGE=${BETA_COVERAGE}"
echo "GAMMA_REDUNDANCY=${GAMMA_REDUNDANCY}"
echo "ETA_SIZE=${ETA_SIZE}"
echo "TOP_K=${TOP_K}"
echo "MIN_CF_DROP=${MIN_CF_DROP}"
echo "POOL_JSONL=${POOL_JSONL}"
echo "OUT_DIR=${OUT_DIR}"
echo "EMBEDDING_FIELD=${EMBEDDING_FIELD}"
echo "GT selector uses --require-cf-flip, --require-final-substructure, --dedup-by-final-fragment"
echo "============================================"

for path in "${POOL_JSONL}" scripts/select_class_counterfactual_subgraphs.py scripts/check_candidate_pool_embeddings.py; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing file: ${path}"
    if [ "${path}" = "${POOL_JSONL}" ]; then
      echo "[HINT] Run sbatch scripts/slurm/add_molclr_gnn_embeddings_gt_label1_seed13.sh first."
    fi
    exit 1
  fi
done

python scripts/check_candidate_pool_embeddings.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --candidate-pool "${POOL_JSONL}" \
  --embedding-field "${EMBEDDING_FIELD}" \
  --max-rows 5

mkdir -p "${OUT_DIR}"
python scripts/select_class_counterfactual_subgraphs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pool-jsonl "${POOL_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --label 1 \
  --alpha-cf "${ALPHA_CF}" \
  --beta-coverage "${BETA_COVERAGE}" \
  --gamma-redundancy "${GAMMA_REDUNDANCY}" \
  --eta-size "${ETA_SIZE}" \
  --top-k "${TOP_K}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --require-cf-flip \
  --require-final-substructure \
  --dedup-by-final-fragment \
  --sim-metric embedding \
  --embedding-field "${EMBEDDING_FIELD}" \
  --embedding-missing-policy error \
  --top-candidates-per-fragment 3

echo "===== GT MOLCLR GNN SELECTOR SUMMARY ====="
cat "${OUT_DIR}/selector_summary.json"
