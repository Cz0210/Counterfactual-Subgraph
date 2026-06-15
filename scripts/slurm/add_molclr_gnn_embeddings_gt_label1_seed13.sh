#!/bin/bash
#SBATCH -J add_molclr_gt_l1
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

MOLCLR_ROOT=${MOLCLR_ROOT:-}
MOLCLR_CKPT=${MOLCLR_CKPT:-}
ENCODER_TYPE=${ENCODER_TYPE:-gin}
POOL_JSONL=${POOL_JSONL:-}
OUT_JSONL=${OUT_JSONL:-outputs/hpc/comparison/hiv_quick/label1_1594411/gt_fullgraph_candidate_pool_with_molclr_gnn_embeddings.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-outputs/hpc/comparison/hiv_quick/label1_1594411/molclr_gnn_embedding_summary.json}
EMBEDDING_FIELD=${EMBEDDING_FIELD:-final_fragment_gnn_embedding}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-auto}
INVALID_POLICY=${INVALID_POLICY:-error}
MAX_ROWS=${MAX_ROWS:-}

if [ -z "${POOL_JSONL}" ]; then
  if [ -f outputs/hpc/comparison/hiv_quick/label1_1594411/gt_fullgraph_candidate_pool.jsonl ]; then
    POOL_JSONL=outputs/hpc/comparison/hiv_quick/label1_1594411/gt_fullgraph_candidate_pool.jsonl
  elif [ -f outputs/hpc/comparison/hiv_quick/label1_1594411/camc_gt_fullgraph_candidate_pool.jsonl ]; then
    POOL_JSONL=outputs/hpc/comparison/hiv_quick/label1_1594411/camc_gt_fullgraph_candidate_pool.jsonl
  else
    echo "[ERROR] Could not find a GT fullgraph candidate pool for label1_1594411."
    echo "[HINT] Run sbatch scripts/slurm/convert_camc_gt_fullgraph_motif_pools_label1_clean.sh first,"
    echo "       or set POOL_JSONL=/path/to/gt_fullgraph_candidate_pool.jsonl."
    exit 1
  fi
fi

echo "===== MOLCLR GNN GT EMBEDDING ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import importlib.util
print("torch available:", importlib.util.find_spec("torch") is not None)
if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("torch_geometric available:", importlib.util.find_spec("torch_geometric") is not None)
PY
echo "MOLCLR_ROOT=${MOLCLR_ROOT}"
echo "MOLCLR_CKPT=${MOLCLR_CKPT}"
echo "ENCODER_TYPE=${ENCODER_TYPE}"
echo "POOL_JSONL=${POOL_JSONL}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "EMBEDDING_FIELD=${EMBEDDING_FIELD}"
echo "============================================="

if [ -z "${MOLCLR_ROOT}" ]; then
  echo "[ERROR] MOLCLR_ROOT is required."
  exit 1
fi
if [ -z "${MOLCLR_CKPT}" ]; then
  echo "[ERROR] MOLCLR_CKPT is required."
  exit 1
fi

for path in "${POOL_JSONL}" "${MOLCLR_ROOT}" "${MOLCLR_CKPT}" scripts/add_candidate_pool_molclr_embeddings.py; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

CMD=(
  python scripts/add_candidate_pool_molclr_embeddings.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --candidate-pool "${POOL_JSONL}"
  --out-jsonl "${OUT_JSONL}"
  --summary-json "${SUMMARY_JSON}"
  --molclr-root "${MOLCLR_ROOT}"
  --molclr-ckpt "${MOLCLR_CKPT}"
  --encoder-type "${ENCODER_TYPE}"
  --smiles-field final_fragment
  --embedding-field "${EMBEDDING_FIELD}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --invalid-policy "${INVALID_POLICY}"
)
if [ -n "${MAX_ROWS}" ]; then
  CMD+=(--max-rows "${MAX_ROWS}")
fi

"${CMD[@]}"

python scripts/check_candidate_pool_embeddings.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --candidate-pool "${OUT_JSONL}" \
  --embedding-field "${EMBEDDING_FIELD}" \
  --max-rows 5

echo "===== GT MOLCLR GNN EMBEDDING DONE ====="
ls -lh "${OUT_JSONL}" "${SUMMARY_JSON}"
cat "${SUMMARY_JSON}"

