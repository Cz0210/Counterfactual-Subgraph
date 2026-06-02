#!/bin/bash
#SBATCH -J add_pool_emb
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

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

mkdir -p logs

INPUT_JSONL=${INPUT_JSONL:-outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool.jsonl}
OUTPUT_JSONL=${OUTPUT_JSONL:-outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_embeddings.jsonl}
MODEL_PATH=${MODEL_PATH:-outputs/hpc/rl_checkpoints/decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500}
EMBEDDING_SOURCE=${EMBEDDING_SOURCE:-final_fragment}
EMBEDDING_FIELD=${EMBEDDING_FIELD:-final_fragment_embedding}
POOLING=${POOLING:-mean}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-128}
DEVICE=${DEVICE:-auto}

echo "===== ENV CHECK ====="
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
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "EMBEDDING_SOURCE=${EMBEDDING_SOURCE}"
echo "EMBEDDING_FIELD=${EMBEDDING_FIELD}"
echo "POOLING=${POOLING}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "DEVICE=${DEVICE}"
echo "====================="

for path in "${INPUT_JSONL}" "${MODEL_PATH}" scripts/add_candidate_pool_embeddings.py; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

python scripts/add_candidate_pool_embeddings.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --embedding-source "${EMBEDDING_SOURCE}" \
  --embedding-field "${EMBEDDING_FIELD}" \
  --pooling "${POOLING}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --device "${DEVICE}"

echo "===== EMBEDDING GENERATION DONE ====="
ls -lh "$(dirname "${OUTPUT_JSONL}")"
echo "===== EMBEDDING SUMMARY ====="
cat "$(dirname "${OUTPUT_JSONL}")/embedding_summary.json"
