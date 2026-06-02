#!/bin/bash
#SBATCH -J check_pool_emb
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

CANDIDATE_POOL=${CANDIDATE_POOL:-/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_embeddings.jsonl}
EMBEDDING_FIELD=${EMBEDDING_FIELD:-final_fragment_embedding}
MAX_ROWS=${MAX_ROWS:-20}

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
echo "CANDIDATE_POOL=${CANDIDATE_POOL}"
echo "EMBEDDING_FIELD=${EMBEDDING_FIELD}"
echo "MAX_ROWS=${MAX_ROWS}"
echo "====================="

if [ ! -f "${CANDIDATE_POOL}" ]; then
  echo "[ERROR] candidate pool not found: ${CANDIDATE_POOL}"
  exit 1
fi

python scripts/check_candidate_pool_embeddings.py \
  --config configs/hpc.yaml \
  --candidate-pool "${CANDIDATE_POOL}" \
  --embedding-field "${EMBEDDING_FIELD}" \
  --max-rows "${MAX_ROWS}"
