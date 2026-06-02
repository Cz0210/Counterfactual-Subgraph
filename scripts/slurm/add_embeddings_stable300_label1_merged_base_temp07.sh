#!/bin/bash
#SBATCH -J add_l1_merge_emb
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

INPUT_JSONL=outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool.jsonl
OUTPUT_JSONL=outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_embeddings.jsonl
MODEL_PATH=outputs/hpc/rl_checkpoints/decoded_chem_ppo_stable300_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500

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
  --embedding-source final_fragment \
  --embedding-field final_fragment_embedding \
  --pooling mean \
  --batch-size 32 \
  --max-length 128 \
  --device auto

echo "===== EMBEDDING GENERATION DONE ====="
ls -lh outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07
echo "===== EMBEDDING SUMMARY ====="
cat outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/embedding_summary.json
