#!/bin/bash
#SBATCH -J emb_gt_camc_l1
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

EMBED_SCRIPT=scripts/add_candidate_pool_embeddings.py
CHECK_SCRIPT=scripts/check_candidate_pool_embeddings.py
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
echo "EMBED_SCRIPT=${EMBED_SCRIPT}"
echo "CHECK_SCRIPT=${CHECK_SCRIPT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "Using clean multi-seed GT pools: 1594411, 1594412, 1594413"
echo "Not using old run: 1593189"
echo "====================="

for path in "${EMBED_SCRIPT}" "${CHECK_SCRIPT}" "${MODEL_PATH}"; do
  if [ ! -e "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

RUN_IDS=(1594411 1594412 1594413)
for RUN_ID in "${RUN_IDS[@]}"; do
  RUN_DIR="outputs/hpc/comparison/hiv_quick/label1_${RUN_ID}"
  INPUT_JSONL="${RUN_DIR}/gt_fullgraph_candidate_pool.jsonl"
  OUTPUT_JSONL="${RUN_DIR}/gt_fullgraph_candidate_pool_with_embeddings.jsonl"

  echo "===== ADD EMBEDDINGS label1_${RUN_ID} ====="
  if [ ! -f "${INPUT_JSONL}" ]; then
    echo "[ERROR] missing converted candidate pool: ${INPUT_JSONL}"
    echo "[HINT] Run sbatch scripts/slurm/convert_camc_gt_fullgraph_motif_pools_label1_clean.sh first."
    exit 1
  fi

  python "${EMBED_SCRIPT}" \
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

  python "${CHECK_SCRIPT}" \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --candidate-pool "${OUTPUT_JSONL}" \
    --embedding-field final_fragment_embedding \
    --max-rows 5
done

echo "===== ALL GT CAMC EMBEDDINGS DONE ====="
