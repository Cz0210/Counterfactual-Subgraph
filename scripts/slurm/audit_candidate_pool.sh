#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/logs/%x-%j.err
#SBATCH --job-name=audit_pool

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
RUN_NAME=decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500
RUN_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}
POOL_JSONL=${RUN_DIR}/candidate_pool.jsonl
OUT_JSON=${PROJECT_DIR}/outputs/hpc/analysis/${RUN_NAME}_audit.json
OUT_TXT=${PROJECT_DIR}/outputs/hpc/analysis/${RUN_NAME}_audit.txt
SIM_SAMPLE_SIZE=5000
TOPK_SHOW=10

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

echo "repo pwd: $(pwd)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "pool_jsonl=${POOL_JSONL}"
echo "out_json=${OUT_JSON}"
echo "out_txt=${OUT_TXT}"
echo "git_commit=$(git rev-parse HEAD)"

if [ ! -f "${POOL_JSONL}" ]; then
  echo "[ERROR] candidate pool not found: ${POOL_JSONL}"
  exit 1
fi

python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "${POOL_JSONL}" \
  --out_json "${OUT_JSON}" \
  --out_txt "${OUT_TXT}" \
  --group_by_label \
  --sim_sample_size "${SIM_SAMPLE_SIZE}" \
  --topk_show "${TOPK_SHOW}"
