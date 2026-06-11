#!/bin/bash
#SBATCH -J sel_ours_emb_wg_l1
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

POOL=outputs/hpc/full_candidate_pools/stable300_label1_merged_base_temp07/candidate_pool_with_embeddings.jsonl
OUT_ROOT=outputs/hpc/selectors/widegrid_ours_embedding_label1
SELECTOR=scripts/select_class_counterfactual_subgraphs.py
CHECKER=scripts/check_candidate_pool_embeddings.py
BETA_COVERAGE_LIST=(5.0 10.0 15.0 20.0)
GAMMA_REDUNDANCY_LIST=(5.0 8.0 10.0 15.0 20.0)

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
echo "POOL=${POOL}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "BETA_COVERAGE_LIST=${BETA_COVERAGE_LIST[*]}"
echo "GAMMA_REDUNDANCY_LIST=${GAMMA_REDUNDANCY_LIST[*]}"
echo "Selector redundancy metric: embedding cosine"
echo "====================="

for path in "${POOL}" "${SELECTOR}" "${CHECKER}"; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    if [ "${path}" = "${POOL}" ]; then
      echo "[HINT] Run sbatch scripts/slurm/add_embeddings_stable300_label1_merged_base_temp07.sh first."
    fi
    exit 1
  fi
done

python "${CHECKER}" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --candidate-pool "${POOL}" \
  --embedding-field final_fragment_embedding \
  --max-rows 5

for BETA in "${BETA_COVERAGE_LIST[@]}"; do
  BETA_TAG=${BETA//./p}
  for GAMMA in "${GAMMA_REDUNDANCY_LIST[@]}"; do
    GAMMA_TAG=${GAMMA//./p}
    OUT_DIR="${OUT_ROOT}/beta_${BETA_TAG}_gamma_${GAMMA_TAG}"
    mkdir -p "${OUT_DIR}"
    echo "===== SELECT OURS beta=${BETA} gamma=${GAMMA} ====="

    python "${SELECTOR}" \
      --config configs/hpc.yaml \
      --set inference.fallback_to_heuristic=false \
      --pool-jsonl "${POOL}" \
      --out-dir "${OUT_DIR}" \
      --label 1 \
      --alpha-cf 0.8 \
      --beta-coverage "${BETA}" \
      --gamma-redundancy "${GAMMA}" \
      --eta-size 0.3 \
      --top-k 20 \
      --require-cf-flip \
      --require-final-substructure \
      --dedup-by-final-fragment \
      --sim-metric embedding \
      --embedding-field final_fragment_embedding \
      --embedding-missing-policy error \
      --top-candidates-per-fragment 3

    echo "===== SUMMARY OURS beta=${BETA} gamma=${GAMMA} ====="
    cat "${OUT_DIR}/selector_summary.json"
  done
done

echo "===== ALL OURS EMBEDDING WIDEGRID SELECTOR RUNS DONE ====="
