#!/bin/bash
#SBATCH -J sel_gt_emb_l1_rx
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

SELECTOR=scripts/select_class_counterfactual_subgraphs.py
CHECKER=scripts/check_candidate_pool_embeddings.py
DIAGNOSE=scripts/diagnose_candidate_pool_for_selector.py
OUT_ROOT=outputs/hpc/selectors/param_sweep_gt_fullgraph_embedding_cov20_relaxed
GAMMAS=(0.7 1.0 1.5 2.0 3.0 5.0)
RUN_IDS=(1594411 1594412 1594413)

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
echo "SELECTOR=${SELECTOR}"
echo "CHECKER=${CHECKER}"
echo "DIAGNOSE=${DIAGNOSE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "GAMMAS=${GAMMAS[*]}"
echo "Using clean multi-seed GT pools: 1594411, 1594412, 1594413"
echo "Not using old run: 1593189"
echo "Relaxed GT selector: no --require-cf-flip, --min-cf-drop -999, keep --require-final-substructure"
echo "====================="

for path in "${SELECTOR}" "${CHECKER}" "${DIAGNOSE}"; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing path: ${path}"
    exit 1
  fi
done

for RUN_ID in "${RUN_IDS[@]}"; do
  GT_POOL="outputs/hpc/comparison/hiv_quick/label1_${RUN_ID}/gt_fullgraph_candidate_pool_with_embeddings.jsonl"
  if [ ! -f "${GT_POOL}" ]; then
    echo "[ERROR] missing GT embedding pool: ${GT_POOL}"
    echo "[HINT] Run conversion and embedding jobs first:"
    echo "  sbatch scripts/slurm/convert_camc_gt_fullgraph_motif_pools_label1_clean.sh"
    echo "  sbatch scripts/slurm/add_embeddings_camc_gt_fullgraph_label1_clean.sh"
    exit 1
  fi

  python "${CHECKER}" \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --candidate-pool "${GT_POOL}" \
    --embedding-field final_fragment_embedding \
    --max-rows 5

  DIAG_DIR="outputs/hpc/diagnostics/camc_gt_fullgraph_label1/label1_${RUN_ID}"
  mkdir -p "${DIAG_DIR}"
  python "${DIAGNOSE}" \
    --pool-jsonl "${GT_POOL}" \
    --label 1 \
    --embedding-field final_fragment_embedding \
    --out-json "${DIAG_DIR}/candidate_pool_with_embeddings_diagnosis.json" \
    --out-txt "${DIAG_DIR}/candidate_pool_with_embeddings_diagnosis.txt"

  for GAMMA in "${GAMMAS[@]}"; do
    GAMMA_TAG=${GAMMA//./p}
    OUT_DIR="${OUT_ROOT}/label1_${RUN_ID}/gamma_${GAMMA_TAG}"
    mkdir -p "${OUT_DIR}"
    echo "===== SELECT RELAXED GT label1_${RUN_ID} gamma=${GAMMA} ====="

    python "${SELECTOR}" \
      --config configs/hpc.yaml \
      --set inference.fallback_to_heuristic=false \
      --pool-jsonl "${GT_POOL}" \
      --out-dir "${OUT_DIR}" \
      --label 1 \
      --alpha-cf 0.8 \
      --beta-coverage 20.0 \
      --gamma-redundancy "${GAMMA}" \
      --eta-size 0.3 \
      --top-k 20 \
      --min-cf-drop -999 \
      --require-final-substructure \
      --dedup-by-final-fragment \
      --sim-metric embedding \
      --embedding-field final_fragment_embedding \
      --embedding-missing-policy error

    echo "===== SUMMARY RELAXED GT label1_${RUN_ID} gamma=${GAMMA} ====="
    cat "${OUT_DIR}/selector_summary.json"
  done
done

echo "===== ALL RELAXED GT EMBEDDING SELECTOR SWEEPS DONE ====="
