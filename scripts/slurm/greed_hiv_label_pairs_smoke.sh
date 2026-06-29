#!/bin/bash
#SBATCH -J greed_pairs_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

GRAPHS_JSONL=${GRAPHS_JSONL:-outputs/hpc/greed_hiv/dataset/graphs.jsonl}
PAIRS_DIR=${PAIRS_DIR:-outputs/hpc/greed_hiv/pairs_smoke}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
GT_FULLGRAPH_DEFAULT="${PROJECT_ROOT}/outputs/hpc/baselines/gt_fullgraph/label1_opposite_fullgraph_candidates_max2000_seed0.csv"
if [ -z "${GT_FULLGRAPH_CANDIDATES_PATH:-}" ] && [ -n "${GCF_CANDIDATES_PATH:-}" ]; then
  GT_FULLGRAPH_CANDIDATES_PATH="${GCF_CANDIDATES_PATH}"
fi
GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH:-${GT_FULLGRAPH_DEFAULT}}

echo "===== GREED HIV PAIRS SMOKE ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
echo "GRAPHS_JSONL=${GRAPHS_JSONL}"
echo "PAIRS_DIR=${PAIRS_DIR}"
echo "OURS_SELECTED_PATH=${OURS_SELECTED_PATH}"
echo "[GT_FULLGRAPH_CONFIG]"
echo "GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH}"

python scripts/generate_hiv_greed_pairs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --graphs-jsonl "${GRAPHS_JSONL}" \
  --out-dir "${PAIRS_DIR}" \
  --num-train-pairs 5000 \
  --num-val-pairs 1000 \
  --num-test-pairs 1000 \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}" \
  --max-parents "${MAX_PARENTS:-200}" \
  --max-candidates "${MAX_CANDIDATES:-50}"

python scripts/label_hiv_greed_pairs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pairs-dir "${PAIRS_DIR}" \
  --fullgraph-label-mode bounded_approx

echo "===== GREED HIV PAIRS SMOKE DONE ====="
ls -lh "${PAIRS_DIR}"/*pairs*.csv
