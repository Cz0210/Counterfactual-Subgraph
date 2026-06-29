#!/bin/bash
#SBATCH -J greed_pairs_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
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
PAIRS_DIR=${PAIRS_DIR:-outputs/hpc/greed_hiv/pairs}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH:-outputs/hpc/selectors/gt_fullgraph_tanimoto_baseline_label1/beta_20p0_gamma_5p0}

echo "===== GREED HIV PAIRS FULL ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version

CMD=(python scripts/generate_hiv_greed_pairs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --graphs-jsonl "${GRAPHS_JSONL}" \
  --out-dir "${PAIRS_DIR}" \
  --num-train-pairs "${NUM_TRAIN_PAIRS:-100000}" \
  --num-val-pairs "${NUM_VAL_PAIRS:-10000}" \
  --num-test-pairs "${NUM_TEST_PAIRS:-10000}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}")
if [ -n "${MAX_PARENTS:-}" ]; then
  CMD+=(--max-parents "${MAX_PARENTS}")
fi
if [ -n "${MAX_CANDIDATES:-}" ]; then
  CMD+=(--max-candidates "${MAX_CANDIDATES}")
fi
"${CMD[@]}"

python scripts/label_hiv_greed_pairs.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pairs-dir "${PAIRS_DIR}" \
  --fullgraph-label-mode bounded_approx

echo "===== GREED HIV PAIRS FULL DONE ====="
ls -lh "${PAIRS_DIR}"/*pairs*.csv
