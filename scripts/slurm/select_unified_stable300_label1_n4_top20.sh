#!/bin/bash
#SBATCH -J sel_uni_l1
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

POOL=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/full_candidate_pools/unified_stable300_label1_n4/candidate_pool.jsonl
OUT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/selectors/unified_stable300_label1_n4_top20_mmr

echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"

if [ ! -e "${POOL}" ]; then
  echo "[ERROR] missing pool: ${POOL}"
  exit 1
fi

python scripts/select_class_counterfactual_subgraphs.py \
  --config configs/hpc.yaml \
  --pool-jsonl "${POOL}" \
  --out-dir "${OUT_DIR}" \
  --label 1 \
  --top-k 20 \
  --alpha-cf 1.0 \
  --beta-coverage 1.0 \
  --gamma-redundancy 0.7 \
  --eta-size 0.3 \
  --min-cf-drop 0.2 \
  --require-cf-flip \
  --require-final-substructure \
  --dedup-by-final-fragment \
  --sim-metric morgan

cat "${OUT_DIR}/selector_report.txt"
