#!/bin/bash
#SBATCH -J cmp_uni_overlap
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

LABEL0_JSON=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/selectors/unified_stable300_label0_n4_top20_mmr/selected_subgraphs.json
LABEL1_JSON=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/selectors/unified_stable300_label1_n4_top20_mmr/selected_subgraphs.json
OUT_DIR=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/overlap/unified_stable300_label0_vs_label1_top20

echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "LABEL0_JSON=${LABEL0_JSON}"
echo "LABEL1_JSON=${LABEL1_JSON}"
echo "OUT_DIR=${OUT_DIR}"

for p in "${LABEL0_JSON}" "${LABEL1_JSON}"; do
  if [ ! -e "${p}" ]; then
    echo "[ERROR] missing path: ${p}"
    exit 1
  fi
done

python scripts/compare_selected_subgraph_overlap.py \
  --config configs/hpc.yaml \
  --label0-selected-json "${LABEL0_JSON}" \
  --label1-selected-json "${LABEL1_JSON}" \
  --out-dir "${OUT_DIR}" \
  --sim-thresholds 0.5 0.7 0.85 0.95

cat "${OUT_DIR}/overlap_report.txt"
