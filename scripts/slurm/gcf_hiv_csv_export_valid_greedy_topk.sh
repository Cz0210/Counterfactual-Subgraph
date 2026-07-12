#!/bin/bash
#SBATCH -J gcf_hiv_valid_topk
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

GCF_ALPHA=${GCF_ALPHA:-0.5}
GCF_TRAIN_THETA=${GCF_TRAIN_THETA:-0.05}
GCF_MAX_STEPS=${GCF_MAX_STEPS:-50000}
TOP_K=${TOP_K:-20}
SEED=${SEED:-0}
RUN_DIR=${RUN_DIR:-outputs/hpc/gcfexplainer_hiv_csv/full/alpha_${GCF_ALPHA}_theta_${GCF_TRAIN_THETA}_steps_${GCF_MAX_STEPS}}
COUNTERFACTUALS=${COUNTERFACTUALS:-${RUN_DIR}/counterfactuals.pt}
CANDIDATE_RECORDS=${CANDIDATE_RECORDS:-${RUN_DIR}/candidate_metadata.csv}
OUT_DIR=${OUT_DIR:-${RUN_DIR}/valid_greedy_export}
VALIDITY_CSV=${VALIDITY_CSV:-${OUT_DIR}/all_candidates_graph_to_smiles.csv}
VALIDITY_REPORT=${VALIDITY_REPORT:-${OUT_DIR}/all_candidates_graph_to_smiles_report.json}
CF_MODE=${CF_MODE:-strict_flip}

echo "===== GCF HIV CSV VALID GREEDY TOP-K EXPORT ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "python version: $(python --version)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CF_MODE=${CF_MODE}"
echo "COUNTERFACTUALS=${COUNTERFACTUALS}"
echo "CANDIDATE_RECORDS=${CANDIDATE_RECORDS}"
echo "VALIDITY_CSV=${VALIDITY_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "GCF_TRAIN_THETA=${GCF_TRAIN_THETA}"

if [[ ! -f "${COUNTERFACTUALS}" ]]; then
  echo "[ERROR] Missing counterfactual payload: ${COUNTERFACTUALS}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

python scripts/convert_gcf_hiv_csv_graphs_to_smiles.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --selected-graphs "${COUNTERFACTUALS}" \
  --out-csv "${VALIDITY_CSV}" \
  --out-report "${VALIDITY_REPORT}"

python scripts/gcf_hiv_csv_export_valid_greedy_topk.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --counterfactuals "${COUNTERFACTUALS}" \
  --candidate-records "${CANDIDATE_RECORDS}" \
  --validity-csv "${VALIDITY_CSV}" \
  --top-k "${TOP_K}" \
  --train-theta "${GCF_TRAIN_THETA}" \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}"

echo "===== GCF HIV CSV VALID GREEDY TOP-K EXPORT DONE ====="
cat "${OUT_DIR}/valid_greedy_selection_report.json"
