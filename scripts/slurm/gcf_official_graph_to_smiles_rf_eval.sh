#!/bin/bash
#SBATCH -J gcf_off_smiles
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
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

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/gcfexplainer_official}
GCF_ALPHA=${GCF_ALPHA:-0.5}
GCF_TRAIN_THETA=${GCF_TRAIN_THETA:-0.05}
GCF_MAX_STEPS=${GCF_MAX_STEPS:-50000}
RUN_DIR=${RUN_DIR:-${OUTPUT_ROOT}/full/alpha_${GCF_ALPHA}_theta_${GCF_TRAIN_THETA}_steps_${GCF_MAX_STEPS}}
SELECTED_GRAPHS_PATH=${SELECTED_GRAPHS_PATH:-${RUN_DIR}/summary_export/selected_counterfactual_graphs.pt}
SMILES_OUT_DIR=${SMILES_OUT_DIR:-${OUTPUT_ROOT}/graph_to_smiles}
GCF_SMILES_CSV=${GCF_SMILES_CSV:-${SMILES_OUT_DIR}/gcf_graph_smiles_candidates.csv}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}

echo "===== GCF OFFICIAL GRAPH TO SMILES + RF DIAGNOSTIC ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
echo "SELECTED_GRAPHS_PATH=${SELECTED_GRAPHS_PATH}"
echo "GCF_SMILES_CSV=${GCF_SMILES_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "CF_MODE=strict_flip"

if [ ! -f "${SELECTED_GRAPHS_PATH}" ]; then
  echo "[ERROR] selected graphs not found: ${SELECTED_GRAPHS_PATH}"
  echo "Run scripts/slurm/gcf_official_aids_summary_export_eval.sh first or set SELECTED_GRAPHS_PATH."
  exit 2
fi

python scripts/convert_gcf_official_graphs_to_smiles.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --selected-graphs-path "${SELECTED_GRAPHS_PATH}" \
  --out-csv "${GCF_SMILES_CSV}" \
  --out-report "${SMILES_OUT_DIR}/gcf_graph_to_smiles_report.json"

python scripts/evaluate_gcf_official_with_project_oracle.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --candidates-csv "${GCF_SMILES_CSV}" \
  --teacher-path "${TEACHER_PATH}" \
  --out-dir "${SMILES_OUT_DIR}/rf_oracle_eval" \
  --target-label "${TARGET_LABEL:-1}"

echo "===== GCF OFFICIAL GRAPH TO SMILES + RF DIAGNOSTIC DONE ====="
cat "${SMILES_OUT_DIR}/gcf_graph_to_smiles_report.json"
cat "${SMILES_OUT_DIR}/rf_oracle_eval/rf_oracle_eval_summary.json"

