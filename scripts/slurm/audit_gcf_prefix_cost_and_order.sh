#!/bin/bash
#SBATCH --job-name=gcf_prefix_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate "${CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

GCF_RUN=${GCF_RUN:-outputs/hpc/gcfexplainer_hiv_csv/full/alpha_0.5_theta_0.05_steps_50000}
FGW_RUN=${FGW_RUN:-outputs/hpc/eval/ccrcov_molclr_node_fgw_full_gcfexplainer_hivcsv_a05_top20_lam05}
PAIR_DETAILS=${PAIR_DETAILS:-${FGW_RUN}/details/pair_details.csv}
CANDIDATE_CSV=${CANDIDATE_CSV:-${GCF_RUN}/smiles_conversion/gcf_hiv_csv_alpha0.5_selected_smiles_top20_for_fgw.csv}
SELECTED_METADATA=${SELECTED_METADATA:-${GCF_RUN}/summary_export/selected_counterfactual_metadata.csv}
SELECTED_GRAPHS=${SELECTED_GRAPHS:-${GCF_RUN}/summary_export/selected_counterfactual_graphs.pt}
CONVERTED_SMILES=${CONVERTED_SMILES:-${GCF_RUN}/smiles_conversion/gcf_hiv_csv_alpha0.5_selected_smiles.csv}
FIGURE3_CSV=${FIGURE3_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/audits/gcf_prefix_cost_and_order}
THETA=${THETA:-0.0328}
MAX_K=${MAX_K:-20}
METHOD=${METHOD:-GCFExplainer}

echo "===== GCF PREFIX COST AND ORDER AUDIT ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python_path=$(which python)"
echo "python_version=$(python --version)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PAIR_DETAILS=${PAIR_DETAILS}"
echo "CANDIDATE_CSV=${CANDIDATE_CSV}"
echo "SELECTED_METADATA=${SELECTED_METADATA}"
echo "SELECTED_GRAPHS=${SELECTED_GRAPHS}"
echo "CONVERTED_SMILES=${CONVERTED_SMILES}"
echo "FIGURE3_CSV=${FIGURE3_CSV:-not_provided}"
echo "THETA=${THETA}"
echo "MAX_K=${MAX_K}"
echo "METHOD=${METHOD}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "distance_recomputed=false"

for required in "${PAIR_DETAILS}" "${CANDIDATE_CSV}" "${SELECTED_METADATA}" "${SELECTED_GRAPHS}" "${CONVERTED_SMILES}"; do
  if [ ! -f "${required}" ]; then
    echo "[GCF_PREFIX_AUDIT_MISSING_INPUT] ${required}"
    exit 2
  fi
done

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --pair-details "${PAIR_DETAILS}"
  --candidate-csv "${CANDIDATE_CSV}"
  --selected-metadata "${SELECTED_METADATA}"
  --selected-graphs "${SELECTED_GRAPHS}"
  --converted-smiles "${CONVERTED_SMILES}"
  --theta "${THETA}"
  --max-k "${MAX_K}"
  --method "${METHOD}"
  --output-dir "${OUTPUT_DIR}"
)
if [ -n "${FIGURE3_CSV}" ]; then
  args+=(--figure3-csv "${FIGURE3_CSV}")
fi

python scripts/audit_gcf_prefix_cost_and_order.py "${args[@]}"

echo "[GCF_PREFIX_AUDIT_OUTPUTS]"
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort
