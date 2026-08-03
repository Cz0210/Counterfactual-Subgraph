#!/bin/bash
# Resource directives copied from scripts/slurm/plot_fgw_sota_figures_gpu.sh.
# The verified template has no account, qos, or constraint directives.
#SBATCH --job-name=mut_gcf3_plot
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
CONDA_SH=/share/home/u20526/anaconda3/etc/profile.d/conda.sh

if [ ! -f "${CONDA_SH}" ]; then
  echo "[ERROR] Conda setup script not found: ${CONDA_SH}" >&2
  exit 2
fi
source "${CONDA_SH}"
conda activate smiles_pip118

cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
export MPLBACKEND=Agg
mkdir -p logs

INPUT_DIR=outputs/hpc/eval/paper/mut_three_method_gcf_style_inputs_v1
FIGURE_DIR=outputs/hpc/eval/paper/mut_three_method_gcf_style_figures_v1
Q20=0.0323756993249126
Q30=0.0385762445762996

echo "===== MUT THREE-METHOD GCF-STYLE PLOT ====="
echo "hostname=$(hostname)"
echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "pwd=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "partition=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python_path=$(command -v python)"
echo "MPLBACKEND=${MPLBACKEND}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "FIGURE_DIR=${FIGURE_DIR}"
echo "==========================================="

nvidia-smi || true

for directory in "${INPUT_DIR}" "${FIGURE_DIR}"; do
  if [ -d "${directory}" ] && [ -n "$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "[ERROR] Refusing to overwrite non-empty output directory: ${directory}" >&2
    exit 2
  fi
done

python -m py_compile \
  scripts/prepare_mut_three_method_gcf_style_inputs.py \
  scripts/plot_fgw_sota_figures.py

python scripts/prepare_mut_three_method_gcf_style_inputs.py \
  --ours-dir outputs/hpc/mutagenicity/final_eval/wnode_frozen_a2_test_p217_k20_v3 \
  --clear-dir outputs/hpc/mutagenicity/final_eval/clear_parent_frequency_top20_plot_artifacts_p217_v1 \
  --globalgce-dir outputs/hpc/mutagenicity/final/globalgce_wnode_frequency_top20_test_v1 \
  --output-dir "${INPUT_DIR}" \
  --theta-star "${Q30}" \
  --figure4-k 10 \
  --expected-num-parents 217

grep -Fq '[MUT_THREE_METHOD_NORMALIZATION_PASS]' \
  "${INPUT_DIR}/mut_three_method_normalization_audit.txt"

python scripts/plot_fgw_sota_figures.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --figure3-report-dir "${INPUT_DIR}/mut_three_method_figure3_coverage_cost_vs_k.csv" \
  --figure4-csv "${INPUT_DIR}/mut_three_method_figure4_coverage_vs_threshold.csv" \
  --output-dir "${FIGURE_DIR}" \
  --expected-methods Ours,CLEAR,GlobalGCE \
  --dataset-name Mutagenicity \
  --figure4-mode quantile \
  --q20 "${Q20}" \
  --q30 "${Q30}"

expected_outputs=(
  mut_figure3_main_k1_10_coverage_conditional_cost.png
  mut_figure3_main_k1_10_coverage_conditional_cost.pdf
  mut_figure3_supplement_k1_20_coverage_conditional_cost.png
  mut_figure3_supplement_k1_20_coverage_conditional_cost.pdf
  mut_figure4_quantile_ccrcov_k10.png
  mut_figure4_quantile_ccrcov_k10.pdf
  mut_table2_k10_q30_three_method.csv
  mut_table2_k10_q30_three_method.md
  mut_table2_k10_q30_three_method.png
  mut_table2_k10_q30_three_method.pdf
  mut_three_method_plot_audit.txt
)
for filename in "${expected_outputs[@]}"; do
  if [ ! -s "${FIGURE_DIR}/${filename}" ]; then
    echo "[ERROR] Expected non-empty output missing: ${FIGURE_DIR}/${filename}" >&2
    exit 3
  fi
done

if [ -e "${FIGURE_DIR}/figure4_low_cost_auc_0_q30.csv" ]; then
  echo "[ERROR] Quantile mode must not emit dense low-cost pAUC artifacts." >&2
  exit 3
fi

echo "[MUT_THREE_METHOD_GCF_STYLE_PLOT_SUCCESS]"
