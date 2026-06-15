#!/bin/bash
#SBATCH -J eval_molclr_camc_l1
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

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

mkdir -p logs

HIV_CSV=${HIV_CSV:-/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
SCRIPT=scripts/eval/compare_hiv_recourse_baselines.py
OURS_SELECTED_DIR=${OURS_SELECTED_DIR:-outputs/hpc/selectors/molclr_gnn_ours_embedding_label1/beta_20p0_gamma_5p0}
GT_SELECTED_DIR=${GT_SELECTED_DIR:-outputs/hpc/selectors/molclr_gnn_gt_fullgraph_embedding_label1_relaxed/label1_1594411/beta_20p0_gamma_5p0}
OURS_OUT_DIR=${OURS_OUT_DIR:-outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/molclr_gnn_ours_beta20_gamma5_seed13}
GT_OUT_DIR=${GT_OUT_DIR:-outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/molclr_gnn_gt_beta20_gamma5_seed13}

echo "===== LEGACY CAMC MOLCLR GNN SELECTOR ENV CHECK ====="
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
try:
    from rdkit import Chem
    print("rdkit available: true")
except Exception as exc:
    print("rdkit available: false", repr(exc))
PY
echo "HIV_CSV=${HIV_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OURS_SELECTED_DIR=${OURS_SELECTED_DIR}"
echo "GT_SELECTED_DIR=${GT_SELECTED_DIR}"
echo "OURS_OUT_DIR=${OURS_OUT_DIR}"
echo "GT_OUT_DIR=${GT_OUT_DIR}"
echo "====================================================="

for path in "${SCRIPT}" "${HIV_CSV}" "${TEACHER_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing file: ${path}"
    exit 1
  fi
done
for path in "${OURS_SELECTED_DIR}" "${GT_SELECTED_DIR}"; do
  if [ ! -d "${path}" ]; then
    echo "[ERROR] missing selector directory: ${path}"
    exit 1
  fi
done

run_legacy_camc() {
  local name="$1"
  local selected_dir="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  echo "===== START LEGACY CAMC: ${name} ====="
  echo "selected_dir=${selected_dir}"
  echo "out_dir=${out_dir}"

  python "${SCRIPT}" \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --hiv-csv "${HIV_CSV}" \
    --teacher-path "${TEACHER_PATH}" \
    --target-label 1 \
    --ours-selected-dir "${selected_dir}" \
    --top-k-list 10 20 \
    --theta-list 0.05 0.10 0.15 0.20 \
    --max-gt-candidates 2000 \
    --out-dir "${out_dir}" \
    --seed 13 \
    --progress-every 100 \
    --camc-delta-list 0.1 0.2 0.3 0.5 \
    --camc-top-k-list 10 20 \
    --camc-min-motif-atoms 2 \
    --enable-camc \
    --smiles-col smiles \
    --label-col HIV_active

  echo "===== FINISHED LEGACY CAMC: ${name} ====="
  cat "${out_dir}/camc_comparison_table.csv"
  cat "${out_dir}/comparison_table.csv"
}

run_legacy_camc "molclr_gnn_ours_beta20_gamma5_seed13" "${OURS_SELECTED_DIR}" "${OURS_OUT_DIR}"
run_legacy_camc "molclr_gnn_gt_beta20_gamma5_seed13" "${GT_SELECTED_DIR}" "${GT_OUT_DIR}"

echo "===== ALL MOLCLR GNN LEGACY CAMC RUNS DONE ====="
