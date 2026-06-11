#!/bin/bash
#SBATCH -J legacy_camc_emb_l1
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

HIV_CSV=${HIV_CSV:-data/raw/AIDS/HIV.csv}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
ROOT_OUT=${ROOT_OUT:-outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1}
SCRIPT=scripts/eval/compare_hiv_recourse_baselines.py

RUN_NAMES=(
  old_morgan
  embedding_conservative
  embedding_lowred
)
SELECTED_DIRS=(
  outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20
  outputs/hpc/selectors/widegrid_ours_embedding_label1/beta_20p0_gamma_5p0
  outputs/hpc/selectors/widegrid_ours_embedding_label1/beta_10p0_gamma_8p0
)
OUT_DIRS=(
  outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/old_morgan_seed13
  outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/embedding_conservative_beta20_gamma5_seed13
  outputs/hpc/comparison/hiv_quick/legacy_camc_eval_label1/embedding_lowred_beta10_gamma8_seed13
)

echo "===== LEGACY CAMC EMBEDDING SELECTOR EVAL ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "PYTHONPATH=${PYTHONPATH}"
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
try:
    from rdkit import Chem
    print("rdkit available: true")
    print("rdkit Chem module:", Chem.__name__)
except Exception as exc:
    print("rdkit available: false")
    print("rdkit import error:", repr(exc))
PY
echo "SCRIPT=${SCRIPT}"
echo "HIV_CSV=${HIV_CSV}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "ROOT_OUT=${ROOT_OUT}"
echo "Runs: ${RUN_NAMES[*]}"
echo "=========================================================="

for path in "${SCRIPT}" "${HIV_CSV}" "${TEACHER_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing file: ${path}"
    exit 1
  fi
done

for selected_dir in "${SELECTED_DIRS[@]}"; do
  if [ ! -d "${selected_dir}" ]; then
    echo "[ERROR] missing selected dir: ${selected_dir}"
    exit 1
  fi
done

mkdir -p "${ROOT_OUT}"

run_one() {
  local run_name="$1"
  local selected_dir="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"

  echo "===== START LEGACY CAMC RUN ====="
  echo "RUN_NAME=${run_name}"
  echo "OURS_SELECTED_DIR=${selected_dir}"
  echo "OUT_DIR=${out_dir}"
  echo "git commit: $(git rev-parse HEAD || true)"
  echo "python path: $(which python)"
  echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
  echo "teacher path: ${TEACHER_PATH}"
  echo "HIV csv: ${HIV_CSV}"
  echo "================================="

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

  echo "===== FINISHED LEGACY CAMC RUN: ${run_name} ====="
  echo "===== generated artifacts for ${run_name} ====="
  for artifact in \
    comparison_table.csv \
    comparison_summary.json \
    camc_comparison_table.csv \
    camc_summary.json \
    diagnostic_counts.json \
    progress.log
  do
    if [ -e "${out_dir}/${artifact}" ]; then
      ls -lh "${out_dir}/${artifact}"
    else
      echo "[MISSING] ${out_dir}/${artifact}"
    fi
  done

  echo "===== ${run_name}: comparison_table.csv ====="
  cat "${out_dir}/comparison_table.csv"
  echo "===== ${run_name}: comparison_summary.json ====="
  cat "${out_dir}/comparison_summary.json"
  echo "===== ${run_name}: camc_comparison_table.csv ====="
  cat "${out_dir}/camc_comparison_table.csv"
  echo "===== ${run_name}: camc_summary.json ====="
  cat "${out_dir}/camc_summary.json"
  echo "===== ${run_name}: diagnostic_counts.json ====="
  cat "${out_dir}/diagnostic_counts.json"
  echo "===== ${run_name}: progress.log tail ====="
  tail -n 120 "${out_dir}/progress.log"
}

for index in "${!RUN_NAMES[@]}"; do
  run_one "${RUN_NAMES[$index]}" "${SELECTED_DIRS[$index]}" "${OUT_DIRS[$index]}"
done

echo "===== ALL LEGACY CAMC SELECTOR EVAL RUNS DONE ====="
echo "ROOT_OUT=${ROOT_OUT}"
