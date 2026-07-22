#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=mut_prep_full
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD

RAW_ROOT=${RAW_ROOT:-data/raw/Mutagenicity}
RAW_CSV=${RAW_CSV:-${RAW_ROOT}/smiles/smiles_mutagenicity_raw.csv}
CURATED_CSV=${CURATED_CSV:-${RAW_ROOT}/smiles/smiles_mutagenicity_curated.csv}
REMOVED_CSV=${REMOVED_CSV:-${RAW_ROOT}/smiles/smiles_mutagenicity_removed.csv}
TU_DIR=${TU_DIR:-${RAW_ROOT}/tudataset/Mutagenicity}
MANIFEST=${MANIFEST:-${RAW_ROOT}/SHA256SUMS}
RUN_ROOT=${RUN_ROOT:-outputs/hpc/datasets/mutagenicity_v1_full}
AUDIT_DIR=${AUDIT_DIR:-${RUN_ROOT}/source_audit}
PROCESSED_DIR=${PROCESSED_DIR:-${RUN_ROOT}/processed}
CANONICAL_DIR=${CANONICAL_DIR:-data/processed/Mutagenicity/v1}
SEED=${SEED:-42}

mkdir -p logs "${RUN_ROOT}" "${AUDIT_DIR}" "${PROCESSED_DIR}"

echo "===== MUTAGENICITY PREPROCESS FULL ====="
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "python=$(which python)"
python --version
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c 'import rdkit; print("rdkit_version=" + rdkit.__version__)'
echo "RAW_ROOT=${RAW_ROOT}"
echo "RAW_CSV=${RAW_CSV}"
echo "CURATED_CSV=${CURATED_CSV}"
echo "REMOVED_CSV=${REMOVED_CSV}"
echo "TU_DIR=${TU_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "PROCESSED_DIR=${PROCESSED_DIR}"
echo "CANONICAL_DIR=${CANONICAL_DIR}"
echo "SEED=${SEED}"

python scripts/data/verify_mutagenicity_download.py \
  --config configs/hpc.yaml \
  --root "${RAW_ROOT}" \
  --manifest "${MANIFEST}" \
  --out-json "${RUN_ROOT}/download_verification.json"

python scripts/data/audit_mutagenicity_sources.py \
  --config configs/hpc.yaml \
  --raw-csv "${RAW_CSV}" \
  --curated-csv "${CURATED_CSV}" \
  --removed-csv "${REMOVED_CSV}" \
  --tu-dir "${TU_DIR}" \
  --output-dir "${AUDIT_DIR}"

python scripts/data/preprocess_mutagenicity.py \
  --config configs/hpc.yaml \
  --curated-csv "${CURATED_CSV}" \
  --raw-csv "${RAW_CSV}" \
  --output-dir "${PROCESSED_DIR}" \
  --seed "${SEED}"

python scripts/data/build_mutagenicity_splits.py \
  --config configs/hpc.yaml \
  --processed-dir "${PROCESSED_DIR}" \
  --output-dir "${PROCESSED_DIR}" \
  --seed "${SEED}"

python scripts/data/validate_mutagenicity_processed.py \
  --config configs/hpc.yaml \
  --processed-dir "${PROCESSED_DIR}" \
  --summary-json "${PROCESSED_DIR}/validation_summary.json" \
  --report-txt "${PROCESSED_DIR}/validation_report.txt"

mkdir -p "${CANONICAL_DIR}"
for filename in \
  mutagenicity_master.csv \
  mutagenicity_benchmark_clean.csv \
  mutagenicity_dropped.csv \
  mutagenicity_duplicates.csv \
  mutagenicity_conflicts.csv \
  mutagenicity_id_map.csv \
  preprocess_summary.json \
  preprocess_report.md \
  mutagenicity_split_manifest.csv \
  train.csv \
  val.csv \
  calibration.csv \
  test.csv \
  split_summary.json \
  split_report.md \
  validation_summary.json \
  validation_report.txt
do
  cp -f "${PROCESSED_DIR}/${filename}" "${CANONICAL_DIR}/${filename}"
done

echo "[MUTAGENICITY_PREPROCESS_FULL_OK]"

