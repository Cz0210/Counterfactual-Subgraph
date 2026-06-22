#!/bin/bash
#SBATCH -J sel_gt_tanimoto_l1
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

GT_MOTIF_POOL=${GT_MOTIF_POOL:-outputs/hpc/comparison/hiv_quick/label1_1594411/camc_gt_fullgraph_motif_pool.csv}
OUT_DIR=${OUT_DIR:-outputs/hpc/selectors/gt_fullgraph_tanimoto_baseline_label1/beta_20p0_gamma_5p0}
TOP_K=${TOP_K:-20}
ALPHA_CF=${ALPHA_CF:-0.8}
BETA_COVERAGE=${BETA_COVERAGE:-20.0}
GAMMA_REDUNDANCY=${GAMMA_REDUNDANCY:-5.0}
ETA_SIZE=${ETA_SIZE:-0.3}
TARGET_LABEL=${TARGET_LABEL:-1}
SEED=${SEED:-13}

echo "===== GT FULLGRAPH TANIMOTO BASELINE SELECTOR ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
try:
    from rdkit import Chem
    print("rdkit available: true")
except Exception as exc:
    print("rdkit available: false", repr(exc))
PY
echo "GT_MOTIF_POOL=${GT_MOTIF_POOL}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOP_K=${TOP_K}"
echo "ALPHA_CF=${ALPHA_CF}"
echo "BETA_COVERAGE=${BETA_COVERAGE}"
echo "GAMMA_REDUNDANCY=${GAMMA_REDUNDANCY}"
echo "ETA_SIZE=${ETA_SIZE}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "SEED=${SEED}"
echo "=============================================================="

for path in "${GT_MOTIF_POOL}" scripts/select_gt_fullgraph_tanimoto_baseline_label1.py; do
  if [ ! -f "${path}" ]; then
    echo "[ERROR] missing file: ${path}"
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"
python scripts/select_gt_fullgraph_tanimoto_baseline_label1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --gt-motif-pool "${GT_MOTIF_POOL}" \
  --out-dir "${OUT_DIR}" \
  --top-k "${TOP_K}" \
  --alpha-cf "${ALPHA_CF}" \
  --beta-coverage "${BETA_COVERAGE}" \
  --gamma-redundancy "${GAMMA_REDUNDANCY}" \
  --eta-size "${ETA_SIZE}" \
  --target-label "${TARGET_LABEL}" \
  --seed "${SEED}"

echo "===== GT FULLGRAPH TANIMOTO BASELINE SELECTOR SUMMARY ====="
cat "${OUT_DIR}/selector_summary.json"
