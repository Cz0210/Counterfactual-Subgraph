#!/bin/bash
#SBATCH --job-name=bace_gcf_native_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
SHARED_PROJECT_ROOT=${SHARED_PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

BASE=${BASE:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2}
DATASET_DIR=${DATASET_DIR:-$BASE/dataset}
VRRW_DIR=${VRRW_DIR:-$BASE/vrrw_m50000_alpha1}
TEACHER_ROOT=${TEACHER_ROOT:-$SHARED_PROJECT_ROOT/outputs/hpc/oracle/bace}
TEACHER_PATH=${TEACHER_PATH:-$TEACHER_ROOT/bace_teacher.pkl}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must name a fresh audit directory}

echo "hostname=$(hostname)"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/baselines/gcfexplainer/audit_bace_native_candidates.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "$DATASET_DIR" \
  --vrrw-dir "$VRRW_DIR" \
  --teacher-path "$TEACHER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --profile full \
  --parent-limit 360 \
  --target-k 20 \
  --scan-limit 0 \
  --calibration-source-csv "$TEACHER_ROOT/teacher_consistent/calibration_source_label1_teacher_correct.csv" \
  --test-source-csv "$TEACHER_ROOT/teacher_consistent/test_source_label1_teacher_correct.csv"

test -s "$OUTPUT_DIR/candidate_attrition_audit.json"
test -s "$OUTPUT_DIR/source_roundtrip_audit.json"
test -s "$OUTPUT_DIR/_AUDIT_COMPLETE.json"
echo "[BACE_GCFEXPLAINER_NATIVE_CANDIDATE_AUDIT_SUCCESS]"
