#!/bin/bash
#SBATCH --job-name=bace_gcf_prepare
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

TEACHER_ROOT=${TEACHER_ROOT:-$SHARED_PROJECT_ROOT/outputs/hpc/oracle/bace}
DATASET_DIR=${DATASET_DIR:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v1/dataset}
NEUROSED_DIR=${NEUROSED_DIR:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v1/neurosed}
SOURCE_NEUROSED=${SOURCE_NEUROSED:-$SHARED_PROJECT_ROOT/outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}

if [ -s "$DATASET_DIR/_RUN_COMPLETE.json" ] && \
   [ -s "$NEUROSED_DIR/best_model.pt" ] && \
   [ -s "$NEUROSED_DIR/projection_manifest.json" ]; then
  echo "[BACE_GCFEXPLAINER_PREPARE_ADOPT_EXISTING]"
  exit 0
fi
if { [ -d "$DATASET_DIR" ] && [ -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; } || \
   { [ -d "$NEUROSED_DIR" ] && [ -n "$(find "$NEUROSED_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; }; then
  echo "[BACE_GCFEXPLAINER_CONFIG_ERROR] non-empty incomplete prepare output" >&2
  exit 2
fi

echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_dir=$DATASET_DIR"
echo "neurosed_dir=$NEUROSED_DIR"
echo "calibration_loaded=false"
echo "test_loaded=false"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/baselines/gcfexplainer/prepare_bace_dataset.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --train-source-csv "$TEACHER_ROOT/teacher_consistent/train_source_label1_teacher_correct.csv" \
  --train-target-csv "$TEACHER_ROOT/teacher_consistent/train_target_label0_teacher_correct.csv" \
  --val-source-csv "$TEACHER_ROOT/teacher_consistent/val_source_label1_teacher_correct.csv" \
  --val-target-csv "$TEACHER_ROOT/teacher_consistent/val_target_label0_teacher_correct.csv" \
  --output-dir "$DATASET_DIR"

mkdir -p "$NEUROSED_DIR"
python scripts/baselines/gcfexplainer/adapt_bace_neurosed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-checkpoint "$SOURCE_NEUROSED" \
  --output-checkpoint "$NEUROSED_DIR/best_model.pt" \
  --manifest-path "$NEUROSED_DIR/projection_manifest.json"

test -s "$DATASET_DIR/_RUN_COMPLETE.json"
test -s "$NEUROSED_DIR/best_model.pt"
test -s "$NEUROSED_DIR/projection_manifest.json"
echo "[BACE_GCFEXPLAINER_PREPARE_SUCCESS]"
