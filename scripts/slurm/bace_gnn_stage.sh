#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

ACTION="${ACTION:?Set ACTION=calibrate or ACTION=oracle-smoke}"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

case "$ACTION" in
  calibrate)
    VALIDATION_CSV="${VALIDATION_CSV:?Set VALIDATION_CSV to the frozen validation CSV}"
    SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the B3 bundle}"
    OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:?Set OUTPUT_CHECKPOINT to a fresh B4 path}"
    python scripts/autodl/bace_gnn_stage.py \
      --config configs/hpc.yaml \
      calibrate \
      --source-checkpoint "$SOURCE_CHECKPOINT" \
      --output-checkpoint "$OUTPUT_CHECKPOINT" \
      --validation-csv "$VALIDATION_CSV"
    ;;
  oracle-smoke)
    CALIBRATION_CSV="${CALIBRATION_CSV:?Set CALIBRATION_CSV to the frozen calibration CSV}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the B4 bundle}"
    OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh B5 path}"
    python scripts/autodl/bace_gnn_stage.py \
      --config configs/hpc.yaml \
      oracle-smoke \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --calibration-csv "$CALIBRATION_CSV" \
      --output-dir "$OUTPUT_DIR" \
      --device cuda:0
    ;;
  *)
    echo "Unsupported ACTION: $ACTION" >&2
    exit 2
    ;;
esac
