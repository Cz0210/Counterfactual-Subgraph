#!/bin/bash
#SBATCH --job-name=bace_teacher
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

DATA_DIR=${DATA_DIR:-data/processed/BACE}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/oracle/bace}

echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "data_dir=$DATA_DIR"
echo "output_dir=$OUTPUT_DIR"

python scripts/train_bace_teacher.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --random-seed 13 \
  --n-jobs 7

test -s "$OUTPUT_DIR/bace_teacher.pkl"
test -s "$OUTPUT_DIR/teacher_summary.json"
echo "[BACE_TRAIN_TEACHER_SUCCESS]"
