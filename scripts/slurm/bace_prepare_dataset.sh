#!/bin/bash
#SBATCH --job-name=bace_prepare
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
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

RAW_CSV=${RAW_CSV:-data/raw/BACE/bace.csv}
OUTPUT_DIR=${OUTPUT_DIR:-data/processed/BACE}
RAW_SMILES_COL=${RAW_SMILES_COL:-smiles}
RAW_LABEL_COL=${RAW_LABEL_COL:-label}
SPLIT_SEED=${SPLIT_SEED:-13}

echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
echo "raw_csv=$RAW_CSV"
echo "output_dir=$OUTPUT_DIR"

python scripts/data/prepare_bace.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --raw-csv "$RAW_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --raw-smiles-col "$RAW_SMILES_COL" \
  --raw-label-col "$RAW_LABEL_COL" \
  --split-seed "$SPLIT_SEED"

test -s "$OUTPUT_DIR/bace_dataset_summary.json"
echo "[BACE_PREPARE_DATASET_SUCCESS]"
