#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${SOURCE_RUN_MANIFEST:?set SOURCE_RUN_MANIFEST}"
: "${SOURCE_TASK_STATE:?set SOURCE_TASK_STATE}"
: "${SOURCE_CHECKPOINT:?set SOURCE_CHECKPOINT}"
: "${SOURCE_SQLITE:?set SOURCE_SQLITE}"
: "${GLOBALGCE_OFFICIAL_ROOT:?set GLOBALGCE_OFFICIAL_ROOT}"
: "${BACE_NATIVE_TRAIN_CSV:?set BACE_NATIVE_TRAIN_CSV}"
: "${BACE_SOURCE_MANIFEST:?set BACE_SOURCE_MANIFEST}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

python scripts/baselines/globalgce/adopt_bace_v5_gspan_mining.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-run-manifest "$SOURCE_RUN_MANIFEST" \
  --source-task-state "$SOURCE_TASK_STATE" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-sqlite "$SOURCE_SQLITE" \
  --official-root "$GLOBALGCE_OFFICIAL_ROOT" \
  --native-train-csv "$BACE_NATIVE_TRAIN_CSV" \
  --source-manifest "$BACE_SOURCE_MANIFEST" \
  --gine-checkpoint "$BACE_GINE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-official-commit 157e65c2850bc787f229a1ee8c60564906b933f2 \
  --expected-pattern-count 5441858 \
  --expected-root-count 19 \
  --min-freq 7 \
  --top-k 20 \
  --seed 13
