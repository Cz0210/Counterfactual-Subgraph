#!/bin/bash
#SBATCH --job-name=bace_gcf_gnn
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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

DATASET_DIR=${DATASET_DIR:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v1/dataset}
GNN_DIR=${GNN_DIR:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v1/gnn}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}
RESUME=${RESUME:-true}

echo "[BACE_GCFEXPLAINER_GNN_CONFIG]"
echo "profile=full"
echo "epochs=1000"
echo "train_limit=869"
echo "val_limit=162"
echo "seed=13"
echo "calibration_loaded=false"
echo "test_loaded=false"

train_args=(
  scripts/baselines/gcfexplainer/train_bace_gnn.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --dataset-dir "$DATASET_DIR"
  --official-root "$OFFICIAL_ROOT"
  --output-dir "$GNN_DIR"
  --profile full
  --epochs 1000
  --train-limit 869
  --val-limit 162
  --batch-size 64
  --learning-rate 0.001
  --dropout 0.0
  --seed 13
  --device cuda:0
)
if [ "$RESUME" = "true" ]; then
  train_args+=(--resume)
elif [ "$RESUME" = "false" ]; then
  train_args+=(--no-resume)
else
  echo "[BACE_GCFEXPLAINER_CONFIG_ERROR] RESUME must be true or false: $RESUME" >&2
  exit 2
fi

python "${train_args[@]}"

test -s "$GNN_DIR/model_best.pth"
test -s "$GNN_DIR/_RUN_COMPLETE.json"
echo "[BACE_GCFEXPLAINER_GNN_SUCCESS]"
