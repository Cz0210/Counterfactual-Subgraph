#!/bin/bash
#SBATCH --job-name=bace_gcf_vrrw
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
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
GNN_CHECKPOINT=${GNN_CHECKPOINT:-$BASE/gnn/model_best.pth}
NEUROSED_CHECKPOINT=${NEUROSED_CHECKPOINT:-$BASE/neurosed/best_model.pt}
NEUROSED_MANIFEST=${NEUROSED_MANIFEST:-$BASE/neurosed/projection_manifest.json}
VRRW_DIR=${VRRW_DIR:-$BASE/vrrw_m50000_alpha1}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}

echo "[BACE_GCFEXPLAINER_VRRW_CONFIG]"
echo "profile=full"
echo "parent_limit=360"
echo "M=50000"
echo "alpha=1.0"
echo "theta=0.05"
echo "seed=13"
echo "calibration_loaded=false"
echo "test_loaded=false"

python scripts/baselines/gcfexplainer/run_bace_vrrw.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "$DATASET_DIR" \
  --official-root "$OFFICIAL_ROOT" \
  --gnn-checkpoint "$GNN_CHECKPOINT" \
  --neurosed-checkpoint "$NEUROSED_CHECKPOINT" \
  --neurosed-manifest "$NEUROSED_MANIFEST" \
  --output-dir "$VRRW_DIR" \
  --profile full \
  --parent-limit 360 \
  --m 50000 \
  --alpha 1.0 \
  --theta 0.05 \
  --teleport 0.1 \
  --candidate-capacity 100000 \
  --sample-size 10000 \
  --seed 13 \
  --device1 cuda:0 \
  --device2 cuda:0 \
  --resume

test -s "$VRRW_DIR/counterfactuals.pt"
test -s "$VRRW_DIR/_RUN_COMPLETE.json"
echo "[BACE_GCFEXPLAINER_VRRW_SUCCESS]"
