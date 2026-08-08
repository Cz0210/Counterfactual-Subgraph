#!/bin/bash
#SBATCH --job-name=bace_gcf_summary
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
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

BASE=${BASE:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v1}
DATASET_DIR=${DATASET_DIR:-$BASE/dataset}
GNN_CHECKPOINT=${GNN_CHECKPOINT:-$BASE/gnn/model_best.pth}
NEUROSED_CHECKPOINT=${NEUROSED_CHECKPOINT:-$BASE/neurosed/best_model.pt}
VRRW_DIR=${VRRW_DIR:-$BASE/vrrw_m50000_alpha1}
SUMMARY_DIR=${SUMMARY_DIR:-$BASE/native_summary}
EXPORT_DIR=${EXPORT_DIR:-$BASE/export}
TEACHER_PATH=${TEACHER_PATH:-$SHARED_PROJECT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$PROJECT_ROOT/baselines/gcfexplainer_official}

if [ ! -s "$SUMMARY_DIR/_RUN_COMPLETE.json" ]; then
  python scripts/baselines/gcfexplainer/run_bace_summary.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --dataset-dir "$DATASET_DIR" \
    --official-root "$OFFICIAL_ROOT" \
    --vrrw-dir "$VRRW_DIR" \
    --gnn-checkpoint "$GNN_CHECKPOINT" \
    --neurosed-checkpoint "$NEUROSED_CHECKPOINT" \
    --output-dir "$SUMMARY_DIR" \
    --profile full \
    --theta 0.1 \
    --minimum-native-export 100 \
    --device cuda:0
fi

if [ ! -s "$EXPORT_DIR/_RUN_COMPLETE.json" ]; then
  python scripts/baselines/gcfexplainer/export_bace_fullgraph_candidates.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --dataset-dir "$DATASET_DIR" \
    --summary-dir "$SUMMARY_DIR" \
    --teacher-path "$TEACHER_PATH" \
    --output-dir "$EXPORT_DIR" \
    --profile full \
    --parent-limit 360 \
    --top-k 20
fi

test -s "$SUMMARY_DIR/_RUN_COMPLETE.json"
test -s "$EXPORT_DIR/selected_top20.csv"
test -s "$EXPORT_DIR/_RUN_COMPLETE.json"
echo "[BACE_GCFEXPLAINER_SUMMARY_EXPORT_SUCCESS]"
