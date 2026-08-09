#!/bin/bash
#SBATCH --job-name=bace_ours_funnel
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs
SELECTED=${SELECTED:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_top20/selected_subgraphs.csv}
CAL_DETAILS=${CAL_DETAILS:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/ours/calibration/details/pair_details.csv}
TEST_DETAILS=${TEST_DETAILS:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/ours/test/details/pair_details.csv}
THRESHOLDS=${THRESHOLDS:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_ours_low_k10_coverage}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$SELECTED" "$CAL_DETAILS" "$TEST_DETAILS" "$THRESHOLDS"; do test -s "$path"; done
THETA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["theta_star"])' "$THRESHOLDS")
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo "[BACE_OURS_FUNNEL_VALIDATE_OK] theta=$THETA"; exit 0; fi
python scripts/audit_bace_ours_coverage_funnel.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --selected-csv "$SELECTED" --calibration-details "$CAL_DETAILS" \
  --test-details "$TEST_DETAILS" --theta-star "$THETA" --output-dir "$OUTPUT_DIR"
echo "[BACE_OURS_FUNNEL_SUCCESS]"
