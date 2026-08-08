#!/bin/bash

set -euo pipefail

: "${BBBP_PLAN:?BBBP_PLAN is required}"
: "${BBBP_STAGE:?BBBP_STAGE is required}"

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

VALIDATE_ONLY=${VALIDATE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

echo "[BBBP_STAGE_CONFIG]"
echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "pwd=$(pwd)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "plan=$BBBP_PLAN"
echo "stage=$BBBP_STAGE"
echo "cf_mode=strict_flip"
echo "distance_line=MolCLR-Node-Wasserstein"
echo "threshold_source=calibration"
echo "selection_performed_in_eval=false"
echo "threshold_fitted_on_test=false"

args=(
  scripts/baselines/bbbp/run_stage.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --project-root "$PROJECT_ROOT"
  --plan "$BBBP_PLAN"
  --stage "$BBBP_STAGE"
)
if [ "$VALIDATE_ONLY" = "1" ]; then
  args+=(--validate-only)
elif [ "$DRY_RUN" = "1" ]; then
  args+=(--dry-run)
else
  args+=(--execute)
fi

python "${args[@]}"
