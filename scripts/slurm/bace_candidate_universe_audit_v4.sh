#!/bin/bash
#SBATCH --job-name=bace_candidate_universe_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

CANDIDATE_POOL=${CANDIDATE_POOL:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_multiseed_v3/merged/candidate_pool.jsonl}
MATRIX_ROOT=${MATRIX_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_residual_v3_expanded/calibration_matrix}
SELECTOR_INPUT=${SELECTOR_INPUT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3_expanded/selected_top20.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_candidate_universe_v4}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$CANDIDATE_POOL" "$MATRIX_ROOT/candidate_universe.jsonl" "$TEACHER_PATH"; do
  test -s "$path" || { echo "[BACE_CANDIDATE_UNIVERSE_V4_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
case "$CANDIDATE_POOL $MATRIX_ROOT $SELECTOR_INPUT" in
  *test*|*gcf*|*GCF*) echo "[BACE_CANDIDATE_UNIVERSE_V4_LEAKAGE] forbidden input" >&2; exit 2 ;;
esac

args=(
  python scripts/audit_bace_candidate_universe_attrition.py
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --candidate-pool "$CANDIDATE_POOL"
  --matrix-root "$MATRIX_ROOT"
  --selector-input "$SELECTOR_INPUT"
  --teacher-path "$TEACHER_PATH"
  --output-dir "$OUTPUT_DIR"
  --expected-pool-unique 151
  --expected-old-matrix-candidates 55
)

echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CANDIDATE_UNIVERSE_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_CANDIDATE_UNIVERSE_V4_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
echo "[BACE_CANDIDATE_UNIVERSE_V4_SUCCESS]"
