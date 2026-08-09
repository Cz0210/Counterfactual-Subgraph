#!/bin/bash
#SBATCH --job-name=bace_conn_common_audit_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
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

COMMON_ROOT=${COMMON_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3}
OURS_ROOT=${OURS_ROOT:-$COMMON_ROOT/ours}
GCF_ROOT=${GCF_ROOT:-$COMMON_ROOT/gcfexplainer}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$COMMON_ROOT/thresholds.json}
GCF_CANDIDATE_AUDIT=${GCF_CANDIDATE_AUDIT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_ours_disconnected_residual_v2/gcf_candidate_connectivity_audit.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$OURS_ROOT/final_artifact_audit.json" "$GCF_ROOT/final_artifact_audit.json" "$THRESHOLDS_JSON" "$GCF_CANDIDATE_AUDIT"; do
  test -s "$path" || { echo "[BACE_CONNECTED_COMMON_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
args=(
  python scripts/audit_bace_connected_common_artifacts.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --ours-root "$OURS_ROOT"
  --gcf-root "$GCF_ROOT"
  --thresholds-json "$THRESHOLDS_JSON"
  --gcf-candidate-audit "$GCF_CANDIDATE_AUDIT"
  --output-root "$COMMON_ROOT"
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CONNECTED_COMMON_AUDIT_V3_VALIDATE_OK]"
  exit 0
fi
"${args[@]}"
echo "[BACE_CONNECTED_COMMON_AUDIT_V3_SUCCESS]"
