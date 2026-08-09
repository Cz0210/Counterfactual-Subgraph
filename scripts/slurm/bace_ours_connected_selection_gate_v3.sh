#!/bin/bash
#SBATCH --job-name=bace_ours_conn_gate_v3
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

SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_residual_v3}
OUTPUT_JSON=${OUTPUT_JSON:-$SELECTOR_ROOT/connected_selection_gate.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

test -s "$SELECTOR_ROOT/selector_audit.json" || { echo "[BACE_CONNECTED_GATE_CONFIG_ERROR] missing selector" >&2; exit 2; }
args=(
  python scripts/audit_bace_ours_wnode_prefix_v2.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --mode selector
  --root "$SELECTOR_ROOT"
  --output-json "$OUTPUT_JSON"
  --require-connected
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_CONNECTED_SELECTION_GATE_V3_VALIDATE_OK]"
  exit 0
fi
"${args[@]}"
test -s "$SELECTOR_ROOT/frozen_selection.json"
echo "[BACE_CONNECTED_SELECTION_GATE_V3_SUCCESS]"
