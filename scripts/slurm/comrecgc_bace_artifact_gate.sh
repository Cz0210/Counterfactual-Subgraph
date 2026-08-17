#!/bin/bash
#SBATCH --job-name=comrecgc_bace_gate
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
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
ROOT=${ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_residual_v1/comrecgc}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$ROOT/final_artifact_audit.json" "$THRESHOLDS_JSON"; do test -s "$path" || { echo "missing input: $path" >&2; exit 2; }; done
args=(python scripts/baselines/comrecgc/audit_bace_artifacts.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --root "$ROOT" --thresholds-json "$THRESHOLDS_JSON" --expected-parent-count 116)
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_GATE_VALIDATE_OK]'; exit 0; fi
"${args[@]}"
test -s "$ROOT/bace_comrecgc_artifact_gate.json"
echo '[COMRECGC_BACE_GATE_SUCCESS]'
