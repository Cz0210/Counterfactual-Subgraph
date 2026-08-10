#!/bin/bash
#SBATCH --job-name=bace_common4_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
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
ROOT=${ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_residual_v1}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for method in ours globalgce gcfexplainer comrecgc; do test -s "$ROOT/$method/final_artifact_audit.json" || { echo "missing method: $method" >&2; exit 2; }; done
test -s "$ROOT/v4_import_manifest.json" || { echo 'missing v4 import manifest' >&2; exit 2; }
args=(python scripts/audit_bace_common4_connected.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --root "$ROOT")
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[BACE_COMMON4_AUDIT_VALIDATE_OK]'; exit 0; fi
"${args[@]}"
test -s "$ROOT/common_protocol_audit.json"
echo '[BACE_COMMON4_AUDIT_SUCCESS]'
