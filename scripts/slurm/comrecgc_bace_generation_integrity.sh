#!/bin/bash
#SBATCH --job-name=comrecgc_bace_integrity
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
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
BASE_ROOT=${BASE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/baselines/comrecgc/bace/project_full_connected_v1}
GENERATION_DIR=${GENERATION_DIR:-$BASE_ROOT/generation}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE_ROOT/generation_integrity_gate}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
test -s "$GENERATION_DIR/_RUN_COMPLETE.json" || { echo 'generation incomplete' >&2; exit 2; }
args=(python scripts/baselines/comrecgc/audit_generation_integrity.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --generation-dir "$GENERATION_DIR" --output-dir "$OUTPUT_DIR" --expected-steps 50000)
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_INTEGRITY_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo '[COMRECGC_BACE_INTEGRITY_SUCCESS]'
