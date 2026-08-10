#!/bin/bash
#SBATCH --job-name=comrecgc_bace_chem
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
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
export PYTHONHASHSEED=0
mkdir -p logs
BASE_ROOT=${BASE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/baselines/comrecgc/bace/project_full_connected_v1}
DATASET_DIR=${DATASET_DIR:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset}
GENERATION_DIR=${GENERATION_DIR:-$BASE_ROOT/generation}
TRACE_DIR=${TRACE_DIR:-$GENERATION_DIR/trace}
COMMON_RECOURSE_DIR=${COMMON_RECOURSE_DIR:-$BASE_ROOT/common_recourse}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE_ROOT/chemistry}
PREREGISTRATION=${PREREGISTRATION:-$BASE_ROOT/preregistration/deterministic_chem_repair.json}
TRACE_EVIDENCE_PATH=${TRACE_EVIDENCE_PATH:-$TRACE_DIR/trace_summary.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$GENERATION_DIR/run_manifest.json" "$TRACE_DIR/candidate_action_lineage.json" "$TRACE_EVIDENCE_PATH" "$COMMON_RECOURSE_DIR/selected_common_recourses.json"; do test -s "$path" || { echo "missing input: $path" >&2; exit 2; }; done
args=(python scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --project-root "$PROJECT_ROOT" --dataset bace --dataset-dir "$DATASET_DIR" --generation-dir "$GENERATION_DIR" --trace-lineage-path "$TRACE_DIR/candidate_action_lineage.json" --trace-evidence-path "$TRACE_EVIDENCE_PATH" --common-recourse-dir "$COMMON_RECOURSE_DIR" --output-dir "$OUTPUT_DIR" --preregistration-path "$PREREGISTRATION" --parent-limit 360)
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_CHEM_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$PREREGISTRATION")"
"${args[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo '[COMRECGC_BACE_CHEM_SUCCESS]'
