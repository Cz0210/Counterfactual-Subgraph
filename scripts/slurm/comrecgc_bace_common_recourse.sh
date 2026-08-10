#!/bin/bash
#SBATCH --job-name=comrecgc_bace_recourse
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
DISTANCE_CHECKPOINT=${DISTANCE_CHECKPOINT:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE_ROOT/common_recourse}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$GENERATION_DIR/_RUN_COMPLETE.json" "$DISTANCE_CHECKPOINT"; do test -s "$path" || { echo "missing input: $path" >&2; exit 2; }; done
args=(python scripts/baselines/comrecgc/run_common_recourse.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --dataset bace --mode full --upstream-root external/COMRECGC --dataset-dir "$DATASET_DIR" --generation-dir "$GENERATION_DIR" --distance-checkpoint "$DISTANCE_CHECKPOINT" --output-dir "$OUTPUT_DIR" --parent-limit 360 --device cuda:0)
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_RECOURSE_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo '[COMRECGC_BACE_RECOURSE_SUCCESS]'
