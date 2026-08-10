#!/bin/bash
#SBATCH --job-name=comrecgc_bace_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
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
CHEMISTRY_DIR=${CHEMISTRY_DIR:-$BASE_ROOT/chemistry}
DATASET_CSV=${DATASET_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_residual_v1/comrecgc}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$CHEMISTRY_DIR/run_manifest.json" "$DATASET_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$THRESHOLDS_JSON"; do test -s "$path" || { echo "missing input: $path" >&2; exit 2; }; done
args=(python scripts/baselines/comrecgc/run_slot_unified_eval.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode full --dataset bace --chemistry-dir "$CHEMISTRY_DIR" --dataset-csv "$DATASET_CSV" --teacher-path "$TEACHER_PATH" --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" --thresholds-json "$THRESHOLDS_JSON" --output-dir "$OUTPUT_DIR" --expected-parent-count 116 --max-k 20 --device cuda)
echo "hostname=$(hostname)"; echo "python=$(which python)"; python --version; python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'; echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then echo '[COMRECGC_BACE_EVAL_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/final_artifact_audit.json"
echo '[COMRECGC_BACE_EVAL_SUCCESS]'
