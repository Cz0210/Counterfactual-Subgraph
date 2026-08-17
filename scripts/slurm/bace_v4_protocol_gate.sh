#!/bin/bash
#SBATCH --job-name=bace_protocol_gate_v4
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
OURS_SELECTION=${OURS_SELECTION:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_candidateaware_v4/frozen_selection.json}
GCF_AUDIT_ROOT=${GCF_AUDIT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_gcf_native_pool_v4}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}; MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$ARTIFACT_ROOT/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}; TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/protocol_gate}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$OURS_SELECTION" "$GCF_AUDIT_ROOT/run_manifest.json" "$THRESHOLDS_JSON" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$CALIBRATION_CSV" "$TEST_CSV"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
args=(python scripts/audit_bace_v4_protocol_gate.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --ours-selection "$OURS_SELECTION" --gcf-audit-root "$GCF_AUDIT_ROOT" --thresholds-json "$THRESHOLDS_JSON" --teacher-path "$TEACHER_PATH" --molclr-checkpoint "$MOLCLR_CHECKPOINT" --calibration-csv "$CALIBRATION_CSV" --test-csv "$TEST_CSV" --output-dir "$OUTPUT_DIR" --git-commit "$(git rev-parse HEAD)")
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_V4_PROTOCOL_GATE_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
echo '[BACE_V4_PROTOCOL_GATE_SUCCESS]'
