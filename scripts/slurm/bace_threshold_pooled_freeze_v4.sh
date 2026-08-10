#!/bin/bash
#SBATCH --job-name=bace_threshold_pool_v4
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
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
OURS_MATRIX_ROOT=${OURS_MATRIX_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/final_calibration_matrix}
GCF_CALIBRATION_ROOT=${GCF_CALIBRATION_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_gcf_candidateaware_v4/calibration_run}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$OURS_MATRIX_ROOT/pair_matrix.jsonl" "$OURS_MATRIX_ROOT/matrix_audit.json" "$GCF_CALIBRATION_ROOT/details/pair_details.csv" "$GCF_CALIBRATION_ROOT/run_config.json" "$CALIBRATION_CSV"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
args=(python scripts/freeze_bace_pooled_connected_thresholds.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --ours-matrix-root "$OURS_MATRIX_ROOT" --gcf-calibration-root "$GCF_CALIBRATION_ROOT" --calibration-csv "$CALIBRATION_CSV" --output-dir "$OUTPUT_DIR")
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_THRESHOLD_POOLED_V4_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python - "$OUTPUT_DIR/threshold_protocol_audit.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"]=="PASS" and p["THRESHOLD_METHOD_INDEPENDENT"] and p["THRESHOLD_TEST_INDEPENDENT"] and p["COMMON_PROTOCOL_GATE_READY"]
PY
echo '[BACE_THRESHOLD_POOLED_V4_SUCCESS]'
