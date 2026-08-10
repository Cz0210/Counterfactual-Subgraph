#!/bin/bash
#SBATCH --job-name=bace_ours_full_conn_matrix_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=72G
#SBATCH --time=3-00:00:00
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

CANDIDATE_POOL=${CANDIDATE_POOL:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_multiseed_v3/merged/candidate_pool.jsonl}
ATTRITION_AUDIT=${ATTRITION_AUDIT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_candidate_universe_v4/candidate_universe_attrition.json}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/calibration_matrix}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DIAGNOSTIC_THRESHOLDS=${DIAGNOSTIC_THRESHOLDS:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$CANDIDATE_POOL" "$ATTRITION_AUDIT" "$CALIBRATION_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$DIAGNOSTIC_THRESHOLDS"; do
  test -s "$path" || { echo "[BACE_FULL_CONNECTED_MATRIX_V4_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
case "$CANDIDATE_POOL $CALIBRATION_CSV $ATTRITION_AUDIT" in
  *test*|*gcf*|*GCF*) echo "[BACE_FULL_CONNECTED_MATRIX_V4_LEAKAGE] forbidden input" >&2; exit 2 ;;
esac
EXPECTED_UNIQUE=$(python - "$ATTRITION_AUDIT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["source_flip_hard_filter_removed"] is True
assert payload["source_cfdrop_hard_filter_removed"] is True
assert payload["test_loaded"] is False
print(int(payload["stage_counts"]["CANDIDATES_AFTER_UNIVERSE_FIX"]))
PY
)

args=(
  python scripts/precompute_wnode_action_matrix.py
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --dataset BACE --split calibration --parent-csv "$CALIBRATION_CSV"
  --candidate-pool "$CANDIDATE_POOL" --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --wnode-cache-db "$WNODE_CACHE_DB" --output-dir "$OUTPUT_DIR"
  --expected-parent-count 60 --expected-pool-rows 389
  --expected-source-parent-count 255 --expected-source-eligible-rows 349
  --expected-unique-candidates "$EXPECTED_UNIQUE"
  --candidate-universe-policy connected_feasible_v4
  --min-source-atom-ratio 0.0 --max-source-atom-ratio 0.85
  --require-candidate-lineage
  --cf-mode strict_flip
  --action-semantics-version connected_sanitized_residual_v1
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
  --device cuda --resume
)

echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
echo "expected_unique=$EXPECTED_UNIQUE"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_FULL_CONNECTED_MATRIX_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_FULL_CONNECTED_MATRIX_V4_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python scripts/audit_wnode_action_matrix.py \
  --config configs/hpc.yaml --run-dir "$OUTPUT_DIR" \
  --expected-parent-count 60 --expected-candidate-count "$EXPECTED_UNIQUE" \
  --require-strict-flip-pair
python - "$OUTPUT_DIR/matrix_audit.json" "$OUTPUT_DIR/run_manifest.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
assert audit["audit_passed"] is True
assert audit["action_semantics_version"] == "connected_sanitized_residual_v1"
assert audit["disconnected_residual_used_count"] == 0
assert audit["stale_cache_rows"] == 0
assert manifest["inputs"]["candidate_universe_policy"] == "connected_feasible_v4"
assert manifest["test_loaded"] is False
PY
python scripts/summarize_bace_candidate_union.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --matrix-root "$OUTPUT_DIR" --thresholds-json "$DIAGNOSTIC_THRESHOLDS" \
  --expected-parent-count 60
echo "[BACE_FULL_CONNECTED_MATRIX_V4_SUCCESS]"
