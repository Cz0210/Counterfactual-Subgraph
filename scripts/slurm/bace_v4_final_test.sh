#!/bin/bash
#SBATCH --job-name=bace_v4_final_test
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
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
COMMON_ROOT=${COMMON_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4}
PROTOCOL_GATE=${PROTOCOL_GATE:-$COMMON_ROOT/protocol_gate/threshold_protocol_gate.json}
SELECTOR_ROOT=${SELECTOR_ROOT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_connected_candidateaware_v4}
OURS_SELECTION=${OURS_SELECTION:-$SELECTOR_ROOT/frozen_selection.json}
GCF_CANDIDATES=${GCF_CANDIDATES:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/summary_retry2_valid_native_rank/export/selected_top20.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}; MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}; MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}; THRESHOLDS_JSON=${THRESHOLDS_JSON:-$COMMON_ROOT/threshold_protocol/thresholds.json}
WORK_ROOT=${WORK_ROOT:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_connected_candidateaware_v4_work}; WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$PROTOCOL_GATE" "$OURS_SELECTION" "$SELECTOR_ROOT/selected_top20.csv" "$GCF_CANDIDATES" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
python - "$PROTOCOL_GATE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"]=="PASS" and p["COMMON_PROTOCOL_GATE_PASS"] and p["test_used_for_selection"] is False and p["threshold_fitted_on_test"] is False
PY
test "$(awk 'END {print NR-1}' "$TEST_CSV")" -eq 116
ours=(python scripts/evaluate_bace_method.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --method ours --candidate-path "$SELECTOR_ROOT" --teacher-path "$TEACHER_PATH" --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" --test-csv "$TEST_CSV" --thresholds-json "$THRESHOLDS_JSON" --work-dir "$WORK_ROOT/ours" --output-dir "$COMMON_ROOT/ours" --expected-test-parents 116 --selection-manifest "$OURS_SELECTION" --test-evaluation-count 1 --action-semantics-version connected_sanitized_residual_v1 --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1 --wnode-cache-db "$WNODE_CACHE_DB")
gcf=(python scripts/evaluate_bace_method.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --method gcfexplainer --candidate-path "$GCF_CANDIDATES" --teacher-path "$TEACHER_PATH" --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" --test-csv "$TEST_CSV" --thresholds-json "$THRESHOLDS_JSON" --work-dir "$WORK_ROOT/gcfexplainer" --output-dir "$COMMON_ROOT/gcfexplainer" --expected-test-parents 116 --test-evaluation-count 1 --action-semantics-version connected_sanitized_residual_v1 --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1 --wnode-cache-db "$WNODE_CACHE_DB")
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'ours_command='; printf '%q ' "${ours[@]}"; printf '\n'; printf 'gcf_command='; printf '%q ' "${gcf[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_V4_FINAL_TEST_VALIDATE_OK]'; exit 0; fi
test ! -e "$COMMON_ROOT/ours" && test ! -e "$COMMON_ROOT/gcfexplainer" || { echo "final test output collision" >&2; exit 2; }
test ! -e "$WORK_ROOT/ours/test" && test ! -e "$WORK_ROOT/gcfexplainer/test" || { echo "final test work collision" >&2; exit 2; }
"${ours[@]}"
"${gcf[@]}"
python scripts/audit_bace_ours_wnode_prefix_v2.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --mode final --root "$COMMON_ROOT/ours" --selector-root "$SELECTOR_ROOT" --require-connected --output-json "$COMMON_ROOT/ours/connected_final_gate.json"
echo '[BACE_V4_FINAL_TEST_SUCCESS]'
