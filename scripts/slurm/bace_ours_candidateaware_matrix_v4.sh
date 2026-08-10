#!/bin/bash
#SBATCH --job-name=bace_ours_final_matrix_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=112G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
POOL_ROOT=${POOL_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/merged}
CANDIDATE_POOL=${CANDIDATE_POOL:-$POOL_ROOT/candidate_pool.jsonl}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}; MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/final_calibration_matrix}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DIAGNOSTIC_THRESHOLDS=${DIAGNOSTIC_THRESHOLDS:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$CANDIDATE_POOL" "$POOL_ROOT/candidate_pool_audit.json" "$CALIBRATION_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$DIAGNOSTIC_THRESHOLDS"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
readarray -t EXPECTED < <(python - "$POOL_ROOT/candidate_pool_audit.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); print(p["candidate_count"]); print(p["parent_count"])
PY
)
args=(python scripts/precompute_wnode_action_matrix.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --dataset BACE --split calibration --parent-csv "$CALIBRATION_CSV" --candidate-pool "$CANDIDATE_POOL" --teacher-path "$TEACHER_PATH" --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" --wnode-cache-db "$WNODE_CACHE_DB" --output-dir "$OUTPUT_DIR" --expected-parent-count 60 --expected-pool-rows "${EXPECTED[0]}" --expected-source-parent-count "${EXPECTED[1]}" --candidate-universe-policy connected_feasible_v4 --min-source-atom-ratio 0.0 --max-source-atom-ratio 0.85 --require-candidate-lineage --cf-mode strict_flip --action-semantics-version connected_sanitized_residual_v1 --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1 --device cuda --resume)
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_OURS_FINAL_MATRIX_V4_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python scripts/audit_wnode_action_matrix.py --config configs/hpc.yaml --run-dir "$OUTPUT_DIR" --expected-parent-count 60 --require-strict-flip-pair
python scripts/summarize_bace_candidate_union.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --matrix-root "$OUTPUT_DIR" --thresholds-json "$DIAGNOSTIC_THRESHOLDS" --expected-parent-count 60
echo '[BACE_OURS_FINAL_MATRIX_V4_SUCCESS]'
