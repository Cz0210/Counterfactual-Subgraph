#!/bin/bash
#SBATCH --job-name=bace_gcf_conn_reval_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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
mkdir -p logs

CANDIDATE_PATH=${CANDIDATE_PATH:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/summary_retry2_valid_native_rank/export/selected_top20.csv}
GCF_CANDIDATE_AUDIT=${GCF_CANDIDATE_AUDIT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_ours_disconnected_residual_v2/gcf_candidate_connectivity_audit.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3/thresholds.json}
WORK_DIR=${WORK_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_gcf_connected_residual_v3_work}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3/gcfexplainer}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$CANDIDATE_PATH" "$GCF_CANDIDATE_AUDIT" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$TEST_CSV" "$THRESHOLDS_JSON"; do
  test -s "$path" || { echo "[BACE_GCF_CONNECTED_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
python - "$GCF_CANDIDATE_AUDIT" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["passed"] is True
assert p["all_candidates_connected"] is True
assert p["native_rank_preserved"] is True
assert p["gcf_candidates_reused"] is True
PY
args=(
  python scripts/evaluate_bace_method.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method gcfexplainer
  --candidate-path "$CANDIDATE_PATH"
  --teacher-path "$TEACHER_PATH"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --test-csv "$TEST_CSV"
  --thresholds-json "$THRESHOLDS_JSON"
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --expected-test-parents 116
  --test-evaluation-count 1
  --action-semantics-version connected_sanitized_residual_v1
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
  --wnode-cache-db "$WNODE_CACHE_DB"
)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_GCF_CONNECTED_REVAL_V3_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_GCF_CONNECTED_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
test ! -e "$WORK_DIR/test" || { echo "[BACE_GCF_CONNECTED_WORK_COLLISION] $WORK_DIR/test" >&2; exit 2; }
"${args[@]}"
echo "[BACE_GCF_CONNECTED_REVAL_V3_SUCCESS]"
