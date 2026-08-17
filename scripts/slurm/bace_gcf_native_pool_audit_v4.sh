#!/bin/bash
#SBATCH --job-name=bace_gcf_native_audit_v4
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
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
mkdir -p logs

DATASET_DIR=${DATASET_DIR:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset}
SUMMARY_DIR=${SUMMARY_DIR:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/summary_retry2_valid_native_rank/native_summary}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_gcf_native_pool_v4}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$DATASET_DIR/run_manifest.json" "$SUMMARY_DIR/run_manifest.json" "$TEACHER_PATH"; do
  test -s "$path" || { echo "[BACE_GCF_NATIVE_V4_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
args=(python scripts/baselines/gcfexplainer/export_bace_fullgraph_candidates.py
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --dataset-dir "$DATASET_DIR" --summary-dir "$SUMMARY_DIR"
  --teacher-path "$TEACHER_PATH" --output-dir "$OUTPUT_DIR"
  --profile full --parent-limit 360 --target-k 20 --scan-limit 0
  --scan-all --require-connected --validate-only)
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_GCF_NATIVE_POOL_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "[BACE_GCF_NATIVE_V4_COLLISION] $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python - "$OUTPUT_DIR/run_manifest.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8")); a=p["candidate_attrition"]
assert p["validation_passed"] is True and p["test_loaded"] is False
assert a["scan_all"] is True and a["scan_exhausted"] is True
assert a["num_retained"] == 20 and a["native_order_preserved"] is True
assert a["rf_reranking_performed"] is False and a["wnode_reranking_performed"] is False
PY
echo '[BACE_GCF_NATIVE_POOL_V4_SUCCESS]'
