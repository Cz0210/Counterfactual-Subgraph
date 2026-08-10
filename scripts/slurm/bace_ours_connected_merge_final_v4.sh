#!/bin/bash
#SBATCH --job-name=bace_ours_conn_merge_final_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
BASE_POOL=${BASE_POOL:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/round1_merged/candidate_pool.jsonl}
ROUND2_ROOT=${ROUND2_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/round2}
TRAIN_CSV=${TRAIN_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/merged}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$BASE_POOL" "$TRAIN_CSV" "$TEST_CSV"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
tmp_ids=$(mktemp -d /tmp/bace_connected_final_ids.XXXXXX); trap 'rm -rf "$tmp_ids"' EXIT
python - "$TRAIN_CSV" "$TEST_CSV" "$tmp_ids" <<'PY'
import csv,sys
from pathlib import Path
for name,path in zip(("train","test"),sys.argv[1:3]):
 rows=list(csv.DictReader(open(path,newline="",encoding="utf-8-sig")))
 Path(sys.argv[3],name+".txt").write_text("".join(str(r["molecule_id"])+"\n" for r in rows))
PY
args=(python scripts/merge_bace_ours_candidate_pools_v2.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --base-pool "$BASE_POOL" --train-parent-ids "$tmp_ids/train.txt" --test-parent-ids "$tmp_ids/test.txt" --source-metadata-csv "$TRAIN_CSV" --output-dir "$OUTPUT_DIR" --require-connected-source-residual --candidateaware-v4)
if [ ! -s "$ROUND2_ROOT/_SKIPPED.json" ]; then
  for regime in D E; do path="$ROUND2_ROOT/regime${regime}/candidate_pool.jsonl"; test -s "$path" || exit 2; args+=(--regime-pool "$path"); done
fi
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_OURS_CONNECTED_FINAL_MERGE_V4_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python - "$OUTPUT_DIR/candidate_pool_audit.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["candidateaware_v4"] and p["test_source_parent_count"]==0 and p["all_retained_source_residuals_connected"]
PY
echo '[BACE_OURS_CONNECTED_FINAL_MERGE_V4_SUCCESS]'
