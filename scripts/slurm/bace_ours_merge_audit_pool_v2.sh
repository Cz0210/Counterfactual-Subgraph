#!/bin/bash
#SBATCH --job-name=bace_ours_merge_v2
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
BASE_POOL=${BASE_POOL:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours/candidate_pool.jsonl}
REGIME_ROOT=${REGIME_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_multiseed_multitemp_v2/regimes}
TRAIN_CSV=${TRAIN_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
TEST_CSV=${TEST_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_multiseed_multitemp_v2/merged}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$BASE_POOL" "$TRAIN_CSV" "$TEST_CSV" "$REGIME_ROOT/regime1/candidate_pool.jsonl" "$REGIME_ROOT/regime2/candidate_pool.jsonl" "$REGIME_ROOT/regime3/candidate_pool.jsonl"; do test -s "$path"; done
tmp_ids=$(mktemp -d /tmp/bace_pool_ids.XXXXXX)
trap 'rm -f "$tmp_ids/train.txt" "$tmp_ids/test.txt"; rmdir "$tmp_ids" 2>/dev/null || true' EXIT
python - "$TRAIN_CSV" "$TEST_CSV" "$tmp_ids" <<'PY'
import csv,sys
from pathlib import Path
for name,path in zip(("train","test"),sys.argv[1:3]):
 with open(path,newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
 Path(sys.argv[3],name+".txt").write_text("".join(str(r["molecule_id"])+"\n" for r in rows))
PY
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo "[BACE_OURS_MERGE_V2_VALIDATE_OK]"; exit 0; fi
python scripts/merge_bace_ours_candidate_pools_v2.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --base-pool "$BASE_POOL" \
  --regime-pool "$REGIME_ROOT/regime1/candidate_pool.jsonl" \
  --regime-pool "$REGIME_ROOT/regime2/candidate_pool.jsonl" \
  --regime-pool "$REGIME_ROOT/regime3/candidate_pool.jsonl" \
  --train-parent-ids "$tmp_ids/train.txt" --test-parent-ids "$tmp_ids/test.txt" \
  --output-dir "$OUTPUT_DIR"
echo "[BACE_OURS_MERGE_V2_SUCCESS]"
