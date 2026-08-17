#!/bin/bash
#SBATCH --job-name=comrecgc_storage_preflight_v6
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs
DATASET=${DATASET:?DATASET is required}
FAILED_GENERATION_DIR=${FAILED_GENERATION_DIR:?FAILED_GENERATION_DIR is required}
PERSISTENT_SCRATCH_ROOT=${PERSISTENT_SCRATCH_ROOT:-/share/project/p20526/u20526/counterfactual-subgraph}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
test -d "$FAILED_GENERATION_DIR" || { echo "missing failed generation: $FAILED_GENERATION_DIR" >&2; exit 2; }
test -d "$PERSISTENT_SCRATCH_ROOT" || { echo "missing scratch: $PERSISTENT_SCRATCH_ROOT" >&2; exit 2; }
echo "hostname=$(hostname) python=$(which python) dataset=$DATASET commit=$(git rev-parse HEAD)"
python --version
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  echo '[COMRECGC_STORAGE_PREFLIGHT_V6_VALIDATE_OK]'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
python scripts/ops/preflight_persistent_scratch.py \
  --root "$PERSISTENT_SCRATCH_ROOT" \
  --output "$OUTPUT_DIR/storage_preflight.json" \
  --min-free-gib 50 \
  --min-free-inodes 100000
python scripts/baselines/comrecgc/audit_generation_checkpoint.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --generation-dir "$FAILED_GENERATION_DIR" \
  --output "$OUTPUT_DIR/checkpoint_audit.json"
python - "$DATASET" "$OUTPUT_DIR/checkpoint_audit.json" "$OUTPUT_DIR/recovery_decision.json" <<'PY'
import json,sys
dataset,source,target=sys.argv[1:]
audit=json.load(open(source))
decision={
  "dataset":dataset,
  "checkpoint_audit":source,
  "checkpoint_safe":bool(audit.get("RESUME_SAFE")),
  "resume_or_fresh":"resume" if audit.get("RESUME_SAFE") else "fresh",
  "scientific_resume_claimed":bool(audit.get("RESUME_SAFE")),
}
open(target,"w").write(json.dumps(decision,indent=2,sort_keys=True)+"\n")
PY
python - "$OUTPUT_DIR/_RUN_COMPLETE.json" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({"run_complete":True,"storage_preflight_pass":True},indent=2)+"\n")
PY
echo '[COMRECGC_STORAGE_PREFLIGHT_V6_PASS]'
