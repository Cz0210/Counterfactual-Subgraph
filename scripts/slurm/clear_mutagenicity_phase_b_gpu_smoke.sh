#!/usr/bin/env bash
# Automation-owned launcher for the already validated 64-parent CLEAR smoke.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=clear_phase_b_smoke
#SBATCH --output=logs/clear_phase_b_smoke_%j.out
#SBATCH --error=logs/clear_phase_b_smoke_%j.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] PROJECT_ROOT is required" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

: "${AUTOMATION_OUTPUT_ROOT:?AUTOMATION_OUTPUT_ROOT is required}"
: "${AUTOMATION_SUBMITTED_COMMIT:?AUTOMATION_SUBMITTED_COMMIT is required}"

case "$AUTOMATION_OUTPUT_ROOT" in
  /*) OUTPUT_DIR="$AUTOMATION_OUTPUT_ROOT" ;;
  *) OUTPUT_DIR="$PROJECT_ROOT/$AUTOMATION_OUTPUT_ROOT" ;;
esac
OUTPUT_DIR="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT_DIR")"
EXPECTED_PREFIX="$PROJECT_ROOT/outputs/hpc/mutagenicity/baselines/clear/automation_phase_b_smoke/"
if [[ "$OUTPUT_DIR/" != "$EXPECTED_PREFIX"* ]]; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] unsafe OUTPUT_DIR=$OUTPUT_DIR" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] output already exists: $OUTPUT_DIR" >&2
  exit 2
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$AUTOMATION_SUBMITTED_COMMIT" ]]; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] submitted commit mismatch" >&2
  echo "actual=$CURRENT_COMMIT" >&2
  echo "expected=$AUTOMATION_SUBMITTED_COMMIT" >&2
  exit 2
fi

for marker in \
  CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT \
  CLEAR_WRAPPER_EXPORT_TEST_COUNTERFACTUALS \
  CLEAR_WRAPPER_SUPPORT_AIDS_DATASET \
  CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET \
  CLEAR_WRAPPER_MUTAGENICITY_PHASE_B_RUNTIME; do
  if ! grep -R -q -- "$marker" baselines/clear_official/src; then
    echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] missing official patch marker=$marker" >&2
    exit 2
  fi
done
if ! grep -q -- CLEAR_WRAPPER_AIDS_WEIGHTED_GRAPHPRED \
  baselines/clear_official/src/train_pred.py; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] missing weighted GraphPred marker" >&2
  exit 2
fi
OFFICIAL_STATUS_BEFORE="$(git -C baselines/clear_official status --porcelain=v1)"

echo "===== AUTOMATION CLEAR MUTAGENICITY PHASE B GPU SMOKE ====="
echo "hostname=$(hostname)"
echo "date=$(date --iso-8601=seconds)"
echo "pwd=$PWD"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "git_commit=$CURRENT_COMMIT"
echo "python=$(command -v python)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PARENT_LIMIT=64"
echo "GRAPHPRED_EPOCHS=5"
echo "CFE_EPOCHS=5"
echo "GENERATION_CHUNK_SIZE=16"
echo "BATCH_SIZE=8"
echo "NUM_WORKERS=0"
echo "SEED=13"
echo "calibration_loaded=false"
echo "test_loaded=false"
python --version
nvidia-smi

export OUTPUT_DIR
export PARENT_LIMIT=64
export GRAPHPRED_EPOCHS=5
export CFE_EPOCHS=5
export GENERATION_CHUNK_SIZE=16
export BATCH_SIZE=8
export NUM_WORKERS=0
export SEED=13
export RESUME=false
export DEVICE=cuda

bash scripts/slurm/clear_mutagenicity_train_pool.sh

OFFICIAL_STATUS_AFTER="$(git -C baselines/clear_official status --porcelain=v1)"
if [[ "$OFFICIAL_STATUS_AFTER" != "$OFFICIAL_STATUS_BEFORE" ]]; then
  echo "[AUTOMATION_CLEAR_PHASE_B_CONFIG_ERROR] official checkout changed during smoke" >&2
  exit 2
fi
test -s "$OUTPUT_DIR/train_pool_audit.json"
test -s "$OUTPUT_DIR/summary.json"
test -s "$OUTPUT_DIR/run_manifest.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"

python - "$OUTPUT_DIR" "$CURRENT_COMMIT" <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

output_dir = Path(sys.argv[1]).resolve()
commit = sys.argv[2]
destination = output_dir / "_AUTOMATION_PHASE_B_GPU_SMOKE_COMPLETE.json"
fd, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=output_dir
)
temporary = Path(temporary_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_complete": True,
                "profile": "smoke",
                "parent_limit": 64,
                "git_commit": commit,
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "output_dir": str(output_dir),
                "calibration_loaded": False,
                "test_loaded": False,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if temporary.exists():
        temporary.unlink()
PY

echo "[AUTOMATION_CLEAR_PHASE_B_GPU_SMOKE_COMPLETE]"
