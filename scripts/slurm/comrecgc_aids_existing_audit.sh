#!/bin/bash
#SBATCH --job-name=comrecgc_aids_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0

SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-outputs/hpc/baselines/comrecgc/native_smoke/aids/comrecgc_native_common_64p_20260806_v6/counterfactuals.pt}"
EXPECTED_SHA256="${EXPECTED_SHA256:-096ddd0f4ac31126a0665a11effb7362c2137229ff6b53b50e16c081ef6c274a}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/aids_native_dbscan_audit_v1}"
[[ -f "$SOURCE_ARTIFACT" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$SOURCE_ARTIFACT" >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_ARTIFACT" | awk '{print $1}')" == "$EXPECTED_SHA256" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] AIDS frozen artifact SHA256 mismatch" >&2; exit 2;
}
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || {
  echo "[COMRECGC_CONFIG_ERROR] output already complete=$OUTPUT_DIR" >&2; exit 2;
}

echo "[COMRECGC_RECOVERY_CONFIG] stage=aids_existing_audit source=$SOURCE_ARTIFACT output=$OUTPUT_DIR"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -m py_compile scripts/baselines/comrecgc/audit_aids_native_dbscan.py
python scripts/baselines/comrecgc/audit_aids_native_dbscan.py \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --counterfactuals-path "$SOURCE_ARTIFACT" \
  --output-dir "$OUTPUT_DIR" \
  --parent-limit 64 \
  --expected-sha256 "$EXPECTED_SHA256" \
  --expected-candidates 31 \
  --expected-distance-pairs 1984 \
  --expected-eligible-pairs 28 \
  --device cuda:0 \
  --batch-size 128
test -s "$OUTPUT_DIR/audit.json"
grep -Fq '[COMRECGC_AIDS_DBSCAN_AUDIT_PASS]' "$OUTPUT_DIR/audit.txt"
echo "[COMRECGC_AIDS_EXISTING_AUDIT_SUCCESS]"
