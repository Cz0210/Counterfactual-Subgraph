#!/bin/bash
#SBATCH --job-name=comrecgc_aids_full
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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

PREREGISTRATION="${PREREGISTRATION:-outputs/hpc/baselines/comrecgc/preregistration/aids_native_full_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/native_full/aids/native_full_v1}"
CACHE_TRUST_BEFORE="${CACHE_TRUST_BEFORE:-${OUTPUT_DIR}.cache_trust_before.json}"
TRUSTED_DATASET_PAYLOAD="${TRUSTED_DATASET_PAYLOAD:-${OUTPUT_DIR}.trusted_dataset.pt}"
CACHE_TRUST_AFTER_LOAD="${CACHE_TRUST_AFTER_LOAD:-${OUTPUT_DIR}.cache_trust_after_load.json}"
[[ -d external/COMRECGC/.git ]] || { echo "[COMRECGC_CONFIG_ERROR] pinned upstream missing" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete=$OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ ! -e "$PREREGISTRATION" ]]; then
  python scripts/baselines/comrecgc/preregister_recovery.py aids-native-full \
    --project-root "$PROJECT_ROOT" \
    --upstream-root external/COMRECGC \
    --output-path "$PREREGISTRATION"
fi
test -s "$PREREGISTRATION"
echo "[COMRECGC_RECOVERY_CONFIG] stage=aids_native_full steps=50000 heads=5 k=100000 sample_size=10000 seed=0"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD) prereg_sha=$(sha256sum "$PREREGISTRATION" | awk '{print $1}')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -m py_compile scripts/baselines/comrecgc/run_generation.py
python scripts/baselines/comrecgc/audit_trusted_aids_cache.py \
  --upstream-root external/COMRECGC \
  --output "$CACHE_TRUST_BEFORE"
CACHE_SHA256_BEFORE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_sha256"])' "$CACHE_TRUST_BEFORE")"
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/baselines/comrecgc/materialize_trusted_aids_cache.py \
  --cache-trust-json "$CACHE_TRUST_BEFORE" \
  --output "$TRUSTED_DATASET_PAYLOAD"
python scripts/baselines/comrecgc/audit_trusted_aids_cache.py \
  --upstream-root external/COMRECGC \
  --output "$CACHE_TRUST_AFTER_LOAD" \
  --expected-inventory-sha256 "$CACHE_SHA256_BEFORE"
python scripts/baselines/comrecgc/run_generation.py \
  --route native \
  --dataset aids \
  --mode full \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --output-dir "$OUTPUT_DIR" \
  --trusted-dataset-payload "$TRUSTED_DATASET_PAYLOAD" \
  --expected-cache-inventory-sha256 "$CACHE_SHA256_BEFORE" \
  --device cuda:0
python scripts/baselines/comrecgc/audit_trusted_aids_cache.py \
  --upstream-root external/COMRECGC \
  --output "$OUTPUT_DIR/cache_trust_after.json" \
  --expected-inventory-sha256 "$CACHE_SHA256_BEFORE"
test -s "$OUTPUT_DIR/run_manifest.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
test -s "$OUTPUT_DIR/native_common_recourse.json"
echo "[COMRECGC_AIDS_NATIVE_FULL_EXECUTION_PASS]"
