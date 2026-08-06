#!/bin/bash
#SBATCH --job-name=comrecgc_mut_chem
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
GENERATION_DIR="${GENERATION_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity/recovery_trace_v1/generation}"
TRACE_DIR="${TRACE_DIR:-$GENERATION_DIR/trace}"
COMMON_RECOURSE_DIR="${COMMON_RECOURSE_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity/smoke_comrecgc_smoke_budget_retry_20260806_v4/common_recourse}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity_chemistry_audit_v1}"
PREREGISTRATION="${PREREGISTRATION:-outputs/hpc/baselines/comrecgc/preregistration/mutagenicity_deterministic_chem_repair_v1.json}"
for input in "$DATASET_DIR" "$GENERATION_DIR/run_manifest.json" "$GENERATION_DIR/trace_parity.json" "$TRACE_DIR/candidate_action_lineage.json" "$COMMON_RECOURSE_DIR/selected_common_recourses.json"; do
  [[ -e "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete=$OUTPUT_DIR" >&2; exit 2; }
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_chemistry_audit parents=64 candidates=164 medoids=4"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -m py_compile scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py
python scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py \
  --project-root "$PROJECT_ROOT" \
  --dataset-dir "$DATASET_DIR" \
  --generation-dir "$GENERATION_DIR" \
  --trace-lineage-path "$TRACE_DIR/candidate_action_lineage.json" \
  --trace-parity-path "$GENERATION_DIR/trace_parity.json" \
  --common-recourse-dir "$COMMON_RECOURSE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preregistration-path "$PREREGISTRATION" \
  --parent-limit 64 \
  --expected-candidate-count 164 \
  --expected-medoid-count 4
test -s "$OUTPUT_DIR/audit.json"
grep -Fq '[COMRECGC_MUT_CHEMISTRY_ENGINEERING_SMOKE_PASS]' "$OUTPUT_DIR/audit.txt"
echo "[COMRECGC_MUT_CHEMISTRY_AUDIT_SUCCESS]"
