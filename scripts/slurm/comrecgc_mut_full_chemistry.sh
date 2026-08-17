#!/bin/bash
#SBATCH --job-name=comrecgc_mut_full_chem
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
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

BASE_ROOT="${BASE_ROOT:-outputs/hpc/baselines/comrecgc/mutagenicity/full_chemrepair_v1}"
DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
GENERATION_DIR="${GENERATION_DIR:-$BASE_ROOT/generation}"
TRACE_DIR="${TRACE_DIR:-$GENERATION_DIR/trace}"
COMMON_RECOURSE_DIR="${COMMON_RECOURSE_DIR:-$BASE_ROOT/common_recourse}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/chemistry}"
PREREGISTRATION="${PREREGISTRATION:-outputs/hpc/baselines/comrecgc/preregistration/mutagenicity_full_deterministic_chem_repair_v1.json}"
for input in "$GENERATION_DIR/run_manifest.json" "$TRACE_DIR/candidate_action_lineage.json" "$COMMON_RECOURSE_DIR/selected_common_recourses.json"; do
  [[ -e "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR/_RUN_COMPLETE.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] output complete=$OUTPUT_DIR" >&2; exit 2; }
# Full uses the trace implementation already proven parity-neutral on the
# preregistered 64-parent smoke; a second 50k-step trace-disabled run is not a
# scientific input and is deliberately not performed.
TRACE_PARITY_PATH="${TRACE_PARITY_PATH:-outputs/hpc/baselines/comrecgc/mutagenicity/recovery_trace_v1/generation/trace_parity.json}"
test -s "$TRACE_PARITY_PATH" || {
  echo "[COMRECGC_CONFIG_ERROR] full trace parity evidence missing=$TRACE_PARITY_PATH" >&2; exit 2;
}
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_full_chemistry parents=1448 dynamic_candidate_count=true"
python scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py \
  --project-root "$PROJECT_ROOT" \
  --dataset-dir "$DATASET_DIR" \
  --generation-dir "$GENERATION_DIR" \
  --trace-lineage-path "$TRACE_DIR/candidate_action_lineage.json" \
  --trace-parity-path "$TRACE_PARITY_PATH" \
  --common-recourse-dir "$COMMON_RECOURSE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preregistration-path "$PREREGISTRATION" \
  --parent-limit 1448
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_MUT_FULL_CHEMISTRY_SUCCESS]"
