#!/bin/bash
#SBATCH --job-name=comrecgc_project_chem
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

DATASET="${DATASET:-}"
MODE="${MODE:-full}"
BASE_ROOT="${BASE_ROOT:-}"
[[ "$DATASET" == "aids" || "$DATASET" == "mutagenicity" ]] || { echo "[COMRECGC_CONFIG_ERROR] dataset=$DATASET" >&2; exit 2; }
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || { echo "[COMRECGC_CONFIG_ERROR] mode=$MODE" >&2; exit 2; }
[[ -n "$BASE_ROOT" ]] || { echo "[COMRECGC_CONFIG_ERROR] explicit BASE_ROOT required" >&2; exit 2; }
if [[ "$MODE" == "smoke" ]]; then EXPECTED_PARENT_LIMIT=64; elif [[ "$DATASET" == "aids" ]]; then EXPECTED_PARENT_LIMIT=1283; else EXPECTED_PARENT_LIMIT=1448; fi
PARENT_LIMIT="${PARENT_LIMIT:-$EXPECTED_PARENT_LIMIT}"
[[ "$PARENT_LIMIT" == "$EXPECTED_PARENT_LIMIT" ]] || { echo "[COMRECGC_CONFIG_ERROR] parent_limit=$PARENT_LIMIT expected=$EXPECTED_PARENT_LIMIT" >&2; exit 2; }
GENERATION_DIR="${GENERATION_DIR:-$BASE_ROOT/generation}"
TRACE_DIR="${TRACE_DIR:-$GENERATION_DIR/trace}"
COMMON_RECOURSE_DIR="${COMMON_RECOURSE_DIR:-$BASE_ROOT/common_recourse}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/chemistry}"
PREREGISTRATION="${PREREGISTRATION:-$BASE_ROOT/preregistration/deterministic_chem_repair.json}"
if [[ "$DATASET" == "aids" ]]; then
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}"
  SOURCE_CSV="${SOURCE_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}"
  SOURCE_ARGS=(--source-csv "$SOURCE_CSV")
  TRACE_EVIDENCE_PATH="${TRACE_EVIDENCE_PATH:-$TRACE_DIR/trace_summary.json}"
else
  DATASET_DIR="${DATASET_DIR:-outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
  SOURCE_ARGS=()
  TRACE_EVIDENCE_PATH="${TRACE_EVIDENCE_PATH:-${TRACE_PARITY_PATH:-}}"
fi
for input in "$GENERATION_DIR/run_manifest.json" "$TRACE_DIR/candidate_action_lineage.json" "$TRACE_EVIDENCE_PATH" "$COMMON_RECOURSE_DIR/selected_common_recourses.json"; do
  [[ -s "$input" ]] || { echo "[COMRECGC_CONFIG_ERROR] missing=$input" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "[COMRECGC_CONFIG_ERROR] output exists=$OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$PREREGISTRATION")"
echo "[COMRECGC_STAGE_CONFIG] stage=project_chemistry dataset=$DATASET mode=$MODE parents=$PARENT_LIMIT output=$OUTPUT_DIR"
python scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py \
  --project-root "$PROJECT_ROOT" \
  --dataset "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  "${SOURCE_ARGS[@]}" \
  --generation-dir "$GENERATION_DIR" \
  --trace-lineage-path "$TRACE_DIR/candidate_action_lineage.json" \
  --trace-evidence-path "$TRACE_EVIDENCE_PATH" \
  --common-recourse-dir "$COMMON_RECOURSE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --preregistration-path "$PREREGISTRATION" \
  --parent-limit "$PARENT_LIMIT"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[COMRECGC_PROJECT_CHEMISTRY_SUCCESS]"
