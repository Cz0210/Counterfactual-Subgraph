#!/bin/bash
#SBATCH --job-name=comrecgc_mut_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
mkdir -p logs

MODE="${MODE:-smoke}"
RESUME="${RESUME:-false}"
[[ "$MODE" == "smoke" || "$MODE" == "full" ]] || { echo "[COMRECGC_CONFIG_ERROR] mode=$MODE" >&2; exit 2; }
[[ "$RESUME" == "true" || "$RESUME" == "false" ]] || { echo "[COMRECGC_CONFIG_ERROR] resume=$RESUME" >&2; exit 2; }
if [[ "$MODE" == "smoke" ]]; then
  CHEMISTRY_DIR="${CHEMISTRY_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity_chemistry_audit_v1}"
  DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/train_source_label1_teacher_correct.csv}"
  EXPECTED_PARENT_COUNT=16
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/mutagenicity_chemistry_audit_v1/unified_eval_smoke_v1}"
else
  [[ -n "${BASE_ROOT:-}" ]] || { echo "[COMRECGC_CONFIG_ERROR] full requires explicit BASE_ROOT" >&2; exit 2; }
  CHEMISTRY_DIR="${CHEMISTRY_DIR:-$BASE_ROOT/chemistry}"
  DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/test_source_label1_teacher_correct.csv}"
  EXPECTED_PARENT_COUNT=217
  OUTPUT_DIR="${OUTPUT_DIR:-$BASE_ROOT/unified_eval}"
fi
TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
MOLCLR_ROOT="${MOLCLR_ROOT:-pretrained_models/MolCLR}"
MOLCLR_CHECKPOINT="${MOLCLR_CHECKPOINT:-pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json}"
EXPECTED_TEACHER_SHA256="af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
EXPECTED_MOLCLR_SHA256="93bc4f02ea8847cd44fa21ec3f65600ff2f4a7ae6d3a85e8519a5bcc56afc20a"

[[ -s "$CHEMISTRY_DIR/run_manifest.json" ]] || { echo "[COMRECGC_CONFIG_ERROR] chemistry manifest missing: $CHEMISTRY_DIR" >&2; exit 2; }
[[ -s "$CHEMISTRY_DIR/medoid_validity.csv" ]] || { echo "[COMRECGC_CONFIG_ERROR] medoid validity missing: $CHEMISTRY_DIR" >&2; exit 2; }
[[ -s "$DATASET_CSV" && -s "$TEACHER_PATH" && -s "$MOLCLR_CHECKPOINT" && -s "$THRESHOLDS_JSON" ]] || { echo "[COMRECGC_CONFIG_ERROR] a frozen evaluation input is missing" >&2; exit 2; }
[[ "$(sha256sum "$TEACHER_PATH" | awk '{print $1}')" == "$EXPECTED_TEACHER_SHA256" ]] || { echo "[COMRECGC_CONFIG_ERROR] teacher SHA256 mismatch" >&2; exit 2; }
[[ "$(sha256sum "$MOLCLR_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_MOLCLR_SHA256" ]] || { echo "[COMRECGC_CONFIG_ERROR] MolCLR SHA256 mismatch" >&2; exit 2; }
if [[ "$RESUME" != "true" && -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[COMRECGC_CONFIG_ERROR] non-empty output with RESUME=false: $OUTPUT_DIR" >&2
  exit 2
fi
RESUME_ARGS=()
[[ "$RESUME" == "true" ]] && RESUME_ARGS=(--resume)
echo "[COMRECGC_RECOVERY_CONFIG] stage=mut_unified_eval mode=$MODE parent_count=$EXPECTED_PARENT_COUNT output=$OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/baselines/comrecgc/run_slot_unified_eval.py \
  --mode "$MODE" \
  --chemistry-dir "$CHEMISTRY_DIR" \
  --dataset-csv "$DATASET_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --thresholds-json "$THRESHOLDS_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT" \
  --max-k 20 \
  --device cuda "${RESUME_ARGS[@]}"
if [[ "$MODE" == "smoke" ]]; then
  test -s "$OUTPUT_DIR/_SMOKE_AUDIT_COMPLETE.json"
else
  test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
fi
echo "[COMRECGC_MUT_UNIFIED_EVAL_SUCCESS]"
