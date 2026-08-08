#!/bin/bash
#SBATCH --job-name=bbbp_method_eval
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
: "${BBBP_METHOD:?BBBP_METHOD is required}"
: "${CANDIDATE_PATH:?CANDIDATE_PATH is required}"
: "${EXPECTED_TEST_PARENTS:?EXPECTED_TEST_PARENTS is required}"
args=(
  scripts/evaluate_bbbp_method.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --method "$BBBP_METHOD"
  --candidate-path "$CANDIDATE_PATH"
  --teacher-path "${TEACHER_PATH:-outputs/hpc/oracle/bbbp/bbbp_teacher.pkl}"
  --test-csv "${TEST_CSV:-data/processed/BBBP/test.csv}"
  --thresholds-json "${THRESHOLDS_JSON:-outputs/hpc/eval/paper/bbbp_common3_standardized_v1/thresholds.json}"
  --work-dir "${WORK_DIR:?WORK_DIR is required}"
  --output-dir "${OUTPUT_DIR:?OUTPUT_DIR is required}"
  --expected-test-parents "$EXPECTED_TEST_PARENTS"
  --protocol-manifest "${PROTOCOL_MANIFEST:?PROTOCOL_MANIFEST is required}"
  --split-manifest "${SPLIT_MANIFEST:?SPLIT_MANIFEST is required}"
  --split-leakage-audit "${SPLIT_LEAKAGE_AUDIT:?SPLIT_LEAKAGE_AUDIT is required}"
  --candidate-lineage-audit "${CANDIDATE_LINEAGE_AUDIT:?CANDIDATE_LINEAGE_AUDIT is required}"
)
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then args+=(--validate-only); fi
if [ "${DRY_RUN:-0}" = "1" ]; then args+=(--dry-run); fi
python "${args[@]}"
