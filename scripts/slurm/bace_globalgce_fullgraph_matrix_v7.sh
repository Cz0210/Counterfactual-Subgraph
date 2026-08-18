#!/bin/bash
#SBATCH --job-name=bace_globalgce_fullgraph_matrix_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
MIN_FREQ=${MIN_FREQ:?MIN_FREQ is required}
SOURCE_CASE_ROOT=${SOURCE_CASE_ROOT:?SOURCE_CASE_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v7}
CASE_ROOT=$OUTPUT_ROOT/candidates/min_freq_$MIN_FREQ
SELECTED_CSV=$SOURCE_CASE_ROOT/selector/selected_top20_for_eval.csv
ADAPTED_CSV=$CASE_ROOT/fullgraph_candidates.csv
MATRIX_ROOT=$CASE_ROOT/fullgraph_calibration
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/thresholds.json}
PERSISTENT_SCRATCH_ROOT=${PERSISTENT_SCRATCH_ROOT:-/share/project/p20526/u20526/counterfactual-subgraph}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$PERSISTENT_SCRATCH_ROOT/globalgce_bace_v7/fullgraph_wnode.sqlite3}
for path in "$SELECTED_CSV" "$CALIBRATION_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$THRESHOLDS_JSON"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
WNODE_THRESHOLDS=$(python scripts/resolve_frozen_wnode_thresholds.py --config configs/hpc.yaml --thresholds-json "$THRESHOLDS_JSON" --format csv)
echo "hostname=$(hostname) min_freq=$MIN_FREQ commit=$(git rev-parse HEAD) gpus=0"
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/globalgce/export_bace_frequency_pool.py --selected-csv "$SELECTED_CSV" --output "$ADAPTED_CSV" --validate-only
  exit 0
fi
test ! -e "$CASE_ROOT" || { echo "output collision: $CASE_ROOT" >&2; exit 2; }
mkdir -p "$CASE_ROOT" "$(dirname "$WNODE_CACHE_DB")"
python scripts/baselines/globalgce/export_bace_frequency_pool.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --selected-csv "$SELECTED_CSV" --output "$ADAPTED_CSV"
python scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --dataset-csv "$CALIBRATION_CSV" --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --label 1 --smiles-col smiles --label-col label --cf-mode strict_flip \
  --output-dir "$MATRIX_ROOT" --max-parents 0 --max-candidates 20 \
  --wnode-thresholds "$WNODE_THRESHOLDS" --feature-cost cosine --node-mass uniform \
  --size-penalty-beta 0.0 --device cpu --preselected-topk 20 \
  --require-preselected-topk 1 --selection-method globalgce_frequency_top20_train_support_v1 \
  --action-semantics-version connected_sanitized_residual_v1 \
  --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1 \
  --wnode-cache-db "$WNODE_CACHE_DB" --skip-redundancy 1 --resume 1 \
  --run-ours 0 --run-fullgraph 1 --fullgraph-candidates-path "$ADAPTED_CSV" \
  --fullgraph-method-name GlobalGCE
python - "$MATRIX_ROOT/details/pair_details.csv" <<'PY'
import csv,sys
from src.baselines.globalgce_bace_action_adapter import assert_nonzero_fullgraph_applicability
rows=list(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8")))
audit=assert_nonzero_fullgraph_applicability(rows, expected_pairs=1200)
assert audit["applicable_count"] == 1200
print(audit)
PY
python scripts/baselines/globalgce/score_bace_min_freq.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --min-freq "$MIN_FREQ" --pool-path "$SOURCE_CASE_ROOT" --selection-csv "$ADAPTED_CSV" --matrix-root "$MATRIX_ROOT" --thresholds-json "$THRESHOLDS_JSON" --output "$CASE_ROOT/calibration_metrics.json"
