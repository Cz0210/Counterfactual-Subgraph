#!/usr/bin/env bash
# Build the frozen GCFExplainer fullgraph WNode calibration matrix.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=mut_gcf_wcal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

FROZEN_ROOT="${FROZEN_ROOT:-outputs/hpc/mutagenicity/frozen_candidates/gcfexplainer_native5000_top20_v1}"
FULLGRAPH_CANDIDATES_PATH="${FULLGRAPH_CANDIDATES_PATH:-$FROZEN_ROOT/export/selected_top20.csv}"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-$FROZEN_ROOT/frozen_candidate_manifest.json}"
DATASET_CSV="${DATASET_CSV:-outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/calibration_source_label1_teacher_correct.csv}"
TEACHER_PATH="${TEACHER_PATH:-outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
MOLCLR_ROOT="${MOLCLR_ROOT:-pretrained_models/MolCLR}"
MOLCLR_CKPT="${MOLCLR_CKPT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/mutagenicity/eval/gcfexplainer_native5000_top20_wnode_calibration_p235_k20_v1}"
WNODE_CACHE_DB="${WNODE_CACHE_DB:-outputs/hpc/cache/distance_cache/mutagenicity_gcfexplainer_wnode_v1.sqlite}"
NODE_EMB_CACHE_DIR="${NODE_EMB_CACHE_DIR:-outputs/hpc/cache/molclr_node_embeddings}"
RESUME="${RESUME:-false}"
PARTIAL_EVERY="${PARTIAL_EVERY:-100}"

EXPECTED_PARENT_COUNT=235
EXPECTED_CANDIDATE_COUNT=20
EXPECTED_PAIR_COUNT=4700
EXPECTED_CANDIDATE_CSV_SHA256="e968fa140a4e34ad6abc6430b17f539e5d568b319d4856d74763496e9181f341"
EXPECTED_CANDIDATE_ORDER_SHA256="e8758517a1c81fa150298497a8a799806abbe5a8c17ba048790636aacf4a1a46"
EXPECTED_NATIVE_RANKS="112,120,177,179,195,217,388,442,605,701,794,810,1034,1095,1417,1786,1788,1975,3815,4198"
EXPECTED_TEACHER_SHA256="af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
EXPECTED_MOLCLR_SHA256="93bc4f02ea8847cd44fa21ec3f65600ff2f4a7ae6d3a85e8519a5bcc56afc20a"
METHOD_NAME="GCFExplainer-Top20"
SELECTION_METHOD="native_gcf_summary_rank_filtered_by_validity"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PROJECT_ROOT" "$1" ;;
  esac
}

for variable in FROZEN_ROOT FULLGRAPH_CANDIDATES_PATH FROZEN_MANIFEST DATASET_CSV TEACHER_PATH MOLCLR_ROOT MOLCLR_CKPT THRESHOLDS_JSON OUTPUT_DIR WNODE_CACHE_DB NODE_EMB_CACHE_DIR; do
  printf -v "$variable" '%s' "$(resolve_path "${!variable}")"
done

for required in "$FULLGRAPH_CANDIDATES_PATH" "$FROZEN_MANIFEST" "$DATASET_CSV" "$TEACHER_PATH" "$MOLCLR_ROOT" "$MOLCLR_CKPT" "$THRESHOLDS_JSON"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Required input does not exist: $required" >&2
    exit 3
  fi
done

case "$(printf '%s' "$RESUME" | tr '[:upper:]' '[:lower:]')" in
  true|1|yes|on) RESUME_VALUE=1 ;;
  false|0|no|off) RESUME_VALUE=0 ;;
  *) echo "[ERROR] RESUME must be boolean." >&2; exit 3 ;;
esac
if [[ -s "$OUTPUT_DIR/_RUN_COMPLETE.json" ]]; then
  echo "[ERROR] Calibration run is already complete: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]] && [[ "$RESUME_VALUE" -eq 0 ]]; then
  echo "[ERROR] OUTPUT_DIR is non-empty and RESUME=false: $OUTPUT_DIR" >&2
  exit 3
fi

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[ERROR] SHA256 mismatch for $path: actual=$actual expected=$expected" >&2
    exit 4
  fi
}
check_sha256 "$FULLGRAPH_CANDIDATES_PATH" "$EXPECTED_CANDIDATE_CSV_SHA256"
check_sha256 "$TEACHER_PATH" "$EXPECTED_TEACHER_SHA256"
check_sha256 "$MOLCLR_CKPT" "$EXPECTED_MOLCLR_SHA256"

IFS=$'\t' read -r WNODE_THRESHOLDS THETA_STAR COST_CAP THRESHOLDS_SHA256 < <(
  python - "$THRESHOLDS_JSON" <<'PY'
import sys
from src.eval.fullgraph_wnode_artifacts import load_frozen_threshold_contract

contract = load_frozen_threshold_contract(sys.argv[1])
print("\t".join((
    ",".join(format(value, ".17g") for value in contract["thresholds"]),
    format(contract["theta_star"], ".17g"),
    format(contract["cost_cap"], ".17g"),
    contract["thresholds_json_sha256"],
)))
PY
)

python - "$FULLGRAPH_CANDIDATES_PATH" "$FROZEN_MANIFEST" "$EXPECTED_CANDIDATE_CSV_SHA256" "$EXPECTED_CANDIDATE_ORDER_SHA256" "$EXPECTED_NATIVE_RANKS" "$SELECTION_METHOD" <<'PY'
import json
import sys
from src.eval.fullgraph_wnode_artifacts import validate_frozen_candidate_contract

result = validate_frozen_candidate_contract(
    candidates_csv=sys.argv[1],
    frozen_manifest_path=sys.argv[2],
    expected_count=20,
    expected_csv_sha256=sys.argv[3],
    expected_order_sha256=sys.argv[4],
    expected_native_ranks=[int(value) for value in sys.argv[5].split(",")],
    expected_selection_method=sys.argv[6],
)
print(json.dumps(result, sort_keys=True))
PY

mkdir -p "$OUTPUT_DIR" "$(dirname "$WNODE_CACHE_DB")" "$NODE_EMB_CACHE_DIR" "$PROJECT_ROOT/logs"

echo "===== GCFEXPLAINER MUTAGENICITY WNODE CALIBRATION ====="
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "pwd=$PWD"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(command -v python)"
echo "candidate_csv=$FULLGRAPH_CANDIDATES_PATH"
echo "candidate_csv_sha256=$EXPECTED_CANDIDATE_CSV_SHA256"
echo "candidate_order_sha256=$EXPECTED_CANDIDATE_ORDER_SHA256"
echo "dataset_csv=$DATASET_CSV"
echo "expected_parent_count=$EXPECTED_PARENT_COUNT"
echo "expected_candidate_count=$EXPECTED_CANDIDATE_COUNT"
echo "expected_pair_count=$EXPECTED_PAIR_COUNT"
echo "thresholds_json=$THRESHOLDS_JSON"
echo "thresholds_json_sha256=$THRESHOLDS_SHA256"
echo "theta_star=$THETA_STAR"
echo "cost_cap=$COST_CAP"
echo "candidate_selection_performed=false"
echo "selection_used_calibration=false"
echo "selection_used_test=false"
echo "candidate_set_preselected=true"
echo "selection_performed_in_eval=false"
python --version
nvidia-smi

python scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "$DATASET_CSV" \
  --teacher-path "$TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CKPT" \
  --label 1 \
  --smiles-col smiles \
  --label-col label \
  --cf-mode strict_flip \
  --output-dir "$OUTPUT_DIR" \
  --max-parents "$EXPECTED_PARENT_COUNT" \
  --max-candidates "$EXPECTED_CANDIDATE_COUNT" \
  --wnode-thresholds "$WNODE_THRESHOLDS" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-emb-cache-dir "$NODE_EMB_CACHE_DIR" \
  --feature-cost cosine \
  --node-mass uniform \
  --size-penalty-beta 0.0 \
  --device cuda \
  --skip-redundancy 1 \
  --partial-every "$PARTIAL_EVERY" \
  --resume "$RESUME_VALUE" \
  --run-distance-self-test 1 \
  --run-ours 0 \
  --run-fullgraph 1 \
  --fullgraph-candidates-path "$FULLGRAPH_CANDIDATES_PATH" \
  --fullgraph-method-name "$METHOD_NAME" \
  --selection-method "$SELECTION_METHOD" \
  --preselected-topk "$EXPECTED_CANDIDATE_COUNT" \
  --require-preselected-topk 1

mv "$OUTPUT_DIR/_RUN_COMPLETE.json" "$OUTPUT_DIR/_EVALUATOR_COMPLETE.json"
python - "$OUTPUT_DIR" "$FULLGRAPH_CANDIDATES_PATH" "$FROZEN_MANIFEST" "$THRESHOLDS_JSON" "$EXPECTED_CANDIDATE_CSV_SHA256" "$EXPECTED_CANDIDATE_ORDER_SHA256" "$EXPECTED_TEACHER_SHA256" "$EXPECTED_MOLCLR_SHA256" "$EXPECTED_NATIVE_RANKS" "$METHOD_NAME" "$SELECTION_METHOD" <<'PY'
import json
import sys
from src.eval.fullgraph_wnode_artifacts import finalize_fullgraph_evaluation_run

result = finalize_fullgraph_evaluation_run(
    run_dir=sys.argv[1],
    frozen_candidates_csv=sys.argv[2],
    frozen_manifest_path=sys.argv[3],
    thresholds_json=sys.argv[4],
    cohort_name="calibration",
    expected_parent_count=235,
    expected_candidate_count=20,
    expected_pair_count=4700,
    expected_candidate_csv_sha256=sys.argv[5],
    expected_candidate_order_sha256=sys.argv[6],
    expected_teacher_sha256=sys.argv[7],
    expected_molclr_checkpoint_sha256=sys.argv[8],
    expected_native_ranks=[int(value) for value in sys.argv[9].split(",")],
    expected_method=sys.argv[10],
    expected_selection_method=sys.argv[11],
)
print(json.dumps(result, sort_keys=True))
PY

test -s "$OUTPUT_DIR/audit.json"
test -s "$OUTPUT_DIR/run_manifest.json"
test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"
echo "[MUTAGENICITY_GCFEXPLAINER_WNODE_CALIBRATION_OK]"
