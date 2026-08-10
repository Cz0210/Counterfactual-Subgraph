#!/bin/bash
#SBATCH --job-name=bace_gcf_cal_wnode_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=72G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}; ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
GCF_AUDIT_ROOT=${GCF_AUDIT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_gcf_native_pool_v4}
CANDIDATE_PATH=${CANDIDATE_PATH:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/summary_retry2_valid_native_rank/export/selected_top20.csv}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_ROOT=${MOLCLR_ROOT:-$ARTIFACT_ROOT/pretrained_models/MolCLR}; MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_gcf_candidateaware_v4/calibration_run}
WNODE_CACHE_DB=${WNODE_CACHE_DB:-$ARTIFACT_ROOT/outputs/hpc/cache/distance_cache/molclr_node_wasserstein_connected_residual_v3.sqlite}
DRY_RUN=${DRY_RUN:-0}; VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$GCF_AUDIT_ROOT/run_manifest.json" "$GCF_AUDIT_ROOT/candidate_universe.jsonl" "$CANDIDATE_PATH" "$CALIBRATION_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
python - "$GCF_AUDIT_ROOT/run_manifest.json" "$GCF_AUDIT_ROOT/candidate_universe.jsonl" "$CANDIDATE_PATH" <<'PY'
import csv,json,sys
p=json.load(open(sys.argv[1])); a=p["candidate_attrition"]
assert p["validation_passed"] and p["test_loaded"] is False
assert a["scan_all"] and a["scan_exhausted"] and a["num_retained"] == 20
audit=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
frozen=list(csv.DictReader(open(sys.argv[3],newline="",encoding="utf-8-sig")))
assert [r["candidate_id"] for r in audit] == [r["candidate_id"] for r in frozen]
assert [int(r["native_rank"]) for r in audit] == [int(r["native_rank"]) for r in frozen]
assert all(bool(r.get("connected")) for r in audit)
PY
args=(python scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py
 --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
 --dataset-csv "$CALIBRATION_CSV" --teacher-path "$TEACHER_PATH"
 --molclr-root "$MOLCLR_ROOT" --molclr-checkpoint "$MOLCLR_CHECKPOINT"
 --label 1 --smiles-col smiles --label-col label --cf-mode strict_flip
 --output-dir "$OUTPUT_DIR" --max-parents 0 --max-candidates 20
 --wnode-thresholds auto_quantile --wnode-quantiles 0.05,0.10,0.20,0.30,0.50,0.70,0.90
 --feature-cost cosine --node-mass uniform --size-penalty-beta 0.0 --device cuda
 --preselected-topk 20 --require-preselected-topk 1
 --selection-method native_gcf_summary_rank_filtered_by_validity
 --action-semantics-version connected_sanitized_residual_v1
 --match-selection-policy existential_min_wnode_among_valid_connected_strict_flips_v1
 --wnode-cache-db "$WNODE_CACHE_DB" --skip-redundancy 1 --resume 0
 --run-ours 0 --run-fullgraph 1 --fullgraph-candidates-path "$CANDIDATE_PATH"
 --fullgraph-method-name GCFExplainer)
echo "hostname=$(hostname)"; echo "git_commit=$(git rev-parse HEAD)"; printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo '[BACE_GCF_CAL_WNODE_V4_VALIDATE_OK]'; exit 0; fi
test ! -e "$OUTPUT_DIR" || { echo "collision $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
python - "$OUTPUT_DIR/details/pair_details.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],newline="",encoding="utf-8")))
assert len({r["parent_id"] for r in rows}) == 60
assert all(r["candidate_smiles"] and "." not in r["candidate_smiles"] for r in rows)
assert all(r["flip_definition"] == "pred_before == target_label and pred_after != target_label" for r in rows)
PY
echo '[BACE_GCF_CAL_WNODE_V4_SUCCESS]'
