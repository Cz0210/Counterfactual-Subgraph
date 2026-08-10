#!/bin/bash
#SBATCH --job-name=bace_ours_conn_gen_r2_v4
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=72G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

MATRIX_ROOT=${MATRIX_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/round1_calibration_matrix}
UNION_AUDIT=${UNION_AUDIT:-$MATRIX_ROOT/candidate_union_summary.json}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
CALIBRATION_CSV=${CALIBRATION_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/calibration_source_label1_teacher_correct.csv}
TRAIN_CSV=${TRAIN_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
BASE_MANIFEST=${BASE_MANIFEST:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours/run_manifest.json}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/round2}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$MATRIX_ROOT/pair_matrix.jsonl" "$UNION_AUDIT" "$THRESHOLDS_JSON" "$CALIBRATION_CSV" "$TRAIN_CSV" "$BASE_MANIFEST" "$TEACHER_PATH"; do
  test -s "$path" || { echo "[BACE_CONNECTED_R2_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
EXPANSION_REQUIRED=$(python - "$UNION_AUDIT" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
assert p["split"] == "calibration" and p["test_loaded"] is False
required=(float(p["CONNECTED_STRICT_FLIP_UNION_COVERAGE"]) < .40 or float(p["CLOSE_UNION_AT_PRIMARY_THETA"]) < .25)
print("true" if required else "false")
PY
)
readarray -t MODEL_PATHS < <(python - "$BASE_MANIFEST" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]
for key in ("BASE_MODEL_PATH", "SFT_LORA_PATH", "PPO_CHECKPOINT_PATH"):
    print(p[key]["path"])
PY
)
BASE_MODEL_PATH=${MODEL_PATHS[0]}; SFT_LORA_PATH=${MODEL_PATHS[1]}; PPO_CHECKPOINT_PATH=${MODEL_PATHS[2]}
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "round2_required=$EXPANSION_REQUIRED"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_OURS_CONNECTED_R2_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_ROOT" || { echo "[BACE_CONNECTED_R2_COLLISION] $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
if [ "$EXPANSION_REQUIRED" != true ]; then
  python - "$OUTPUT_ROOT/_SKIPPED.json" "$UNION_AUDIT" <<'PY'
import json, sys
json.dump({"run_complete":True,"round2_run":False,"reason":"round1_calibration_not_candidate_limited","union_audit":sys.argv[2],"test_loaded":False},open(sys.argv[1],"w"),indent=2,sort_keys=True)
PY
  echo "[BACE_OURS_CONNECTED_R2_V4_NOT_REQUIRED]"
  exit 0
fi
SOURCE_CSV="$OUTPUT_ROOT/round2_train_source.csv"
python scripts/build_bace_round2_source_cohort.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --pair-matrix "$MATRIX_ROOT/pair_matrix.jsonl" --thresholds-json "$THRESHOLDS_JSON" \
  --calibration-csv "$CALIBRATION_CSV" --train-csv "$TRAIN_CSV" \
  --output-csv "$SOURCE_CSV" --manifest-path "$OUTPUT_ROOT/source_cohort_manifest.json" \
  --nearest-per-hard-parent 2
temperatures=(0.35 0.65)
top_ps=(0.90 0.95)
seeds=(59 71)
regimes=(D E)
for index in 0 1; do
  regime=${regimes[$index]}
  out="$OUTPUT_ROOT/regime${regime}"
  mkdir -p "$out"
  python scripts/generate_full_candidate_pool.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --dataset-path "$SOURCE_CSV" --base-model-path "$BASE_MODEL_PATH" \
    --sft-lora-path "$SFT_LORA_PATH" --ppo-checkpoint-path "$PPO_CHECKPOINT_PATH" \
    --teacher-path "$TEACHER_PATH" --out-jsonl "$out/candidate_pool.raw.jsonl" \
    --out-summary-json "$out/generation_summary.json" --label-col label --smiles-col smiles \
    --target-label 1 --prompt-mode connected_deletion_v1 --num-return-sequences 8 \
    --generation-temperature "${temperatures[$index]}" \
    --generation-top-p "${top_ps[$index]}" --generation-do-sample true \
    --max-new-tokens 96 --batch-size 1 --seed "${seeds[$index]}" \
    --enable-parent-projection --enable-projected-cf-reward \
    --enable-substructure-distance-reward --substructure-distance-reward-weight 0.3 \
    --projection-penalty 1.0 --enable-minimal-syntax-repair --enable-component-salvage
  python scripts/baselines/bace/enrich_ours_candidate_pool.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --raw-pool-jsonl "$out/candidate_pool.raw.jsonl" --parent-csv "$SOURCE_CSV" \
    --output-jsonl "$out/candidate_pool.enriched.jsonl" \
    --manifest-path "$out/candidate_lineage_manifest.json" \
    --expected-candidates-per-parent 8
  python scripts/filter_bace_connected_source_candidates.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --input-jsonl "$out/candidate_pool.enriched.jsonl" --parent-csv "$SOURCE_CSV" \
    --output-jsonl "$out/candidate_pool.jsonl" --audit-json "$out/source_gate_audit.json" \
    --generation-round 2 --generation-regime "$regime" --prompt-mode connected_deletion_v1
done
python - "$OUTPUT_ROOT/expansion_protocol.json" <<'PY'
import json, sys
json.dump({
 "schema_version":"bace_connected_candidateaware_generation_v4","round":2,"round2_run":True,
 "regimes":[{"name":"D","temperature":.35,"top_p":.90,"seed":59,"num_return_sequences":8},{"name":"E","temperature":.65,"top_p":.95,"seed":71,"num_return_sequences":8}],
 "source_selection":"calibration_hard_groups_to_train_scaffold_morgan_nearest_v1","test_loaded":False,"run_complete":True,
},open(sys.argv[1],"w"),indent=2,sort_keys=True)
PY
echo '[BACE_OURS_CONNECTED_R2_V4_SUCCESS]'
