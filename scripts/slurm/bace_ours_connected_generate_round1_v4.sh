#!/bin/bash
#SBATCH --job-name=bace_ours_conn_gen_r1_v4
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

UNION_AUDIT=${UNION_AUDIT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_ours_connected_candidateaware_v4/calibration_matrix/candidate_union_summary.json}
BASE_MANIFEST=${BASE_MANIFEST:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours/run_manifest.json}
DATASET_PATH=${DATASET_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_connected_candidateaware_v4/round1}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

for path in "$UNION_AUDIT" "$BASE_MANIFEST" "$DATASET_PATH" "$TEACHER_PATH"; do
  test -s "$path" || { echo "[BACE_CONNECTED_R1_CONFIG_ERROR] missing $path" >&2; exit 2; }
done
readarray -t MODEL_PATHS < <(python - "$BASE_MANIFEST" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))["inputs"]
for key in ("BASE_MODEL_PATH", "SFT_LORA_PATH", "PPO_CHECKPOINT_PATH"):
    print(p[key]["path"])
PY
)
BASE_MODEL_PATH=${MODEL_PATHS[0]}
SFT_LORA_PATH=${MODEL_PATHS[1]}
PPO_CHECKPOINT_PATH=${MODEL_PATHS[2]}
for path in "$BASE_MODEL_PATH" "$SFT_LORA_PATH" "$PPO_CHECKPOINT_PATH"; do
  test -e "$path" || { echo "[BACE_CONNECTED_R1_CONFIG_ERROR] missing $path" >&2; exit 2; }
done

EXPANSION_REQUIRED=$(python - "$UNION_AUDIT" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["split"] == "calibration" and p["test_loaded"] is False
required = (
    float(p["CONNECTED_STRICT_FLIP_UNION_COVERAGE"]) < 0.40
    or float(p["CLOSE_UNION_AT_PRIMARY_THETA"]) < 0.25
)
print("true" if required else "false")
PY
)

echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
echo "candidate_expansion_required=$EXPANSION_REQUIRED"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_OURS_CONNECTED_R1_V4_VALIDATE_OK]"
  exit 0
fi
test ! -e "$OUTPUT_ROOT" || { echo "[BACE_CONNECTED_R1_COLLISION] $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
if [ "$EXPANSION_REQUIRED" != true ]; then
  python - "$OUTPUT_ROOT/_SKIPPED.json" "$UNION_AUDIT" <<'PY'
import json, sys
json.dump({
    "run_complete": True,
    "round1_run": False,
    "reason": "calibration_candidate_pool_not_limited",
    "union_audit": sys.argv[2],
    "test_loaded": False,
}, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
  echo "[BACE_OURS_CONNECTED_R1_V4_NOT_REQUIRED]"
  exit 0
fi

temperatures=(0.25 0.50 0.75)
top_ps=(0.90 0.90 0.95)
seeds=(17 29 43)
regimes=(A B C)
for index in 0 1 2; do
  regime=${regimes[$index]}
  out="$OUTPUT_ROOT/regime${regime}"
  mkdir -p "$out"
  python scripts/generate_full_candidate_pool.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --dataset-path "$DATASET_PATH" --base-model-path "$BASE_MODEL_PATH" \
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
    --raw-pool-jsonl "$out/candidate_pool.raw.jsonl" --parent-csv "$DATASET_PATH" \
    --output-jsonl "$out/candidate_pool.enriched.jsonl" \
    --manifest-path "$out/candidate_lineage_manifest.json" \
    --expected-candidates-per-parent 8
  python scripts/filter_bace_connected_source_candidates.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --input-jsonl "$out/candidate_pool.enriched.jsonl" --parent-csv "$DATASET_PATH" \
    --output-jsonl "$out/candidate_pool.jsonl" --audit-json "$out/source_gate_audit.json" \
    --generation-round 1 --generation-regime "$regime" \
    --prompt-mode connected_deletion_v1
done
python - "$OUTPUT_ROOT/expansion_protocol.json" "$UNION_AUDIT" <<'PY'
import json, sys
payload = {
    "schema_version": "bace_connected_candidateaware_generation_v4",
    "round": 1,
    "round1_run": True,
    "regimes": [
        {"name": "A", "temperature": 0.25, "top_p": 0.90, "seed": 17, "num_return_sequences": 8},
        {"name": "B", "temperature": 0.50, "top_p": 0.90, "seed": 29, "num_return_sequences": 8},
        {"name": "C", "temperature": 0.75, "top_p": 0.95, "seed": 43, "num_return_sequences": 8},
    ],
    "connected_prompt_mode": "connected_deletion_v1",
    "source_effect_fields_are_features_not_gates": True,
    "source_cohort": "frozen_train_label1_teacher_correct",
    "source_parent_budget_balanced": True,
    "molclr_cluster_file_available": False,
    "test_loaded": False,
    "union_audit": sys.argv[2],
    "run_complete": True,
}
json.dump(payload, open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY
echo '[BACE_OURS_CONNECTED_R1_V4_SUCCESS]'
