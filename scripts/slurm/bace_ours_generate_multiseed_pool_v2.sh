#!/bin/bash
#SBATCH --job-name=bace_ours_expand_v2
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

LIMITATION_AUDIT=${LIMITATION_AUDIT:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_wnode_prefix_v2/candidate_pool_limitation_audit.json}
BASE_MANIFEST=${BASE_MANIFEST:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours/run_manifest.json}
DATASET_PATH=${DATASET_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/candidate_pools/bace_ours_multiseed_multitemp_v2/regimes}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
for path in "$LIMITATION_AUDIT" "$BASE_MANIFEST" "$DATASET_PATH" "$TEACHER_PATH"; do test -s "$path"; done
python - "$LIMITATION_AUDIT" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))["candidate_expansion_required"] is not True:
    raise SystemExit("[BACE_V2_CONFIG_ERROR] expansion was not authorized by calibration audit")
PY
readarray -t MODEL_PATHS < <(python - "$BASE_MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))["inputs"]
for key in ("BASE_MODEL_PATH","SFT_LORA_PATH","PPO_CHECKPOINT_PATH"):
    print(p[key]["path"])
PY
)
BASE_MODEL_PATH=${MODEL_PATHS[0]}
SFT_LORA_PATH=${MODEL_PATHS[1]}
PPO_CHECKPOINT_PATH=${MODEL_PATHS[2]}
for path in "$BASE_MODEL_PATH" "$SFT_LORA_PATH" "$PPO_CHECKPOINT_PATH"; do test -e "$path"; done
echo "git_commit=$(git rev-parse HEAD)"
echo "ppo_checkpoint=$PPO_CHECKPOINT_PATH"
echo "test_loaded=false"
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then echo "[BACE_OURS_EXPANSION_V2_VALIDATE_OK]"; exit 0; fi
test ! -e "$OUTPUT_ROOT" || { echo "[BACE_V2_COLLISION] $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
temperatures=(0.30 0.60 0.80)
top_ps=(0.90 0.90 0.95)
seeds=(17 29 43)
for index in 0 1 2; do
  regime=$((index + 1))
  out="$OUTPUT_ROOT/regime${regime}"
  mkdir -p "$out"
  python scripts/generate_full_candidate_pool.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --dataset-path "$DATASET_PATH" --base-model-path "$BASE_MODEL_PATH" \
    --sft-lora-path "$SFT_LORA_PATH" --ppo-checkpoint-path "$PPO_CHECKPOINT_PATH" \
    --teacher-path "$TEACHER_PATH" --out-jsonl "$out/candidate_pool.raw.jsonl" \
    --out-summary-json "$out/generation_summary.json" --label-col label --smiles-col smiles \
    --target-label 1 --num-return-sequences 4 \
    --generation-temperature "${temperatures[$index]}" \
    --generation-top-p "${top_ps[$index]}" --generation-do-sample true \
    --max-new-tokens 96 --batch-size 1 --seed "${seeds[$index]}" \
    --enable-parent-projection --enable-projected-cf-reward \
    --enable-substructure-distance-reward --substructure-distance-reward-weight 0.3 \
    --projection-penalty 1.0 --enable-minimal-syntax-repair --enable-component-salvage
  python scripts/baselines/bace/enrich_ours_candidate_pool.py \
    --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
    --raw-pool-jsonl "$out/candidate_pool.raw.jsonl" --parent-csv "$DATASET_PATH" \
    --output-jsonl "$out/candidate_pool.jsonl" --manifest-path "$out/candidate_lineage_manifest.json" \
    --expected-candidates-per-parent 4
done
echo '{"run_complete":true,"test_loaded":false}' > "$OUTPUT_ROOT/_RUN_COMPLETE.json"
echo "[BACE_OURS_EXPANSION_V2_SUCCESS]"
