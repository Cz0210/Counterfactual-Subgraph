#!/bin/bash
#SBATCH --job-name=bace_ours_pool
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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
mkdir -p logs

DATASET_PATH=${DATASET_PATH:-outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-pretrained_models/ChemLLM-7B-Chat}
SFT_LORA_PATH=${SFT_LORA_PATH:-outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500}
PPO_CHECKPOINT_PATH=${PPO_CHECKPOINT_PATH:-outputs/hpc/rl_checkpoints/decoded_chem_ppo_stable300_unified_sftv3_projcf_dist03_projpen1_label01_ckpt500}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/bace/bace_teacher.pkl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/hpc/candidate_pools/bace_ours}
RESUME=${RESUME:-false}
RECOVER_GENERATION=${RECOVER_GENERATION:-false}
SOURCE_GENERATION_JOB_ID=${SOURCE_GENERATION_JOB_ID:-}
EXPECTED_RAW_POOL_SHA256=${EXPECTED_RAW_POOL_SHA256:-}

NUM_RETURN_SEQUENCES=4
GEN_TEMPERATURE=0.5
GEN_TOP_P=0.8
MAX_NEW_TOKENS=96
SEED=13
RAW_POOL=$OUTPUT_DIR/candidate_pool.raw.jsonl
POOL=$OUTPUT_DIR/candidate_pool.jsonl
SUMMARY=$OUTPUT_DIR/generation_summary.json
LINEAGE_MANIFEST=$OUTPUT_DIR/candidate_lineage_manifest.json
RUN_MANIFEST=$OUTPUT_DIR/run_manifest.json
COMPLETE_MARKER=$OUTPUT_DIR/_RUN_COMPLETE.json

if [ "$RESUME" = "true" ] && [ -s "$COMPLETE_MARKER" ]; then
  echo "[BACE_OURS_CANDIDATE_POOL_ADOPT_EXISTING] output_dir=$OUTPUT_DIR"
  exit 0
fi
if [ "$RECOVER_GENERATION" != "true" ] && \
   [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]; then
  echo "[BACE_CONFIG_ERROR] candidate output is non-empty: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

if [ "$RECOVER_GENERATION" = "true" ]; then
  if [ -z "$SOURCE_GENERATION_JOB_ID" ]; then
    echo "[BACE_CONFIG_ERROR] recovery requires SOURCE_GENERATION_JOB_ID" >&2
    exit 2
  fi
  if ! [[ "$EXPECTED_RAW_POOL_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[BACE_CONFIG_ERROR] recovery requires a lowercase SHA256" >&2
    exit 2
  fi
  for path in "$RAW_POOL" "$SUMMARY"; do
    if [ ! -s "$path" ]; then
      echo "[BACE_CONFIG_ERROR] missing completed generation artifact: $path" >&2
      exit 2
    fi
  done
  for path in "$POOL" "$LINEAGE_MANIFEST" "$RUN_MANIFEST" "$COMPLETE_MARKER"; do
    if [ -e "$path" ]; then
      echo "[BACE_CONFIG_ERROR] recovery output already exists: $path" >&2
      exit 2
    fi
  done
  UNEXPECTED_OUTPUT=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
    ! -name "$(basename "$RAW_POOL")" \
    ! -name "$(basename "$SUMMARY")" -print -quit)
  if [ -n "$UNEXPECTED_OUTPUT" ]; then
    echo "[BACE_CONFIG_ERROR] unexpected recovery input: $UNEXPECTED_OUTPUT" >&2
    exit 2
  fi
fi

for path in "$DATASET_PATH" "$BASE_MODEL_PATH" "$SFT_LORA_PATH" "$PPO_CHECKPOINT_PATH" "$TEACHER_PATH"; do
  if [ ! -e "$path" ]; then
    echo "[BACE_CONFIG_ERROR] missing candidate-generation input: $path" >&2
    exit 2
  fi
done

PARENT_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "$DATASET_PATH")
if [ "$PARENT_COUNT" -le 0 ]; then
  echo "[BACE_CONFIG_ERROR] empty BACE train-source generation cohort" >&2
  exit 2
fi
EXPECTED_ROWS=$((PARENT_COUNT * NUM_RETURN_SEQUENCES))

echo "[BACE_OURS_CANDIDATE_POOL_CONFIG]"
echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "pwd=$(pwd)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
nvidia-smi || true
echo "dataset_path=$DATASET_PATH"
echo "teacher_path=$TEACHER_PATH"
echo "ppo_checkpoint_path=$PPO_CHECKPOINT_PATH"
echo "parent_count=$PARENT_COUNT"
echo "expected_rows=$EXPECTED_ROWS"
echo "output_dir=$OUTPUT_DIR"
echo "recover_generation=$RECOVER_GENERATION"
echo "source_generation_job_id=${SOURCE_GENERATION_JOB_ID:-none}"
echo "calibration_loaded=false"
echo "test_loaded=false"

if [ "$RECOVER_GENERATION" = "true" ]; then
  ACTUAL_RAW_POOL_SHA256=$(sha256sum "$RAW_POOL" | awk '{print $1}')
  if [ "$ACTUAL_RAW_POOL_SHA256" != "$EXPECTED_RAW_POOL_SHA256" ]; then
    echo "[BACE_CONFIG_ERROR] recovered raw candidate SHA256 mismatch" >&2
    exit 2
  fi
  if [ "$(wc -l < "$RAW_POOL")" -ne "$EXPECTED_ROWS" ]; then
    echo "[BACE_CONFIG_ERROR] recovered raw candidate row count mismatch" >&2
    exit 2
  fi
  echo "[BACE_OURS_GENERATION_ADOPT_EXISTING] algorithm_rerun=false"
else
  python scripts/generate_full_candidate_pool.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --dataset-path "$DATASET_PATH" \
    --base-model-path "$BASE_MODEL_PATH" \
    --sft-lora-path "$SFT_LORA_PATH" \
    --ppo-checkpoint-path "$PPO_CHECKPOINT_PATH" \
    --teacher-path "$TEACHER_PATH" \
    --out-jsonl "$RAW_POOL" \
    --out-summary-json "$SUMMARY" \
    --label-col label \
    --smiles-col smiles \
    --target-label 1 \
    --num-return-sequences "$NUM_RETURN_SEQUENCES" \
    --generation-temperature "$GEN_TEMPERATURE" \
    --generation-top-p "$GEN_TOP_P" \
    --generation-do-sample true \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size 1 \
    --seed "$SEED" \
    --enable-parent-projection \
    --enable-projected-cf-reward \
    --enable-substructure-distance-reward \
    --substructure-distance-reward-weight 0.3 \
    --projection-penalty 1.0 \
    --enable-minimal-syntax-repair \
    --enable-component-salvage
fi

python scripts/baselines/bace/enrich_ours_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --raw-pool-jsonl "$RAW_POOL" \
  --parent-csv "$DATASET_PATH" \
  --output-jsonl "$POOL" \
  --manifest-path "$LINEAGE_MANIFEST" \
  --expected-candidates-per-parent "$NUM_RETURN_SEQUENCES"

export DATASET_PATH BASE_MODEL_PATH SFT_LORA_PATH PPO_CHECKPOINT_PATH
export TEACHER_PATH OUTPUT_DIR POOL SUMMARY LINEAGE_MANIFEST RUN_MANIFEST
export PARENT_COUNT EXPECTED_ROWS NUM_RETURN_SEQUENCES GEN_TEMPERATURE GEN_TOP_P
export MAX_NEW_TOKENS SEED
export RECOVER_GENERATION SOURCE_GENERATION_JOB_ID EXPECTED_RAW_POOL_SHA256
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def identity(value: str) -> dict[str, object]:
    path = Path(value).resolve()
    result: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.update({"bytes": path.stat().st_size, "sha256": digest})
    return result

manifest = {
    "schema_version": "bace_ours_candidate_pool_v1",
    "status": "complete",
    "dataset": "BACE",
    "method": "Ours",
    "git_commit": os.popen("git rev-parse HEAD").read().strip(),
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "inputs": {
        key: identity(os.environ[key])
        for key in (
            "DATASET_PATH", "BASE_MODEL_PATH", "SFT_LORA_PATH",
            "PPO_CHECKPOINT_PATH", "TEACHER_PATH"
        )
    },
    "sampling": {
        "num_return_sequences": int(os.environ["NUM_RETURN_SEQUENCES"]),
        "generation_temperature": float(os.environ["GEN_TEMPERATURE"]),
        "generation_top_p": float(os.environ["GEN_TOP_P"]),
        "generation_do_sample": True,
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
        "seed": int(os.environ["SEED"]),
    },
    "algorithm": {
        "checkpoint_family": "stable300",
        "parent_projection": True,
        "projected_cf_reward": True,
        "substructure_distance_reward": True,
        "substructure_distance_reward_weight": 0.3,
        "projection_penalty": 1.0,
    },
    "parent_count": int(os.environ["PARENT_COUNT"]),
    "candidate_count": int(os.environ["EXPECTED_ROWS"]),
    "generation_recovery": {
        "adopted_existing": os.environ["RECOVER_GENERATION"] == "true",
        "algorithm_rerun": os.environ["RECOVER_GENERATION"] != "true",
        "source_generation_job_id": os.environ["SOURCE_GENERATION_JOB_ID"] or None,
        "expected_raw_pool_sha256": os.environ["EXPECTED_RAW_POOL_SHA256"] or None,
    },
    "candidate_pool": identity(os.environ["POOL"]),
    "generation_summary": identity(os.environ["SUMMARY"]),
    "lineage_manifest": identity(os.environ["LINEAGE_MANIFEST"]),
    "calibration_loaded": False,
    "test_loaded": False,
}
Path(os.environ["RUN_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

test "$(wc -l < "$POOL")" -eq "$EXPECTED_ROWS"
printf '{"status":"complete","stage":"bace_ours_candidate_pool"}\n' > "$COMPLETE_MARKER"
echo "[BACE_OURS_CANDIDATE_POOL_SUCCESS] parents=$PARENT_COUNT candidates=$EXPECTED_ROWS"
