#!/bin/bash
#SBATCH --job-name=bace_globalgce_minfreq2_stream_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
OUTPUT_ROOT=${OUTPUT_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v7}
CASE_ROOT=$OUTPUT_ROOT/native/min_freq_2
PERSISTENT_SCRATCH_ROOT=${PERSISTENT_SCRATCH_ROOT:-/share/project/p20526/u20526/counterfactual-subgraph}
POOL_DIR=${POOL_DIR:-$PERSISTENT_SCRATCH_ROOT/globalgce_bace_v7/min_freq_2/pool}
TRAIN_CSV=${TRAIN_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/train_source_label1_teacher_correct.csv}
NATIVE_TRAIN_CSV=${NATIVE_TRAIN_CSV:-$ARTIFACT_ROOT/data/processed/BACE/train.csv}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
OFFICIAL_ROOT=${OFFICIAL_ROOT:-$ARTIFACT_ROOT/baselines/globalgce_official}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$ARTIFACT_ROOT/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/thresholds.json}
for path in "$TRAIN_CSV" "$NATIVE_TRAIN_CSV" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$THRESHOLDS_JSON"; do test -s "$path" || { echo "missing $path" >&2; exit 2; }; done
test -d "$OFFICIAL_ROOT"
echo "hostname=$(hostname) commit=$(git rev-parse HEAD) min_freq=2 gpus=0"
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/globalgce/build_bace_train_pool.py --help >/dev/null
  echo '[BACE_GLOBALGCE_MINFREQ2_STREAM_V7_VALIDATE_OK]'; exit 0
fi
mkdir -p "$CASE_ROOT" "$(dirname "$POOL_DIR")"
export GLOBALGCE_HEARTBEAT_SECONDS=300
python scripts/baselines/globalgce/build_bace_train_pool.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --train-csv "$TRAIN_CSV" --native-train-csv "$NATIVE_TRAIN_CSV" \
  --teacher-path "$TEACHER_PATH" --official-root "$OFFICIAL_ROOT" \
  --output-dir "$POOL_DIR" --expected-parent-count 360 --seed 13 --epochs 100 \
  --top-k-native 20 --learning-rate 0.1 --dropout 0.5 --device cpu \
  --generation-chunk-size 16 --memory-log-every-chunks 1 --min-freq 2 --resume \
  --gspan-flush-every 128 --gspan-max-in-memory-candidates 128
python scripts/baselines/globalgce/freeze_bace_frequency_top20.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --run-dir "$POOL_DIR" --teacher-path "$TEACHER_PATH" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" --thresholds-json "$THRESHOLDS_JSON" \
  --output-dir "$CASE_ROOT/selector" --target-k 20
python - "$CASE_ROOT/run_manifest.json" "$POOL_DIR" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); pool=Path(sys.argv[2])
out.write_text(json.dumps({"pool_path":str(pool),"min_freq":2,"streaming":True,"gpus":0,"complete":True},indent=2)+"\n")
PY
