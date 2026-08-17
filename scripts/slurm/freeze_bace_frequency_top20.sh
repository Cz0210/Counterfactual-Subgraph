#!/bin/bash
#SBATCH --job-name=bace_globalgce_top20
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

RUN_DIR=${RUN_DIR:-$ARTIFACT_ROOT/outputs/hpc/baselines/globalgce/bace/train_pool_connected_v1}
MIN_FREQ_MANIFEST=${MIN_FREQ_MANIFEST:-}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$ARTIFACT_ROOT/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_connected_candidateaware_v4/threshold_protocol/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_globalgce_frequency_top20_connected_v1}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

if [[ -n "$MIN_FREQ_MANIFEST" ]]; then
  test -s "$MIN_FREQ_MANIFEST" || { echo "missing min-freq manifest: $MIN_FREQ_MANIFEST" >&2; exit 2; }
  RUN_DIR=$(python - "$MIN_FREQ_MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
assert p["dataset"] == "BACE"
assert p["selection_split"] == "calibration"
assert p["test_loaded"] is False
print(p["selected_pool_path"])
PY
)
fi

for path in "$RUN_DIR/_RUN_COMPLETE.json" "$RUN_DIR/candidate_universe.jsonl" "$TEACHER_PATH" "$MOLCLR_CHECKPOINT" "$THRESHOLDS_JSON"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done

args=(
  python scripts/baselines/globalgce/freeze_bace_frequency_top20.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --run-dir "$RUN_DIR"
  --teacher-path "$TEACHER_PATH"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --thresholds-json "$THRESHOLDS_JSON"
  --output-dir "$OUTPUT_DIR"
  --target-k 20
)
echo "hostname=$(hostname)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "git_commit=$(git rev-parse HEAD)"
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  echo '[BACE_GLOBALGCE_TOP20_VALIDATE_OK]'
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
"${args[@]}"
test -s "$OUTPUT_DIR/selected_top20_for_eval.csv"
test -s "$OUTPUT_DIR/selection_audit.json"
echo '[BACE_GLOBALGCE_TOP20_SUCCESS]'
