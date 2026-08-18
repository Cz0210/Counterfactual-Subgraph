#!/bin/bash
#SBATCH --job-name=bace_globalgce_minfreq_cal_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v7}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common4_connected_candidateaware_v4_storage_v6/thresholds.json}
METRICS_CSV=$OUTPUT_DIR/min_freq_calibration.csv
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/globalgce/select_bace_min_freq.py --help >/dev/null
  echo '[BACE_GLOBALGCE_MIN_FREQ_CAL_V7_VALIDATE_OK]'; exit 0
fi
python - "$OUTPUT_DIR" "$METRICS_CSV" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2]); rows=[]
for value in (2,4,7,18):
 path=root/f"candidates/min_freq_{value}/calibration_metrics.json"
 if not path.is_file(): raise FileNotFoundError(path)
 rows.append(json.loads(path.read_text()))
fields=list(rows[0])
with output.open("x",newline="",encoding="utf-8") as handle:
 writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
PY
python scripts/baselines/globalgce/select_bace_min_freq.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --metrics-csv "$METRICS_CSV" --output-dir "$OUTPUT_DIR" \
  --source-train-parent-count 360 --teacher-path "$TEACHER_PATH" \
  --thresholds-json "$THRESHOLDS_JSON" --git-commit "$(git rev-parse HEAD)"
