#!/bin/bash
#SBATCH --job-name=bace_globalgce_minfreq_select_v4
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
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v4_retry2}
TEACHER_PATH=${TEACHER_PATH:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-$ARTIFACT_ROOT/outputs/hpc/eval/paper/bace_common3_connected_residual_v3_expanded/thresholds.json}
METRICS_CSV=$OUTPUT_DIR/min_freq_calibration.csv
echo "hostname=$(hostname) git_commit=$(git rev-parse HEAD)"
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then python scripts/baselines/globalgce/select_bace_min_freq.py --help >/dev/null; echo '[BACE_GLOBALGCE_MIN_FREQ_SELECT_VALIDATE_OK]'; exit 0; fi
python - "$OUTPUT_DIR" "$METRICS_CSV" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); output=Path(sys.argv[2]); rows=[]
for value in (2,4,7,18):
 p=root/f"candidates/min_freq_{value}/calibration_metrics.json"
 if not p.is_file(): raise FileNotFoundError(p)
 rows.append(json.loads(p.read_text()))
fields=list(rows[0]); output.parent.mkdir(parents=True,exist_ok=True)
with output.open("x",newline="",encoding="utf-8") as handle:
 writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
PY
python scripts/baselines/globalgce/select_bace_min_freq.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --metrics-csv "$METRICS_CSV" --output-dir "$OUTPUT_DIR" --source-train-parent-count 360 --teacher-path "$TEACHER_PATH" --thresholds-json "$THRESHOLDS_JSON" --git-commit "$(git rev-parse HEAD)"
echo '[BACE_GLOBALGCE_MIN_FREQ_CALIBRATION_SUCCESS]'
