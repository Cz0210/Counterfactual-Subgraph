#!/usr/bin/env bash
# Resource directives copied from the successful AIDS/MUT WNode plotting wrapper.
#SBATCH --job-name=aids_mut_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "[ERROR] PROJECT_ROOT or SLURM_SUBMIT_DIR is required." >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

set +u
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate smiles_pip118
set -u

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
mkdir -p logs

INPUT_DIR="${INPUT_DIR:-outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_matched_aids_v2_copy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_matched_aids_v3}"
FIGURE3_COVERAGE_YMAX="${FIGURE3_COVERAGE_YMAX:-90}"

echo "===== AIDS + MUTAGENICITY CSV-REPLAY V3 ====="
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "pwd=$PWD"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_commit=$(git rev-parse HEAD)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python=$(command -v python)"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "FIGURE3_COVERAGE_YMAX=$FIGURE3_COVERAGE_YMAX"
python --version
nvidia-smi || true

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] Frozen CSV input directory does not exist: $INPUT_DIR" >&2
  exit 3
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[ERROR] Refusing to overwrite output directory: $OUTPUT_DIR" >&2
  exit 3
fi
for filename in \
  figure3_gcf_style_aids_mut_data.csv \
  figure4_gcf_style_aids_mut_data.csv \
  table2_gcf_style_aids_mut.csv; do
  if [[ ! -s "$INPUT_DIR/$filename" ]]; then
    echo "[ERROR] Missing frozen source CSV: $INPUT_DIR/$filename" >&2
    exit 3
  fi
done

python -m py_compile \
  scripts/paper/plot_aids_mut_gcf_style.py \
  scripts/paper/replot_aids_mut_gcf_style_from_csv.py

python scripts/paper/replot_aids_mut_gcf_style_from_csv.py \
  --project-root "$PROJECT_ROOT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --figure3-coverage-ymax "$FIGURE3_COVERAGE_YMAX"

expected_outputs=(
  figure3_gcf_style_aids_mut.png
  figure3_gcf_style_aids_mut.pdf
  figure3_gcf_style_aids_mut_data.csv
  figure4_gcf_style_aids_mut.png
  figure4_gcf_style_aids_mut.pdf
  figure4_gcf_style_aids_mut_data.csv
  table2_gcf_style_aids_mut.csv
  table2_gcf_style_aids_mut.md
  table2_gcf_style_aids_mut.png
  table2_gcf_style_aids_mut.pdf
  combined_audit_report.txt
  combined_manifest.json
  _RUN_COMPLETE.json
)
for filename in "${expected_outputs[@]}"; do
  if [[ ! -s "$OUTPUT_DIR/$filename" ]]; then
    echo "[ERROR] Missing V3 output: $OUTPUT_DIR/$filename" >&2
    exit 4
  fi
done

grep -Fq '[AIDS_MUT_WNODE_GCF_STYLE_CSV_REPLAY_V3_OK]' \
  "$OUTPUT_DIR/combined_audit_report.txt"

echo "[AIDS_MUT_WNODE_GCF_STYLE_V3_SUCCESS]"
