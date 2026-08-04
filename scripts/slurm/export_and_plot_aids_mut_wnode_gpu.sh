#!/usr/bin/env bash
# Resource directives copied from scripts/slurm/plot_fgw_sota_figures_gpu.sh.
# The verified template has no account, qos, or constraint directives.
#SBATCH --job-name=aids_mut_wnode
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

AIDS_OURS_ROOT="${AIDS_OURS_ROOT:-outputs/hpc/eval/paper/aids_common3_standardized_v2/ours}"
AIDS_GLOBALGCE_ROOT="${AIDS_GLOBALGCE_ROOT:-outputs/hpc/eval/paper/aids_common3_standardized_v2/globalgce}"
AIDS_CLEAR_ROOT="${AIDS_CLEAR_ROOT:-outputs/hpc/eval/paper/aids_common3_standardized_v2/clear}"
AIDS_GCF_ROOT="${AIDS_GCF_ROOT:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_gcfexplainer_top20_normalized_final}"
MUT_OURS_ROOT="${MUT_OURS_ROOT:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1}"
MUT_GLOBALGCE_ROOT="${MUT_GLOBALGCE_ROOT:-outputs/hpc/mutagenicity/final/globalgce_wnode_frequency_top20_test_v1}"
MUT_CLEAR_ROOT="${MUT_CLEAR_ROOT:-outputs/hpc/mutagenicity/final/clear_wnode_parent_frequency_test_v1}"
MUT_GCF_ROOT="${MUT_GCF_ROOT:-outputs/hpc/mutagenicity/final/gcfexplainer_native5000_top20_wnode_test_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_v1}"

echo "===== AIDS + MUTAGENICITY FOUR-METHOD WNODE PLOT ====="
echo "hostname=$(hostname)"
echo "date=$(date -Iseconds)"
echo "pwd=$PWD"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_commit=$(git rev-parse HEAD)"
echo "conda_env=${CONDA_DEFAULT_ENV:-unset}"
echo "python=$(command -v python)"
echo "distance_label=MolCLR-Node-Wasserstein"
echo "distance_type=node_wasserstein"
echo "cf_mode=strict_flip"
echo "AIDS_OURS_ROOT=$AIDS_OURS_ROOT"
echo "AIDS_GLOBALGCE_ROOT=$AIDS_GLOBALGCE_ROOT"
echo "AIDS_CLEAR_ROOT=$AIDS_CLEAR_ROOT"
echo "AIDS_GCF_ROOT=$AIDS_GCF_ROOT"
echo "MUT_OURS_ROOT=$MUT_OURS_ROOT"
echo "MUT_GLOBALGCE_ROOT=$MUT_GLOBALGCE_ROOT"
echo "MUT_CLEAR_ROOT=$MUT_CLEAR_ROOT"
echo "MUT_GCF_ROOT=$MUT_GCF_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
python --version
nvidia-smi || true

declare -A TABLE_FILES=(
  ["$AIDS_OURS_ROOT"]="table2_ours_k10.csv"
  ["$AIDS_GLOBALGCE_ROOT"]="table2_globalgce_k10.csv"
  ["$AIDS_CLEAR_ROOT"]="table2_clear_k10.csv"
  ["$MUT_OURS_ROOT"]="table2_ours_k10.csv"
  ["$MUT_GLOBALGCE_ROOT"]="table2_globalgce_k10.csv"
  ["$MUT_CLEAR_ROOT"]="table2_clear_k10.csv"
  ["$MUT_GCF_ROOT"]="table2_gcfexplainer_k10.csv"
)
for root in "${!TABLE_FILES[@]}"; do
  for filename in \
    figure3_coverage_vs_k.csv \
    figure4_coverage_vs_threshold.csv \
    "${TABLE_FILES[$root]}"; do
    if [[ ! -s "$root/$filename" ]]; then
      echo "[ERROR] Missing frozen WNode plotting artifact: $root/$filename" >&2
      exit 3
    fi
  done
done

for filename in \
  run_config.json \
  cache_stats.json \
  details/pair_details.csv \
  combined/combined_threshold_summary.csv \
  _RUN_COMPLETE.json; do
  if [[ ! -s "$AIDS_GCF_ROOT/$filename" ]]; then
    echo "[ERROR] Missing frozen AIDS GCFExplainer WNode run artifact: $AIDS_GCF_ROOT/$filename" >&2
    exit 3
  fi
done

case "$AIDS_OURS_ROOT $AIDS_GLOBALGCE_ROOT $AIDS_CLEAR_ROOT $AIDS_GCF_ROOT" in
  *ccrcov_molclr_node_fgw_*|*node_fgw*|*lam05*|*gt_fullgraph*|*opposite_fullgraph*|*opposite-label*)
    echo "[ERROR] AIDS roots include a forbidden non-WNode or legacy result path." >&2
    exit 3
    ;;
esac

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "[ERROR] Refusing to overwrite existing output root: $OUTPUT_ROOT" >&2
  exit 3
fi

python -m py_compile scripts/paper/plot_aids_mut_gcf_style.py

python scripts/paper/plot_aids_mut_gcf_style.py \
  --project-root "$PROJECT_ROOT" \
  --aids-ours-root "$AIDS_OURS_ROOT" \
  --aids-globalgce-root "$AIDS_GLOBALGCE_ROOT" \
  --aids-clear-root "$AIDS_CLEAR_ROOT" \
  --aids-gcf-root "$AIDS_GCF_ROOT" \
  --mut-ours-root "$MUT_OURS_ROOT" \
  --mut-globalgce-root "$MUT_GLOBALGCE_ROOT" \
  --mut-clear-root "$MUT_CLEAR_ROOT" \
  --mut-gcf-root "$MUT_GCF_ROOT" \
  --output-dir "$OUTPUT_ROOT"

expected_outputs=(
  figure3_aids_mut_gcf_style.png
  figure3_aids_mut_gcf_style.pdf
  figure3_aids_mut_source.csv
  figure4_aids_mut_gcf_style.png
  figure4_aids_mut_gcf_style.pdf
  figure4_aids_mut_source.csv
  table2_aids_mut_gcf_style.csv
  table2_aids_mut_gcf_style.md
  table2_aids_mut_gcf_style.png
  table2_aids_mut_gcf_style.pdf
  table2_aids_mut_full.csv
  combined_audit_report.txt
  combined_manifest.json
  _RUN_COMPLETE.json
)
for filename in "${expected_outputs[@]}"; do
  if [[ ! -s "$OUTPUT_ROOT/$filename" ]]; then
    echo "[ERROR] Missing combined plotting output: $OUTPUT_ROOT/$filename" >&2
    exit 4
  fi
done

grep -Fq '[AIDS_MUT_GCF_STYLE_PLOT_OK]' "$OUTPUT_ROOT/combined_audit_report.txt"

python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "combined_manifest.json").read_text(encoding="utf-8"))
complete = json.loads((root / "_RUN_COMPLETE.json").read_text(encoding="utf-8"))
expected = {
    "distance_label": "MolCLR-Node-Wasserstein",
    "distance_type": "node_wasserstein",
    "cf_mode": "strict_flip",
}
for field, value in expected.items():
    if manifest.get(field) != value or complete.get(field) != value:
        raise SystemExit(f"[ERROR] Combined WNode provenance mismatch: {field}")
if complete.get("run_complete") is not True:
    raise SystemExit("[ERROR] Combined completion marker is false.")
if manifest.get("distance_recomputed") is not False:
    raise SystemExit("[ERROR] Plot manifest claims distance recomputation.")
print("[AIDS_MUT_WNODE_MANIFEST_AUDIT_OK]")
PY

echo "[AIDS_MUT_WNODE_PLOT_SUCCESS]"
