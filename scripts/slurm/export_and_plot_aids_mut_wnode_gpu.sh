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

AIDS_FIGURE3_CSV="${AIDS_FIGURE3_CSV:-outputs/hpc/eval/paper/molclr_node_wasserstein_figure3_theta005_raw/wnode_fig3_theta005_figure3_wnode_coverage_cost_vs_k.csv}"
AIDS_FIGURE4_CSV="${AIDS_FIGURE4_CSV:-outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10/wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv}"
AIDS_OURS_ROOT="${AIDS_OURS_ROOT:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_ours_top20_final}"
AIDS_GLOBALGCE_ROOT="${AIDS_GLOBALGCE_ROOT:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_globalgce_frequency_top20_final}"
AIDS_CLEAR_ROOT="${AIDS_CLEAR_ROOT:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_clear_parent_frequency_top20_final}"
AIDS_GCF_ROOT="${AIDS_GCF_ROOT:-outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_gcfexplainer_top20_normalized_final}"
MUT_OURS_ROOT="${MUT_OURS_ROOT:-outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1}"
MUT_GLOBALGCE_ROOT="${MUT_GLOBALGCE_ROOT:-outputs/hpc/mutagenicity/final/globalgce_wnode_frequency_top20_test_v1}"
MUT_CLEAR_ROOT="${MUT_CLEAR_ROOT:-outputs/hpc/mutagenicity/final/clear_wnode_parent_frequency_test_v1}"
MUT_GCF_ROOT="${MUT_GCF_ROOT:-outputs/hpc/mutagenicity/final/gcfexplainer_native5000_top20_wnode_test_v1}"
MUT_THRESHOLD_MODE="${MUT_THRESHOLD_MODE:-match-aids}"
case "$MUT_THRESHOLD_MODE" in
  match-aids)
    DEFAULT_OUTPUT_ROOT="outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_matched_aids_v1"
    ;;
  native)
    DEFAULT_OUTPUT_ROOT="outputs/hpc/eval/paper/aids_mutagenicity_wnode_gcf_style_v2"
    ;;
  *)
    echo "[ERROR] MUT_THRESHOLD_MODE must be native or match-aids." >&2
    exit 2
    ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"

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
echo "distance_line=MolCLR-Node-Wasserstein"
echo "distance_type=node_wasserstein"
echo "cf_mode=strict_flip"
echo "AIDS_FIGURE3_CSV=$AIDS_FIGURE3_CSV"
echo "AIDS_FIGURE4_CSV=$AIDS_FIGURE4_CSV"
echo "AIDS_OURS_ROOT=$AIDS_OURS_ROOT"
echo "AIDS_GLOBALGCE_ROOT=$AIDS_GLOBALGCE_ROOT"
echo "AIDS_CLEAR_ROOT=$AIDS_CLEAR_ROOT"
echo "AIDS_GCF_ROOT=$AIDS_GCF_ROOT"
echo "MUT_OURS_ROOT=$MUT_OURS_ROOT"
echo "MUT_GLOBALGCE_ROOT=$MUT_GLOBALGCE_ROOT"
echo "MUT_CLEAR_ROOT=$MUT_CLEAR_ROOT"
echo "MUT_GCF_ROOT=$MUT_GCF_ROOT"
echo "MUT_THRESHOLD_MODE=$MUT_THRESHOLD_MODE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
python --version
nvidia-smi || true

for source_csv in "$AIDS_FIGURE3_CSV" "$AIDS_FIGURE4_CSV"; do
  if [[ ! -s "$source_csv" ]]; then
    echo "[ERROR] Missing frozen AIDS plotting CSV: $source_csv" >&2
    exit 3
  fi
done

for root in \
  "$AIDS_OURS_ROOT" \
  "$AIDS_GLOBALGCE_ROOT" \
  "$AIDS_CLEAR_ROOT" \
  "$AIDS_GCF_ROOT"; do
  if [[ ! -s "$root/details/pair_details.csv" ]]; then
    echo "[ERROR] Missing frozen AIDS WNode pair details: $root/details/pair_details.csv" >&2
    exit 3
  fi
done

if [[ "$MUT_THRESHOLD_MODE" == "native" ]]; then
  declare -A TABLE_FILES=(
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
else
  mut_pair_artifacts=(
    "$MUT_OURS_ROOT/pair_matrix.jsonl"
    "$MUT_OURS_ROOT/selected_sequence.jsonl"
    "$MUT_GLOBALGCE_ROOT/test_pair_details.csv"
    "$MUT_GLOBALGCE_ROOT/selected_top20.csv"
    "$MUT_CLEAR_ROOT/test/k20_pair_details.csv"
    "$MUT_CLEAR_ROOT/selected_candidates.csv"
    "$MUT_GCF_ROOT/test_pair_details.csv"
    "$MUT_GCF_ROOT/selected_sequence.jsonl"
  )
  for artifact in "${mut_pair_artifacts[@]}"; do
    if [[ ! -s "$artifact" ]]; then
      echo "[ERROR] Missing frozen MUT pair/order artifact: $artifact" >&2
      exit 3
    fi
  done
fi

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
  --aids-figure3-csv "$AIDS_FIGURE3_CSV" \
  --aids-figure4-csv "$AIDS_FIGURE4_CSV" \
  --aids-ours-root "$AIDS_OURS_ROOT" \
  --aids-globalgce-root "$AIDS_GLOBALGCE_ROOT" \
  --aids-clear-root "$AIDS_CLEAR_ROOT" \
  --aids-gcf-root "$AIDS_GCF_ROOT" \
  --mut-ours-root "$MUT_OURS_ROOT" \
  --mut-globalgce-root "$MUT_GLOBALGCE_ROOT" \
  --mut-clear-root "$MUT_CLEAR_ROOT" \
  --mut-gcf-root "$MUT_GCF_ROOT" \
  --mut-threshold-mode "$MUT_THRESHOLD_MODE" \
  --output-dir "$OUTPUT_ROOT"

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
  if [[ ! -s "$OUTPUT_ROOT/$filename" ]]; then
    echo "[ERROR] Missing combined plotting output: $OUTPUT_ROOT/$filename" >&2
    exit 4
  fi
done

grep -Fq '[AIDS_MUT_WNODE_GCF_STYLE_V2_OK]' "$OUTPUT_ROOT/combined_audit_report.txt"

python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "combined_manifest.json").read_text(encoding="utf-8"))
complete = json.loads((root / "_RUN_COMPLETE.json").read_text(encoding="utf-8"))
expected = {
    "distance_line": "MolCLR-Node-Wasserstein",
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

if [[ "$MUT_THRESHOLD_MODE" == "match-aids" ]]; then
  echo "[AIDS_MUT_WNODE_GCF_STYLE_MATCHED_AIDS_SUCCESS]"
else
  echo "[AIDS_MUT_WNODE_GCF_STYLE_V2_SUCCESS]"
fi
