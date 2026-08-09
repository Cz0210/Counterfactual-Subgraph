#!/bin/bash
#SBATCH --job-name=bace_delete_audit_v3
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
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

OURS_RUN=${OURS_RUN:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_prefix_v2_work/test_run/test}
OURS_CANDIDATES=${OURS_CANDIDATES:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_ours_wnode_prefix_v2/selected_top20.csv}
ELIGIBLE_PARENT_CSV=${ELIGIBLE_PARENT_CSV:-$ARTIFACT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
GCF_RUN=${GCF_RUN:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_gcfexplainer_wnode_work_retry2_valid_native_rank/test}
GCF_CANDIDATES=${GCF_CANDIDATES:-$ARTIFACT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/summary_retry2_valid_native_rank/export/selected_top20.csv}
GCF_TARGET_LABEL=${GCF_TARGET_LABEL:-0}
OLD_THRESHOLDS=${OLD_THRESHOLDS:-$ARTIFACT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_ours_disconnected_residual_v2}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
AIDS_OURS_RUN=${AIDS_OURS_RUN:-$ARTIFACT_ROOT/outputs/hpc/eval/ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_ours_top20_final}
MUT_OURS_RUN=${MUT_OURS_RUN:-$ARTIFACT_ROOT/outputs/hpc/mutagenicity/final_eval/wnode_frozen_a2_test_p217_k20_v3}

for path in "$OURS_RUN/details/pair_details.csv" "$OURS_CANDIDATES" "$ELIGIBLE_PARENT_CSV" "$GCF_RUN/run_config.json" "$GCF_CANDIDATES" "$OLD_THRESHOLDS"; do
  test -s "$path" || { echo "[BACE_CONNECTED_AUDIT_CONFIG_ERROR] missing $path" >&2; exit 2; }
done

args=(
  python scripts/audit_bace_hard_deletion_semantics.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --ours-run "$OURS_RUN"
  --ours-candidates "$OURS_CANDIDATES"
  --eligible-parent-csv "$ELIGIBLE_PARENT_CSV"
  --gcf-run "$GCF_RUN"
  --gcf-candidates "$GCF_CANDIDATES"
  --gcf-target-label "$GCF_TARGET_LABEL"
  --old-thresholds "$OLD_THRESHOLDS"
  --output-dir "$OUTPUT_DIR"
)
if [ -n "$AIDS_OURS_RUN" ]; then
  test -s "$AIDS_OURS_RUN/details/pair_details.csv" || test -s "$AIDS_OURS_RUN/match_instances.jsonl"
  args+=(--aids-ours-run "$AIDS_OURS_RUN")
fi
if [ -n "$MUT_OURS_RUN" ]; then
  test -s "$MUT_OURS_RUN/details/pair_details.csv" || test -s "$MUT_OURS_RUN/match_instances.jsonl"
  args+=(--mut-ours-run "$MUT_OURS_RUN")
fi
echo "hostname=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
printf 'command='; printf '%q ' "${args[@]}"; printf '\n'
if [ "$DRY_RUN" = 1 ] || [ "$VALIDATE_ONLY" = 1 ]; then
  echo "[BACE_HARD_DELETION_AUDIT_V3_VALIDATE_OK]"
  exit 0
fi
"${args[@]}"
test -s "$OUTPUT_DIR/hard_deletion_semantics_audit.json"
echo "[BACE_HARD_DELETION_AUDIT_V3_SUCCESS]"
