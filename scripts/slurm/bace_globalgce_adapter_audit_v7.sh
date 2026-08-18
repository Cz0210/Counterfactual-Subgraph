#!/bin/bash
#SBATCH --job-name=bace_globalgce_adapter_audit_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
OLD_ROOT=${OLD_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v4_storage_v6}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_globalgce_adapter_v7}
args=(--config configs/hpc.yaml --set inference.fallback_to_heuristic=false --output-dir "$OUTPUT_DIR")
for value in 18 7 4; do
  selected="$OLD_ROOT/candidates/min_freq_$value/selector/selected_top20_for_eval.csv"
  matrix="/share/project/p20526/u20526/counterfactual-subgraph/globalgce_bace_v6/min_freq_$value/calibration_matrix/pair_matrix.jsonl"
  test -s "$selected"; test -s "$matrix"
  args+=(--selected-csv "$selected" --old-pair-matrix "$matrix")
done
echo "hostname=$(hostname) commit=$(git rev-parse HEAD) gpus=0"
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/globalgce/audit_bace_action_adapter.py "${args[@]}" --validate-only
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
python scripts/baselines/globalgce/audit_bace_action_adapter.py "${args[@]}"
