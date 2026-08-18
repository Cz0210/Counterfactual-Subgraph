#!/bin/bash
#SBATCH --job-name=bace_globalgce_summary_v7
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
CANDIDATE_ROOT=${CANDIDATE_ROOT:-$ARTIFACT_ROOT/outputs/hpc/optimization/bace_globalgce_v7}
CALIBRATION_MANIFEST=${CALIBRATION_MANIFEST:-$CANDIDATE_ROOT/globalgce_bace_min_freq_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/selectors/bace_globalgce_v7}
args=(--config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --calibration-manifest "$CALIBRATION_MANIFEST" --candidate-root "$CANDIDATE_ROOT"
  --output-dir "$OUTPUT_DIR" --git-commit "$(git rev-parse HEAD)")
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/globalgce/freeze_bace_calibrated_summary.py "${args[@]}" --validate-only
  exit 0
fi
python scripts/baselines/globalgce/freeze_bace_calibrated_summary.py "${args[@]}"
