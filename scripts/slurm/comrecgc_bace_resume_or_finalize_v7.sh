#!/bin/bash
#SBATCH --job-name=comrecgc_bace_cont_v7
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; mkdir -p logs
GENERATION_DIR=${GENERATION_DIR:?GENERATION_DIR is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXPECTED_STEPS=${EXPECTED_STEPS:-50000}
RESUME_COMMAND_JSON=${RESUME_COMMAND_JSON:-}
args=(--config configs/hpc.yaml --set inference.fallback_to_heuristic=false
  --generation-dir "$GENERATION_DIR" --output-dir "$OUTPUT_DIR"
  --expected-steps "$EXPECTED_STEPS")
[[ -z "$RESUME_COMMAND_JSON" ]] || args+=(--resume-command-json "$RESUME_COMMAND_JSON")
echo "hostname=$(hostname) commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID:-unset}"
python -c 'import torch; assert torch.cuda.device_count() == 1; print("cuda_devices=1")'
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/comrecgc/resume_or_finalize_generation.py "${args[@]}" --validate-only
  exit $?
fi
python scripts/baselines/comrecgc/resume_or_finalize_generation.py "${args[@]}"
