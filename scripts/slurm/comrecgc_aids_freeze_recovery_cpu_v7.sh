#!/bin/bash
#SBATCH --job-name=comrecgc_aids_freeze_cpu_v7
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u; source ~/.bashrc; conda activate smiles_pip118; set -u
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}}
cd "$PROJECT_ROOT"; export PYTHONPATH=$PWD; export CUDA_VISIBLE_DEVICES=""; mkdir -p logs
ACTION=${ACTION:?ACTION must be validate or recover}
SOURCE_GENERATION_DIR=${SOURCE_GENERATION_DIR:?SOURCE_GENERATION_DIR is required}
DATASET_DIR=${DATASET_DIR:?DATASET_DIR is required}
SOURCE_CSV=${SOURCE_CSV:?SOURCE_CSV is required}
AUDIT_OUTPUT=${AUDIT_OUTPUT:?AUDIT_OUTPUT is required}
OUTPUT_DIR=${OUTPUT_DIR:-}
EXPECTED_PROJECT_COMMIT=${EXPECTED_PROJECT_COMMIT:-}
[[ "$ACTION" == validate || "$ACTION" == recover ]] || exit 2
args=(--source-generation-dir "$SOURCE_GENERATION_DIR" --dataset aids
  --dataset-dir "$DATASET_DIR" --source-csv "$SOURCE_CSV"
  --audit-output "$AUDIT_OUTPUT" --expected-steps 50000)
[[ -z "$EXPECTED_PROJECT_COMMIT" ]] || args+=(--expected-project-commit "$EXPECTED_PROJECT_COMMIT")
if [[ "$ACTION" == validate ]]; then args+=(--validate-only); else
  [[ -n "$OUTPUT_DIR" ]] || { echo 'OUTPUT_DIR required for recover' >&2; exit 2; }
  args+=(--output-dir "$OUTPUT_DIR")
fi
echo "hostname=$(hostname) action=$ACTION commit=$(git rev-parse HEAD) gpus=0"
python -c 'import os,torch; assert os.environ.get("CUDA_VISIBLE_DEVICES")==""; assert torch.cuda.device_count()==0; print("cpu_only=true")'
if [[ ${DRY_RUN:-0} == 1 || ${VALIDATE_ONLY:-0} == 1 ]]; then
  python scripts/baselines/comrecgc/recover_completed_generation_freeze.py --help >/dev/null
  echo '[COMRECGC_AIDS_FREEZE_CPU_V7_VALIDATE_OK]'; exit 0
fi
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python scripts/baselines/comrecgc/recover_completed_generation_freeze.py "${args[@]}"
test -s "$AUDIT_OUTPUT"
if [[ "$ACTION" == recover ]]; then test -s "$OUTPUT_DIR/_RUN_COMPLETE.json"; fi
