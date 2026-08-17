#!/bin/bash
#SBATCH --job-name=bace_comrecgc_external_gate_v4
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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
COMRECGC_EXPECTED_COMMIT=${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}
COMRECGC_ROOT=${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}
OUTPUT_DIR=${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/hpc/postmortems/bace_v4_infra_recovery_20260811/comrecgc_external_gate}
DRY_RUN=${DRY_RUN:-0}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
echo "hostname=$(hostname)"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "git_commit=$(git rev-parse HEAD)"
test -d "$COMRECGC_ROOT" || { echo "missing COMRECGC_ROOT=$COMRECGC_ROOT" >&2; exit 2; }
if [[ "$DRY_RUN" == 1 || "$VALIDATE_ONLY" == 1 ]]; then
  python scripts/verify_comrecgc_checkout.py --help >/dev/null
  echo '[BACE_COMRECGC_EXTERNAL_GATE_VALIDATE_OK]'
  exit 0
fi
test ! -e "$OUTPUT_DIR" || { echo "output collision: $OUTPUT_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
python scripts/verify_comrecgc_checkout.py \
  --config configs/hpc.yaml \
  --root "$COMRECGC_ROOT" \
  --expected-commit "$COMRECGC_EXPECTED_COMMIT" \
  --validate-imports \
  --output "$OUTPUT_DIR/external_checkout_audit.json"
test -s "$OUTPUT_DIR/external_checkout_audit.json"
echo '[BACE_COMRECGC_EXTERNAL_GATE_SUCCESS]'
