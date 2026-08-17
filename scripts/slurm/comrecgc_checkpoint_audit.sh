#!/bin/bash
#SBATCH --job-name=comrecgc_checkpoint_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs

GENERATION_DIR="${GENERATION_DIR:?GENERATION_DIR is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
echo "[COMRECGC_STAGE_CONFIG] stage=checkpoint_audit generation_dir=$GENERATION_DIR output=$OUTPUT"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
python -V
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/baselines/comrecgc/audit_generation_checkpoint.py \
  --config configs/hpc.yaml \
  --generation-dir "$GENERATION_DIR" \
  --output "$OUTPUT"
test -s "$OUTPUT"
