#!/usr/bin/env bash
#SBATCH --job-name=t14-resume-spec
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${TASTEMOLNET_T14_OUTPUT:?required}"
: "${T14_RESUME_EXECUTION_COMMIT:?required}"
: "${T14_HISTORICAL_PROCESS_PEAK_BYTES:?required}"
: "${T14_HISTORICAL_CHECKPOINT_PEAK_BYTES:?required}"
: "${T14_RESUME_SPEC:?required}"

python scripts/autodl/build_t14_low_memory_resume_spec.py \
  --config configs/hpc.yaml \
  --output-root "$TASTEMOLNET_T14_OUTPUT" \
  --checkpoint-dir "$TASTEMOLNET_T14_OUTPUT/checkpoints/step-000000012500" \
  --resume-execution-commit "$T14_RESUME_EXECUTION_COMMIT" \
  --historical-process-peak-bytes "$T14_HISTORICAL_PROCESS_PEAK_BYTES" \
  --historical-checkpoint-peak-bytes "$T14_HISTORICAL_CHECKPOINT_PEAK_BYTES" \
  --spec-out "$T14_RESUME_SPEC"
