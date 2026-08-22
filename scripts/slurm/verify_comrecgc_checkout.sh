#!/bin/bash
#SBATCH --job-name=verify_comrecgc_checkout
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs

COMRECGC_EXPECTED_COMMIT="${COMRECGC_EXPECTED_COMMIT:-122f9341a360e9f06bb58a2f5823bb596021f6bf}"
COMRECGC_ROOT="${COMRECGC_ROOT:-/share/home/u20526/czx/vendor/COMRECGC/$COMRECGC_EXPECTED_COMMIT}"
OUTPUT="${OUTPUT:-outputs/hpc/baselines/comrecgc/external_checkout_audit.json}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/verify_comrecgc_checkout.py \
  --config configs/hpc.yaml \
  --root "$COMRECGC_ROOT" \
  --expected-commit "$COMRECGC_EXPECTED_COMMIT" \
  --validate-imports \
  --output "$OUTPUT"
