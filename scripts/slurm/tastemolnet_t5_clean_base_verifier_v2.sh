#!/usr/bin/env bash
#SBATCH --job-name=taste-t5-base-verify-v2
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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

echo "REFUSING_HPC_EXECUTION: TasteMolNet clean-base verification is AutoDL-only." >&2
echo "CLI parity: python scripts/autodl/tastemolnet_t5_clean_base_verifier_v2.py --config configs/hpc.yaml --sealed SEALED --final-path FINAL --source-model SOURCE --expected-attempt-id UUID --expected-generation-token UUID --expected-controller-id ID --expected-git-commit COMMIT --expected-source-inventory-sha256 SHA256" >&2
exit 75
