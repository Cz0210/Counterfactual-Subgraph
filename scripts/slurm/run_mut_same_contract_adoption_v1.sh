#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${MUT_AB_TASK_SPEC:?absolute A/B task spec required}"
: "${MUT_AB_OWNER_TERMINAL:?absolute A/B owner terminal required}"
: "${MUT_AB_GATE:?absolute same-contract gate required}"
: "${MUT_AUTHORIZATION_RECEIPT:?absolute authorization receipt required}"
: "${MUT_HISTORICAL_SOURCE_ROOT:?absolute historical source root required}"
: "${MUT_COMPLETED_COMMON_ROOT:?absolute completed common root required}"
: "${MUT_ADOPTION_OUTPUT_ROOT:?fresh absolute adoption root required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/run_mut_same_contract_adoption_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ab-task-spec "$MUT_AB_TASK_SPEC" \
  --ab-owner-terminal "$MUT_AB_OWNER_TERMINAL" \
  --same-contract-gate "$MUT_AB_GATE" \
  --authorization-receipt "$MUT_AUTHORIZATION_RECEIPT" \
  --historical-source-root "$MUT_HISTORICAL_SOURCE_ROOT" \
  --completed-common-root "$MUT_COMPLETED_COMMON_ROOT" \
  --output-root "$MUT_ADOPTION_OUTPUT_ROOT" \
  --proc-root /proc
