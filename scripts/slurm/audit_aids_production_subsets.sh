#!/bin/bash
#SBATCH --job-name=aids_subset_audit_disabled
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'

# Static CLI synchronization only. This function is intentionally never
# called by the Slurm wrapper; the scientific route is AutoDL CPU-only.
run_autodl_only_reference() {
  python scripts/baselines/comrecgc/audit_aids_production_subsets.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --close-pair-contract "$CLOSE_PAIR_CONTRACT" \
    --expected-close-pair-contract-sha256 "$EXPECTED_CLOSE_PAIR_CONTRACT_SHA256" \
    --physical-pairs "$PHYSICAL_PAIRS" \
    --expected-physical-pairs-sha256 "$EXPECTED_PHYSICAL_PAIRS_SHA256" \
    --expected-sklearn-version "$EXPECTED_SKLEARN_VERSION" \
    --output-dir "$OUTPUT_DIR"
}

echo "[CONFIG_ERROR] AIDS production-subset audit is AutoDL CPU-only; do not submit it through sbatch." >&2
echo "[SYNCED_CLI] python scripts/baselines/comrecgc/audit_aids_production_subsets.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false <authority-bound arguments>" >&2
exit 64
