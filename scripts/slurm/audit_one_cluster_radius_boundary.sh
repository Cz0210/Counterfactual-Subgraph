#!/bin/bash
#SBATCH --job-name=comrecgc_radius_audit_disabled
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

run_autodl_only_reference() {
  python scripts/baselines/comrecgc/audit_one_cluster_radius_boundary.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --terminal-one-cluster-manifest "$TERMINAL_ONE_CLUSTER_MANIFEST" \
    --expected-terminal-manifest-sha256 "$EXPECTED_TERMINAL_MANIFEST_SHA256" \
    --output-dir "$OUTPUT_DIR"
}

echo "[CONFIG_ERROR] one-cluster radius audit is AutoDL CPU-only; do not use sbatch." >&2
exit 64
