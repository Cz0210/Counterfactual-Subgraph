#!/bin/bash
#SBATCH --job-name=bace_artifact_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
mkdir -p logs

PAPER_ROOT=${PAPER_ROOT:-outputs/hpc/eval/paper/bace_ours_wnode}
THRESHOLDS_JSON=${THRESHOLDS_JSON:-outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
EXPECTED_METHODS=${EXPECTED_METHODS:-ours}

echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(which python)"
python --version
echo "paper_root=$PAPER_ROOT"
echo "thresholds_json=$THRESHOLDS_JSON"
echo "expected_methods=$EXPECTED_METHODS"

python scripts/audit_bace_paper_artifacts.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --root "$PAPER_ROOT" \
  --thresholds-json "$THRESHOLDS_JSON" \
  --methods "$EXPECTED_METHODS"

test -s "$PAPER_ROOT/bace_paper_artifact_audit.json"
echo "[BACE_AUDIT_PAPER_ARTIFACTS_SUCCESS]"
