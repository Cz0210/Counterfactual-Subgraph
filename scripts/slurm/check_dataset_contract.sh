#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=check_dataset_contract
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

CSV=${CSV:-data/raw/AIDS/HIV.csv}
SMILES_COLUMN=${SMILES_COLUMN:-smiles}
LABEL_COLUMN=${LABEL_COLUMN:-HIV_active}
EXPECTED_TOTAL=${EXPECTED_TOTAL:-41127}
EXPECTED_LABEL0=${EXPECTED_LABEL0:-39684}
EXPECTED_LABEL1=${EXPECTED_LABEL1:-1443}
OUT_JSON=${OUT_JSON:-outputs/hpc/diagnostics/dataset_contract_aids_hiv.json}

mkdir -p logs
mkdir -p "$(dirname "$OUT_JSON")"

echo "[DATASET_CONTRACT_CONFIG]"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "git_commit=$(git rev-parse --short HEAD || true)"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"
echo "python=$(which python)"
python --version
echo "CSV=$CSV"
echo "SMILES_COLUMN=$SMILES_COLUMN"
echo "LABEL_COLUMN=$LABEL_COLUMN"
echo "EXPECTED_TOTAL=$EXPECTED_TOTAL"
echo "EXPECTED_LABEL0=$EXPECTED_LABEL0"
echo "EXPECTED_LABEL1=$EXPECTED_LABEL1"
echo "OUT_JSON=$OUT_JSON"

python scripts/check_dataset_contract.py \
  --config configs/hpc.yaml \
  --csv "$CSV" \
  --smiles-column "$SMILES_COLUMN" \
  --label-column "$LABEL_COLUMN" \
  --expected-total "$EXPECTED_TOTAL" \
  --expected-label0 "$EXPECTED_LABEL0" \
  --expected-label1 "$EXPECTED_LABEL1" \
  --out-json "$OUT_JSON"
