#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

DATASET="${DATASET:-bace}"
case "${DATASET}" in
  bace)
    DATA_DIR="${DATA_DIR:-data/processed/BACE}"
    ;;
  tastemolnet|taste)
    DATA_DIR="${DATA_DIR:-data/processed/tastemolnet}"
    ;;
  *)
    echo "unsupported DATASET=${DATASET}; expected bace or tastemolnet" >&2
    exit 2
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/graph_cache/${DATASET}}"

echo "python=$(which python)"
python --version
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY
echo "dataset=${DATASET}"
echo "data_dir=${DATA_DIR}"
echo "output_dir=${OUTPUT_DIR}"

python scripts/build_molecular_graph_cache.py \
  --config configs/hpc.yaml \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}"
