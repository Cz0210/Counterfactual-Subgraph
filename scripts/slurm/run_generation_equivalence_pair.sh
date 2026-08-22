#!/bin/bash
#SBATCH --job-name=comrecgc_ab_pair
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/baselines/comrecgc/run_generation_equivalence_pair.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --upstream-root "${COMRECGC_ROOT:?}" \
  --dataset-dir "${DATASET_DIR:?}" \
  --gnn-checkpoint "${GNN_CHECKPOINT:?}" \
  --distance-checkpoint "${DISTANCE_CHECKPOINT:?}" \
  --output-dir "${OUTPUT_DIR:?}" \
  --steps "${STEPS:?500 or 1000}" \
  --workers "${WORKERS:-4}" \
  --max-inflight "${MAX_INFLIGHT:-64}" \
  --source-cache-capacity "${SOURCE_CACHE_CAPACITY:-1024}" \
  --candidate-cache-capacity "${CANDIDATE_CACHE_CAPACITY:-8192}"
