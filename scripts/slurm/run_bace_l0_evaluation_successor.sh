#!/usr/bin/env bash
# User-authorized L0 evaluation-only CPU exception to the default A800 template.
# Never requests a main GPU, generates proposals or repeats GNN correction.
# import-result is AutoDL-only and is intentionally not submitted through Slurm.
#SBATCH --partition=intel
#SBATCH --job-name=bace-l0-at-most-k
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
: "${LLM_EXECUTION_WORKTREE:?immutable driver required}"
cd "$LLM_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
if [[ "${1:-}" == "import-result" ]]; then
  echo "L0 result import is AutoDL-only, not an HPC Slurm stage" >&2
  exit 64
fi
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("L0 evaluation-only CPU")'
exec python -I -B scripts/hpc/llm/run_bace_l0_evaluation_successor.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
