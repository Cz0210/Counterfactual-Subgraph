#!/bin/bash
#SBATCH --job-name=bace_ours
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u

export BACE_METHOD=ours
export CANDIDATE_PATH=${CANDIDATE_PATH:-outputs/hpc/selectors/bace_ours_top20}
export WORK_ROOT=${WORK_ROOT:-outputs/hpc/eval/bace_ours_wnode_work_v1}
export THRESHOLDS_JSON=${THRESHOLDS_JSON:-$WORK_ROOT/thresholds.json}
export PAPER_ROOT=${PAPER_ROOT:-outputs/hpc/eval/paper/bace_ours_wnode}
export OUTPUT_DIR=${OUTPUT_DIR:-$PAPER_ROOT}
PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
exec bash "$PROJECT_ROOT/scripts/slurm/bace_eval_method_common.sh"
