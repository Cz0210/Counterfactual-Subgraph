#!/bin/bash
#SBATCH --job-name=bace_gcf_eval
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

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
SHARED_PROJECT_ROOT=${SHARED_PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
export BACE_METHOD=gcfexplainer
export CANDIDATE_PATH=${CANDIDATE_PATH:-$SHARED_PROJECT_ROOT/outputs/hpc/bace/baselines/gcfexplainer/full_v2/export/selected_top20.csv}
export TEACHER_PATH=${TEACHER_PATH:-$SHARED_PROJECT_ROOT/outputs/hpc/oracle/bace/bace_teacher.pkl}
export MOLCLR_ROOT=${MOLCLR_ROOT:-$SHARED_PROJECT_ROOT/pretrained_models/MolCLR}
export MOLCLR_CHECKPOINT=${MOLCLR_CHECKPOINT:-$SHARED_PROJECT_ROOT/pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth}
export TEST_CSV=${TEST_CSV:-$SHARED_PROJECT_ROOT/outputs/hpc/oracle/bace/teacher_consistent/test_source_label1_teacher_correct.csv}
export THRESHOLDS_JSON=${THRESHOLDS_JSON:-$SHARED_PROJECT_ROOT/outputs/hpc/eval/bace_ours_wnode_work_v1/thresholds.json}
export WORK_ROOT=${WORK_ROOT:-$SHARED_PROJECT_ROOT/outputs/hpc/eval/bace_gcfexplainer_wnode_work_v2}
export WORK_DIR=${WORK_DIR:-$WORK_ROOT}
export PAPER_ROOT=${PAPER_ROOT:-$SHARED_PROJECT_ROOT/outputs/hpc/eval/paper/bace_common3_standardized_v1/gcfexplainer}
export OUTPUT_DIR=${OUTPUT_DIR:-$PAPER_ROOT}

cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
exec bash "$PROJECT_ROOT/scripts/slurm/bace_eval_method_common.sh"
