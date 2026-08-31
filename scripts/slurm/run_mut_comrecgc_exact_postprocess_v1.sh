#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=mut_exact_post

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${AUTODL_RUNTIME_ROOT:=/autodl-fs/data/counterfactual-subgraph-runtime}"
: "${AUTODL_INCOMING_ROOT:=/autodl-fs/data/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload}"
: "${MUT_EXACT_SOURCE_ROOT:=$AUTODL_RUNTIME_ROOT/outputs/autodl/repairs/mut_comrecgc_exact_multicomponent_v1_20260830T184359Z}"
: "${MUT_EXACT_COMMON_ROOT:=$MUT_EXACT_SOURCE_ROOT/full}"
: "${MUT_EXACT_ADOPTION_RECEIPT:=$AUTODL_RUNTIME_ROOT/control/fast_16of16_v2/adoptions/mut_comrecgc_exact_multicomponent_adoption.json}"
: "${MUT_GENERATION_ROOT:=$AUTODL_RUNTIME_ROOT/outputs/autodl/recovery/mutagenicity_comrecgc_lineage_v3_20260822T025620Z}"
: "${MUT_UPSTREAM_ROOT:=$AUTODL_INCOMING_ROOT/vendor/COMRECGC/122f9341a360e9f06bb58a2f5823bb596021f6bf}"
: "${MUT_DATASET_DIR:=$AUTODL_INCOMING_ROOT/project/outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset}"
: "${MUT_DISTANCE_CHECKPOINT:=$AUTODL_INCOMING_ROOT/project/outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt}"
: "${MUT_TEST_DATASET_CSV:=$AUTODL_INCOMING_ROOT/project/outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/test_source_label1_teacher_correct.csv}"
: "${MUT_TEACHER_PATH:=$AUTODL_INCOMING_ROOT/project/outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl}"
: "${MOLCLR_ROOT:=$AUTODL_INCOMING_ROOT/project/pretrained_models/MolCLR}"
: "${MOLCLR_CHECKPOINT:=$MOLCLR_ROOT/ckpt/pretrained_gin/checkpoints/model.pth}"
: "${MUT_THRESHOLDS_PATH:=$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/four_methods_four_datasets_repair_v1/cells/mutagenicity/comrecgc/threshold-freeze/attempt-0/frozen_threshold_contract.json}"
: "${MUT_TRACE_PARITY:?set MUT_TRACE_PARITY to a mut_trace_on_off_parity_v1 PASS receipt}"
: "${MUT_POSTPROCESS_OUTPUT_ROOT:?set a fresh MUT_POSTPROCESS_OUTPUT_ROOT}"
: "${PRIOR_MATRIX_ROOT:?set PRIOR_MATRIX_ROOT to the current closed matrix authority}"
: "${MUT_MATRIX_OUTPUT_ROOT:?set a fresh MUT_MATRIX_OUTPUT_ROOT}"
: "${MUT_POSTPROCESS_RESUME:=0}"

export CUDA_VISIBLE_DEVICES=""
export DEVICE=cpu
export GPU_REQUIRED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "route=read_only_exact_adoption_then_chemistry_wnode_freeze_matrix"
echo "common_root=$MUT_EXACT_COMMON_ROOT"
echo "trace_parity=$MUT_TRACE_PARITY"

resume_args=()
if [[ "$MUT_POSTPROCESS_RESUME" == "1" ]]; then
  resume_args+=(--resume)
elif [[ "$MUT_POSTPROCESS_RESUME" != "0" ]]; then
  echo "MUT_POSTPROCESS_RESUME must be 0 or 1" >&2
  exit 2
fi

python scripts/autodl/run_mut_comrecgc_exact_postprocess_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-generation-root "$MUT_GENERATION_ROOT" \
  --upstream-root "$MUT_UPSTREAM_ROOT" \
  --dataset-dir "$MUT_DATASET_DIR" \
  --distance-checkpoint "$MUT_DISTANCE_CHECKPOINT" \
  --dataset-csv "$MUT_TEST_DATASET_CSV" \
  --teacher-path "$MUT_TEACHER_PATH" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --thresholds-path "$MUT_THRESHOLDS_PATH" \
  --exact-adoption-receipt "$MUT_EXACT_ADOPTION_RECEIPT" \
  --common-root "$MUT_EXACT_COMMON_ROOT" \
  --trace-parity "$MUT_TRACE_PARITY" \
  --prior-matrix-root "$PRIOR_MATRIX_ROOT" \
  --matrix-output-root "$MUT_MATRIX_OUTPUT_ROOT" \
  --output-root "$MUT_POSTPROCESS_OUTPUT_ROOT" \
  "${resume_args[@]}"
