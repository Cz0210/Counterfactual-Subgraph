#!/bin/bash
#SBATCH --job-name=comrecgc_identity
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
set +u
source ~/.bashrc
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-smiles_pip118}"

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p logs

OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/baselines/comrecgc/audits}"
echo "[COMRECGC_STAGE_CONFIG] stage=dataset_identity output_dir=$OUTPUT_DIR"
echo "hostname=$(hostname) job_id=${SLURM_JOB_ID:-unset} commit=$(git rev-parse HEAD)"
python -m py_compile scripts/baselines/comrecgc/audit_dataset_identity.py
python scripts/baselines/comrecgc/audit_dataset_identity.py \
  --project-root "$PROJECT_ROOT" \
  --upstream-root external/COMRECGC \
  --aids-dataset-dir outputs/hpc/gcfexplainer_hiv_csv/dataset \
  --aids-source-csv outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv \
  --aids-eval-parent-ids outputs/hpc/eval/paper/molclr_node_wasserstein_figure3_theta005_raw/reference_parent_ids.csv \
  --mut-dataset-dir outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset \
  --mut-eval-parent-csv outputs/hpc/datasets/mutagenicity_v1_teacher_consistent/test_source_label1_teacher_correct.csv \
  --output-dir "$OUTPUT_DIR"
grep -q '\[COMRECGC_DATASET_IDENTITY_PASS\]' "$OUTPUT_DIR/dataset_identity_audit.txt"
echo "[COMRECGC_DATASET_IDENTITY_SUCCESS]"
