#!/bin/bash
# V6 overrides GPU defaults. Exactly two independent CPU chains maximum.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${CM4_V6_CODE:?immutable code root}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "CPU-only CM4 job=$SLURM_JOB_ID python=$(command -v python) host=$(hostname)"
python -V
python -c 'import torch; print("CUDA available:",torch.cuda.is_available())'
failures=0
if [[ "$CM4_V6_CHAIN" == "taste-bace" ]]; then
 python -I -B scripts/run_cm4_v6_reselect.py --config configs/hpc.yaml --action first-prototype \
  --contract "$CM4_V5_ROOT/taste_eval_contract.json" --v5-cm-root "$CM4_V5_ROOT/cm" \
  --source-root "$CM4_BASE/postfilter-20260914/tastemolnet-production" --output-root "$CM4_V6_ROOT/taste-flat-audit" || failures=1
 datasets=(BACE)
else
 datasets=(Mutagenicity AIDS)
fi
for dataset in "${datasets[@]}"; do
 case "$dataset" in
 BACE) source_root="$CM4_BASE/bace-k20-20260910";;
 Mutagenicity) source_root="$CM4_BASE/postfilter-20260914/mutagenicity-production";;
 AIDS) source_root="$CM4_BASE/aids-gap-first-20260914/source-descriptive-v1-20260914T151000Z";;
 esac
 # Each dataset is independent; failure must not suppress the next dataset.
 (set -e; for stage in select evaluate audit; do
   python -I -B scripts/run_cm4_v6_reselect.py --config configs/hpc.yaml --action "$stage" \
    --dataset "$dataset" --source-root "$source_root" --output-root "$CM4_V6_ROOT/$dataset"
 done) || { echo "DATASET_FAILED_WITH_PRESERVED_RECEIPTS=$dataset"; failures=1; }
done
exit "$failures"
