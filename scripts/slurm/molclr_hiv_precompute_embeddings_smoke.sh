#!/bin/bash
#SBATCH -J molclr_pre_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

if [ -z "${MOLCLR_ROOT:-}" ] || [ -z "${MOLCLR_CKPT:-}" ]; then
  echo "[ERROR] MOLCLR_ROOT and MOLCLR_CKPT are required."
  exit 2
fi

echo "===== MOLCLR CCRCov PRECOMPUTE SMOKE ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
python - <<'PY'
import importlib.util, torch
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("torch_geometric available:", importlib.util.find_spec("torch_geometric") is not None)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

python scripts/precompute_molclr_embeddings_for_ccrcov.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}" \
  --ours-selected-path "${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH:-outputs/hpc/selectors/gt_fullgraph_tanimoto_baseline_label1/beta_20p0_gamma_5p0}" \
  --molclr-root "${MOLCLR_ROOT}" \
  --molclr-checkpoint "${MOLCLR_CKPT}" \
  --output-dir "${OUTPUT_DIR:-outputs/hpc/molclr_ccrcov_embeddings_smoke}" \
  --label "${LABEL:-1}" \
  --max-parents "${MAX_PARENTS:-25}" \
  --max-candidates "${MAX_CANDIDATES:-20}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --device "${DEVICE:-cuda}" \
  --invalid-policy skip

echo "===== MOLCLR CCRCov PRECOMPUTE SMOKE DONE ====="
