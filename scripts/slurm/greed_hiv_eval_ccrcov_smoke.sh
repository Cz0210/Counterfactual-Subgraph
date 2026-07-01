#!/bin/bash
#SBATCH -J greed_eval_smoke
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

DATASET_CSV=${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
OURS_SELECTED_PATH=${OURS_SELECTED_PATH:-outputs/hpc/selectors/stable300_label1_merged_base_temp07_top20_mmr_cov20}
GT_FULLGRAPH_DEFAULT="${PROJECT_ROOT}/outputs/hpc/baselines/gt_fullgraph/label1_opposite_fullgraph_candidates_max2000_seed0.csv"
if [ -z "${GT_FULLGRAPH_CANDIDATES_PATH:-}" ] && [ -n "${GCF_CANDIDATES_PATH:-}" ]; then
  GT_FULLGRAPH_CANDIDATES_PATH="${GCF_CANDIDATES_PATH}"
fi
GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH:-${GT_FULLGRAPH_DEFAULT}}
GREED_CHECKPOINT=${GREED_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged_smoke.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_greed_hiv_smoke}
TEACHER_PATH=${TEACHER_PATH:-}
CF_MODE=${CF_MODE:-strict_flip}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

resolve_teacher_path() {
  if [ -n "${TEACHER_PATH:-}" ]; then
    return
  fi
  local default_teacher="${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl"
  if [ -f "${default_teacher}" ]; then
    TEACHER_PATH="${default_teacher}"
    return
  fi
  local candidate
  for candidate in "${PROJECT_ROOT}"/outputs/hpc/oracle/*.pkl; do
    if [ -f "${candidate}" ]; then
      TEACHER_PATH="${candidate}"
      return
    fi
  done
  if [ -d "${PROJECT_ROOT}/outputs/hpc" ]; then
    while IFS= read -r candidate; do
      if [ -f "${candidate}" ]; then
        TEACHER_PATH="${candidate}"
        return
      fi
    done < <(find "${PROJECT_ROOT}/outputs/hpc" -type f \( -iname "*rf*model*.pkl" -o -iname "*aids*model*.pkl" -o -iname "*hiv*model*.pkl" \) | sort)
  fi
}

resolve_teacher_path

echo "[TEACHER_CONFIG]"
echo "TEACHER_PATH=${TEACHER_PATH:-}"

if [ -z "${TEACHER_PATH:-}" ] || [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] TEACHER_PATH is required. Please submit with:"
  echo "TEACHER_PATH=/path/to/aids_rf_model.pkl sbatch scripts/slurm/greed_hiv_eval_ccrcov_smoke.sh"
  exit 2
fi

echo "===== GREED CCRCov SMOKE ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
python --version
echo "[GT_FULLGRAPH_CONFIG]"
echo "GT_FULLGRAPH_CANDIDATES_PATH=${GT_FULLGRAPH_CANDIDATES_PATH}"
echo "[CF_CONFIG]"
echo "CF_MODE=${CF_MODE}"
echo "MIN_CF_DROP=${MIN_CF_DROP}"
python - <<'PY'
import importlib.util, torch
print("rdkit available:", importlib.util.find_spec("rdkit") is not None)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

python scripts/evaluate_ccrcov_with_greed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --ours-selected-path "${OURS_SELECTED_PATH}" \
  --gt-fullgraph-candidates-path "${GT_FULLGRAPH_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --greed-checkpoint "${GREED_CHECKPOINT}" \
  --label "${LABEL:-1}" \
  --thresholds "${THRESHOLDS:-0.05,0.10,0.20}" \
  --cf-mode "${CF_MODE}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-parents "${MAX_PARENTS:-25}" \
  --max-candidates "${MAX_CANDIDATES:-20}" \
  --partial-every 500

echo "===== GREED CCRCov SMOKE DONE ====="
find "${OUTPUT_ROOT}" -maxdepth 3 -type f | sort
