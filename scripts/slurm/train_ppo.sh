#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-}
GEN_TOP_P=${GEN_TOP_P:-}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

cd "${PROJECT_DIR}"

export PYTHONPATH=$PWD
echo "repo pwd: $(pwd)"
echo "PYTHONPATH(after export): ${PYTHONPATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-<unset>}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE:-<unset>}"
echo "GEN_TOP_P=${GEN_TOP_P:-<unset>}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-<unset>}"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  exit 1
fi

echo "===== RUNNING PPO TRAINING ====="
cmd=(
  python
  scripts/train_ppo.py
  --config
  configs/hpc.yaml
  --teacher-path
  "${TEACHER_PATH}"
  --require-teacher-sem
  --teacher-sem-scale
  1.0
  --teacher-sem-missing-penalty
  -5.0
  --teacher-cf-flip-bonus
  1.0
)

if [ -n "${GEN_MAX_NEW_TOKENS}" ]; then
  cmd+=(--gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}")
fi
if [ -n "${GEN_TEMPERATURE}" ]; then
  cmd+=(--gen-temperature "${GEN_TEMPERATURE}")
fi
if [ -n "${GEN_TOP_P}" ]; then
  cmd+=(--gen-top-p "${GEN_TOP_P}")
fi
if [ -n "${GEN_DO_SAMPLE}" ]; then
  case "${GEN_DO_SAMPLE,,}" in
    1|true|yes|on)
      cmd+=(--gen-do-sample)
      ;;
    0|false|no|off)
      cmd+=(--no-gen-do-sample)
      ;;
    *)
      echo "[ERROR] GEN_DO_SAMPLE must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
cmd+=("$@")

"${cmd[@]}"
echo "===== DONE ====="
