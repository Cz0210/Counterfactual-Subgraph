#!/usr/bin/env bash
# Shared helpers for running the official CLEAR / GraphCFE baseline.

set -euo pipefail

_CLEAR_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${_CLEAR_COMMON_DIR}/../../.." && pwd)"
CLEAR_DIR="${ROOT_DIR}/baselines/clear_official"
CLEAR_SRC_DIR="${CLEAR_DIR}/src"
CLEAR_DATASET_DIR="${CLEAR_DIR}/dataset"
CLEAR_LOG_DIR="${ROOT_DIR}/logs/baselines/clear"
CLEAR_MODEL_DIR="${CLEAR_DIR}/models_save"

ensure_clear_dirs() {
  mkdir -p \
    "${CLEAR_DATASET_DIR}" \
    "${CLEAR_MODEL_DIR}" \
    "${CLEAR_MODEL_DIR}/prediction" \
    "${CLEAR_DIR}/logs" \
    "${CLEAR_LOG_DIR}"
}

_clear_dataset_files() {
  local dataset="$1"
  case "${dataset}" in
    community)
      printf '%s\n' \
        "${CLEAR_DATASET_DIR}/community_3.pickle" \
        "${CLEAR_DATASET_DIR}/community_datasplit.pickle"
      ;;
    ogbg_molhiv)
      printf '%s\n' \
        "${CLEAR_DATASET_DIR}/ogbg_molhiv_full.pickle" \
        "${CLEAR_DATASET_DIR}/ogbg_molhiv_datasplit.pickle"
      ;;
    imdb_m)
      printf '%s\n' \
        "${CLEAR_DATASET_DIR}/imdb_m.pickle" \
        "${CLEAR_DATASET_DIR}/imdb_m_datasplit.pickle" \
        "${CLEAR_DATASET_DIR}/IMDBMULTI.mat"
      ;;
    *)
      echo "[CLEAR_ERROR] Unsupported CLEAR dataset: ${dataset}" >&2
      echo "[CLEAR_ERROR] Supported datasets: community, ogbg_molhiv, imdb_m" >&2
      return 2
      ;;
  esac
}

check_clear_dataset() {
  local dataset="$1"
  local missing=0
  local dataset_files
  local file_path

  echo "[CLEAR_DATASET_CHECK] dataset=${dataset}"
  if ! dataset_files="$(_clear_dataset_files "${dataset}")"; then
    return 2
  fi

  while IFS= read -r file_path; do
    [ -n "${file_path}" ] || continue
    if [ -f "${file_path}" ]; then
      echo "[CLEAR_DATASET_CHECK] ok: ${file_path}"
    else
      echo "[CLEAR_DATASET_CHECK] missing: ${file_path}" >&2
      missing=1
    fi
  done <<< "${dataset_files}"

  if [ "${missing}" -ne 0 ]; then
    echo "[CLEAR_ERROR] Missing CLEAR dataset files for ${dataset}." >&2
    echo "[CLEAR_ERROR] Copy dataset files manually into: ${CLEAR_DATASET_DIR}" >&2
    echo "[CLEAR_ERROR] Large datasets/checkpoints should not be managed by ordinary Git." >&2
    return 1
  fi
}

print_clear_env() {
  echo "===== CLEAR ENV CHECK ====="
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "CLEAR_DIR=${CLEAR_DIR}"
  echo "CLEAR_SRC_DIR=${CLEAR_SRC_DIR}"
  echo "CLEAR_DATASET_DIR=${CLEAR_DATASET_DIR}"
  echo "CLEAR_MODEL_DIR=${CLEAR_MODEL_DIR}"
  echo "CLEAR_LOG_DIR=${CLEAR_LOG_DIR}"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python || true)"
  python --version || true
  echo "git_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true)"
  echo "git_branch=$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  python - <<'PY' || true
import importlib.util
print("torch available:", importlib.util.find_spec("torch") is not None)
if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda device 0:", torch.cuda.get_device_name(0))
print("networkx available:", importlib.util.find_spec("networkx") is not None)
print("numpy available:", importlib.util.find_spec("numpy") is not None)
PY
  echo "==========================="
}
