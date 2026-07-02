#!/usr/bin/env bash
# Apply project-owned runtime patches to the official CLEAR source checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLEAR_DIR="${ROOT_DIR}/baselines/clear_official"
CLEAR_MAIN="${CLEAR_DIR}/src/main.py"

if [ ! -d "${ROOT_DIR}/.git" ] || [ ! -f "${ROOT_DIR}/README.md" ]; then
  echo "[CLEAR_PATCH_ERROR] Could not locate project root: ${ROOT_DIR}" >&2
  exit 2
fi

if [ ! -f "${CLEAR_MAIN}" ]; then
  echo "[CLEAR_PATCH_ERROR] Missing CLEAR official main.py: ${CLEAR_MAIN}" >&2
  exit 2
fi

apply_clear_patch() {
  local patch_file="$1"
  local patch_marker="$2"

  if [ ! -f "${patch_file}" ]; then
    echo "[CLEAR_PATCH_ERROR] Missing CLEAR patch file: ${patch_file}" >&2
    exit 2
  fi

  if grep -q "${patch_marker}" "${CLEAR_MAIN}"; then
    echo "[CLEAR_PATCH] already_applied marker=${patch_marker}"
    return 0
  fi

  echo "[CLEAR_PATCH] applying ${patch_file}"
  git -C "${CLEAR_DIR}" apply "${patch_file}"

  if grep -q "${patch_marker}" "${CLEAR_MAIN}"; then
    echo "[CLEAR_PATCH] applied marker=${patch_marker}"
  else
    echo "[CLEAR_PATCH_ERROR] Patch command finished but marker is still absent: ${patch_marker}" >&2
    exit 1
  fi
}

apply_clear_patch "${ROOT_DIR}/patches/clear_official/001_save_cfe_checkpoints.patch" "CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT"
apply_clear_patch "${ROOT_DIR}/patches/clear_official/002_export_test_counterfactuals.patch" "CLEAR_WRAPPER_EXPORT_TEST_COUNTERFACTUALS"
apply_clear_patch "${ROOT_DIR}/patches/clear_official/003_support_aids_dataset.patch" "CLEAR_WRAPPER_SUPPORT_AIDS_DATASET"
