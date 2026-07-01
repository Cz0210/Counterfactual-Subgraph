#!/usr/bin/env bash
# Apply project-owned runtime patches to the official CLEAR source checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CLEAR_DIR="${ROOT_DIR}/baselines/clear_official"
CLEAR_MAIN="${CLEAR_DIR}/src/main.py"
PATCH_FILE="${ROOT_DIR}/patches/clear_official/001_save_cfe_checkpoints.patch"
PATCH_MARKER="CLEAR_WRAPPER_SAVE_CFE_CHECKPOINT"

if [ ! -d "${ROOT_DIR}/.git" ] || [ ! -f "${ROOT_DIR}/README.md" ]; then
  echo "[CLEAR_PATCH_ERROR] Could not locate project root: ${ROOT_DIR}" >&2
  exit 2
fi

if [ ! -f "${CLEAR_MAIN}" ]; then
  echo "[CLEAR_PATCH_ERROR] Missing CLEAR official main.py: ${CLEAR_MAIN}" >&2
  exit 2
fi

if [ ! -f "${PATCH_FILE}" ]; then
  echo "[CLEAR_PATCH_ERROR] Missing CLEAR patch file: ${PATCH_FILE}" >&2
  exit 2
fi

if grep -q "${PATCH_MARKER}" "${CLEAR_MAIN}"; then
  echo "[CLEAR_PATCH] already_applied marker=${PATCH_MARKER}"
  exit 0
fi

echo "[CLEAR_PATCH] applying ${PATCH_FILE}"
git -C "${CLEAR_DIR}" apply "${PATCH_FILE}"

if grep -q "${PATCH_MARKER}" "${CLEAR_MAIN}"; then
  echo "[CLEAR_PATCH] applied marker=${PATCH_MARKER}"
else
  echo "[CLEAR_PATCH_ERROR] Patch command finished but marker is still absent: ${PATCH_MARKER}" >&2
  exit 1
fi
