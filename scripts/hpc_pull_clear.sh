#!/usr/bin/env bash
# Pull project/submodules on HPC and prepare CLEAR runtime directories.

set -euo pipefail

if [ ! -d ".git" ] || [ ! -f "README.md" ]; then
  echo "[CLEAR_HPC_PULL_ERROR] Run this script from the project root." >&2
  exit 2
fi

echo "===== CLEAR HPC PULL ====="
echo "pwd=$(pwd)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "commit_before=$(git rev-parse HEAD)"

git pull --recurse-submodules
git submodule sync --recursive
git submodule update --init --recursive

echo "commit_after=$(git rev-parse HEAD)"

# shellcheck source=scripts/baselines/clear/common.sh
source scripts/baselines/clear/common.sh
ensure_clear_dirs

mkdir -p \
  baselines/clear_official/dataset \
  baselines/clear_official/models_save/prediction \
  baselines/clear_official/logs \
  logs/baselines/clear

missing_any=0
for dataset in community ogbg_molhiv imdb_m; do
  if check_clear_dataset "${dataset}"; then
    echo "[CLEAR_HPC_PULL] dataset ${dataset}: ready"
  else
    missing_any=1
    echo "[CLEAR_HPC_PULL] dataset ${dataset}: missing files"
  fi
done

if [ "${missing_any}" -ne 0 ]; then
  cat <<'EOF'

[CLEAR_HPC_PULL_NOTICE]
Some CLEAR dataset files are missing.
Please copy dataset files manually into:
  baselines/clear_official/dataset/

Large datasets, checkpoints, model weights, and logs should not be managed by
ordinary Git. If model weights must be pulled through Git, use Git LFS.
EOF
fi

echo "===== CLEAR HPC PULL DONE ====="
