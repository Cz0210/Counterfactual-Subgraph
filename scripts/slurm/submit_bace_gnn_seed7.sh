#!/usr/bin/env bash
# This is a submission-only CLI; nested sbatch submission is deliberately disabled.
set -euo pipefail
echo 'Use scripts/hpc/gnn/launch_bace_gnn_seed7.sh from login; it submits CPU-only jobs.' >&2
exit 2
