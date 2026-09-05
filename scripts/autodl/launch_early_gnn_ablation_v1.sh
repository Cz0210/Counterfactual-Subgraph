#!/usr/bin/env bash
# CPU execution is submitted on HPC, never from an AutoDL main GPU reservation.
set -euo pipefail
echo 'Use the immutable HPC scripts/hpc/gnn/launch_bace_gnn_seed7.sh CPU route.'
echo 'AutoDL GPU fallback requires a fresh live main-resource gate and a checkpointed run spec.'
exit 3
