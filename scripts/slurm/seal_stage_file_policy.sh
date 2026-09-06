#!/bin/bash
# AutoDL-only statvfs domain policy: this is intentionally not an HPC science job.
# A Slurm node is not the AutoDL resource domain and cannot seal its admission.
set -euo pipefail
echo 'REFUSED: run scripts/autodl/seal_stage_file_policy.py --config configs/hpc.yaml on the actual AutoDL resource domain.' >&2
exit 64
