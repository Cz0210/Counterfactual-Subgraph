#!/bin/bash
# The real paired producer must run on its existing AutoDL GPU lease.
# HPC is CPU-only: it must never silently substitute a CPU arm or request GPU.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
echo 'T12 real adapter producer is AutoDL-only; use its UUID/lease entrypoint.' >&2
exit 64
