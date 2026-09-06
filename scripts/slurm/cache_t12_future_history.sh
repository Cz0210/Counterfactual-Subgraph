#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# CPU-only storage verification, not the generic A800 scientific job template.
# This paired entrypoint is an intentional refusal: T12 inputs remain AutoDL-only.
set -eo pipefail
echo 'T12 future history cache is AutoDL-only; no HPC transfer or GPU task is authorized.' >&2
exit 64
