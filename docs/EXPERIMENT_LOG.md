# Experiment Log

This file is append-only. Each Slurm experiment should be submitted through `scripts/exp_sbatch.py` or `scripts/exp_sbatch.sh` so that job id, command, git commit, environment snapshot, and expected output paths are preserved.

Recommended command:

```bash
scripts/exp_sbatch.sh \
  --name "short experiment name" \
  --tags "label1,selector,ccrcov" \
  --notes "short note" \
  --expected-output-root "outputs/hpc/..." \
  -- scripts/slurm/xxx.sh
```
