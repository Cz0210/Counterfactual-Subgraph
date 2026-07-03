# Experiment Tracking Protocol

## 1. Motivation

The project now runs many Slurm experiments, including PPO, candidate pool generation, selector sweeps, baseline evaluation, CCRCov threshold sweeps, and visualization jobs. Each submission must preserve enough metadata to recover the result later: job id, submission command, script, important parameters, expected output path, Slurm log hints, git commit, and experiment notes.

The standard entrypoint is `scripts/exp_sbatch.py`. It calls the real `sbatch` command and appends records to both the human-readable experiment log and the machine-readable registry.

## 2. Standard Submission

Recommended Python entrypoint:

```bash
python scripts/exp_sbatch.py \
  --name "label1 CCRCov ours vs GCF" \
  --tags "label1,ccrcov,gcf,ged,embedding" \
  --notes "GED and embedding threshold sweep" \
  --expected-output-root "outputs/hpc/eval_close_cf_coverage" \
  -- scripts/slurm/evaluate_close_cf_coverage_label1_ours_gcf_all.sh
```

Equivalent lightweight shell wrapper:

```bash
scripts/exp_sbatch.sh \
  --name "label1 selector cov20" \
  --tags "label1,selector,cov20" \
  --notes "coverage-heavy MMR selector" \
  -- scripts/slurm/select_label1_top20_cov20.sh
```

The wrapper writes:

- `docs/EXPERIMENT_LOG.md`
- `outputs/hpc/experiment_registry/jobs.jsonl`

## 3. Optional Shell Alias

On HPC, add this to `~/.bashrc`:

```bash
alias esbatch='python /share/home/u20526/czx/counterfactual-subgraph/scripts/exp_sbatch.py'
```

Then submit with:

```bash
esbatch --name "..." --tags "..." -- scripts/slurm/xxx.sh
```

If stricter logging is useful, define a shell function:

```bash
sbatch_logged() {
  python /share/home/u20526/czx/counterfactual-subgraph/scripts/exp_sbatch.py -- "$@"
}
```

Do not override the system `sbatch` command by default unless the team has agreed to that workflow.

## 4. Recovering Direct sbatch Jobs

If a job was submitted directly with `sbatch`, record it manually in `docs/EXPERIMENT_LOG.md`. A dry-run entry can help assemble the metadata without submitting another job:

```bash
python scripts/exp_sbatch.py \
  --name "manual backfill job 123456" \
  --tags "manual,backfill" \
  --notes "Submitted directly via sbatch; manually recorded." \
  --dry-run \
  -- scripts/slurm/xxx.sh
```

A future `scripts/record_existing_job.py` could automate this backfill, but it is not part of the current protocol.

## 5. Status Sync

After jobs finish, append Slurm status snapshots:

```bash
python scripts/sync_experiment_status.py \
  --registry-jsonl outputs/hpc/experiment_registry/jobs.jsonl \
  --markdown-log docs/EXPERIMENT_LOG.md
```

For a single job:

```bash
python scripts/sync_experiment_status.py --job-id 1234567
```

The sync script uses `sacct` first and falls back to `squeue` when needed. If neither command is available, it records `STATUS_QUERY_FAILED` rather than crashing.

## 6. Logging Policy

- `docs/EXPERIMENT_LOG.md` is a human-readable append-only experiment log and can be committed to git.
- `outputs/hpc/experiment_registry/jobs.jsonl` is the machine-readable experiment registry.
- Each experiment should at least fill `--name` and `--tags`.
- Final AIDS/HIV baseline jobs should record the dataset contract metadata in `--notes` or environment variables: `DATASET_SOURCE=data/raw/AIDS/HIV.csv`, `SMILES_COLUMN=smiles`, `LABEL_COLUMN=HIV_active`, `TARGET_LABEL=1`, `BASELINE_DATASET_KEY=<hiv|aids|...>`, `TEACHER_PATH` or `TEACHER_KIND`, and `CF_MODE=strict_flip`.
- `--expected-output-root` should be filled whenever the output root is known before submission.
- Important result paths should be recorded in `--notes`.
- After completion, append key result paths such as `selector_report`, `audit_report`, `combined_report`, figures, and final tables to a later status section or a follow-up note.
