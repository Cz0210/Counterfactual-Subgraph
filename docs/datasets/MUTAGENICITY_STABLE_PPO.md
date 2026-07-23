# Mutagenicity Stable PPO

## Shared Core Audit

Mutagenicity reuses the decoded-chemistry loop in
`scripts/train_ppo_stable.py`; it does not implement another PPO algorithm.
One stable PPO update consumes one DataLoader rollout batch. In this loop,
`mini_batch_size` and `gradient_accumulation_steps` do not split the rollout:
with `ppo_epochs=1`, one rollout batch produces one optimizer update.

The existing AIDS conservative wrapper leaves `batch_size` at 64. Therefore
`max_steps=300` means up to 300 rollout batches, not 300 individual parents.
The DataLoader itself uses `shuffle=False` and the legacy loop restarts it when
exhausted.

The shared model stack is:

1. a 4-bit ChemLLM causal LM plus one trainable LoRA policy adapter;
2. a separately loaded, frozen base plus the same LoRA reference adapter;
3. a separately loaded ChemLLM backbone with only the TRL value head trainable.

The stable loop applies PPO clipping, gradient clipping, reward clipping and
normalization, target/hard KL checks, adaptive KL, periodic checkpoints, and
validation-based best-checkpoint saving. Candidate traces are JSONL reward
records; policy adapters and the standalone value head are saved together.

## Mutagenicity Contract

- Policy initialization:
  `outputs/hpc/mutagenicity/sft_continued_v1/checkpoint-200`, normally reached
  through `outputs/hpc/mutagenicity/final/sft_continued_v1_best`.
- Train parents: 1,448 teacher-correct label-1 molecules.
- Validation parents: 260 teacher-correct label-1 molecules.
- Calibration and test are not loaded.
- Source/target direction: mutagenic `1` to non-mutagenic `0`.
- Strict flip: `pred_before == 1 and pred_after == 0`.
- Counterfactual drop: `p1_before - p1_after`.

The full route hashes `seed:molecule_id` to form a deterministic order, uses
64 parents per nominal rollout, and runs exactly
`ceil(1448 / 64) = 23` updates. The last rollout has 40 parents. The smoke
route uses five deterministically selected parents, batch size one, and five
updates. Both routes reject repeated `molecule_id` values within this first
epoch and require final unique-parent coverage of 1.0.

## HPC Commands

From the repository root:

```bash
mkdir -p logs
sbatch scripts/slurm/train_mutagenicity_ppo_stable_smoke.sh
```

After smoke audit succeeds:

```bash
sbatch scripts/slurm/train_mutagenicity_ppo_stable_full.sh
```

The wrappers use `PROJECT_ROOT`, then `SLURM_SUBMIT_DIR`, then `git
rev-parse` to locate the repository. They never derive the root from the
spooled Slurm script path.

## Outputs

Smoke writes `outputs/hpc/mutagenicity/ppo_stable_v1_smoke`; full writes
`outputs/hpc/mutagenicity/ppo_stable_v1`. Each run includes resolved config,
dataset/model/teacher audits, parent coverage, update and validation metrics,
candidate pool, checkpoints, best-checkpoint metadata, and a training report.
