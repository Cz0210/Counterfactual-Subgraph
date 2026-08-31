# TasteMolNet T8 deadline recovery managed-v2 adoption

This dataset-specific adapter converts only a real, fresh, fixed 25-epoch T8
deadline recovery PASS into the existing managed-execution-v2 terminal shape
required by T13. It never runs, copies, repairs, or changes GlobalGCE science.

## Accepted source

`scripts/autodl/adopt_tastemolnet_t8_deadline_v2.py` reopens both physical
roots produced by `run_tastemolnet_t8_deadline.py`:

- the terminal output containing exactly `science.json`, `manifest.json`,
  `gate.json`, `output_hashes.json`, and `PASS`;
- the private state tree containing the two native branch checkpoints and
  official startup/import evidence.

The adapter independently reruns the deadline preflight against the supplied
T3/T4 roots, frozen GINE checkpoint, prepared train split, and pinned official
GlobalGCE source. It reconstructs the complete deadline manifest and gate and
requires exact equality. The recovery attempt UUID must be fresh relative to
the recorded failed source UUID, and the only accepted recovery configuration
is the tracked 25-epoch contract. Validation, calibration, test, RF, heuristic
fallback, data redistribution, and GNN ablation must all remain absent.

## Two-process publication

Mode `run` starts two isolated Python processes in order:

1. `worker` retains and revalidates the deadline terminal/state closure, then
   writes only managed raw evidence, worker exit, and `SEALED.json` under a
   fresh managed UUID;
2. `verifier` independently reopens the SEALED tree and both deadline roots,
   repeats every source check, then atomically publishes the outer managed-v2
   `verification.json`, `gate.json`, and `PASS` into a fresh final path.

The nested verification schema is
`tastemolnet_t8_independent_verification_v2`, so the final publication is
consumed without a T13 code change by
`tastemolnet_globalgce_full.validate_t8_pass()`. The verifier performs that
consumer reopen before reporting success.

## AutoDL invocation

All UUIDs and roots below must be fresh except the immutable failed-source
UUID and existing input authorities:

```bash
export RUN_GNN_ABLATION=0
export T8_DEADLINE_OUTPUT_ROOT=/absolute/fresh/deadline-output
export T8_DEADLINE_STATE_ROOT=/absolute/fresh/deadline-state
export T8_DEADLINE_ATTEMPT_ID=<fresh-recovery-uuidv4>
export T8_RECOVERY_SOURCE_ATTEMPT_ID=4376be2b-42de-46d4-a3c6-ad291dd3f9f0
export TASTEMOLNET_T3_OUTPUT=/absolute/t3-pass
export TASTEMOLNET_T4_OUTPUT=/absolute/t4-pass
export TASTEMOLNET_GNN_CHECKPOINT=/absolute/t3-checkpoint
export TASTEMOLNET_TRAIN_CSV=/absolute/prepared/train.csv
export TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT=/absolute/pinned-GlobalGCE
export T8_ADOPTION_STAGE_ROOT=/absolute/existing-managed-stage
export T8_ADOPTION_FINAL_PATH=/absolute/fresh/managed-v2-final
export T8_ADOPTION_MANAGED_ATTEMPT_ID=<fresh-managed-uuidv4>
export T8_ADOPTION_RUN_ID=<controller-run-id>

sbatch scripts/slurm/adopt_tastemolnet_t8_deadline_v2.sh
```

The adapter itself is CPU-only; the paired Slurm file retains the repository's
mandatory A800 script baseline. A persistent controller may invoke the Python
entrypoint directly after the deadline process exits, avoiding a second GPU
reservation.

## Focused verification

```bash
conda run -n smiles_local python -m pytest -q \
  tests/autodl/test_tastemolnet_t8_deadline_managed_v2.py \
  tests/autodl/test_tastemolnet_t8_deadline.py \
  tests/baselines/test_tastemolnet_globalgce_full.py
python -m compileall src scripts
git diff --check
bash -n scripts/slurm/adopt_tastemolnet_t8_deadline_v2.sh
```
