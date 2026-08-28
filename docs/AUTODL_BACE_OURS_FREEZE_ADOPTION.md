# AutoDL BACE Ours frozen-cell adoption

This route adopts one already-frozen BACE Ours paper cell without re-running
science.  It never opens the raw test split, invokes the GINE, regenerates
candidates, recalibrates a threshold, recomputes MolCLR distance, or copies
Figure/Table numbers.

## Frozen authority

The checked-in policy is
`configs/autodl/bace_ours_freeze_adoption_v1.json`.  It pins:

- the physical standardized `attempt-0` root;
- the separate BACE Ours raw-writer guard root;
- the exact sixteen-file source inventory;
- GINE/oracle/checkpoint, dataset, test split, MolCLR, threshold,
  temperature-scaling, and feature-schema identities;
- strict flip, source label 1, `K=20`, and Table 2 at `K=10`.

Do not edit that policy on AutoDL.  Any successor identity needs a reviewed
code commit and a fresh receipt directory.

## Preconditions

Run only from the clean reviewed AutoDL execution worktree.  The source and
writer-guard roots must be physical directories and Linux `/proc` must be
available.  The tool fails if any process holds a writable file descriptor
under either root.  It also reuses the ordinary four-by-four registry gate and
requires the source to remain exactly `FROZEN_PASS` with no reason codes.

The tool writes only a receipt.  A receipt is not a substitute for a fresh
matrix audit of the authoritative standardized root.

## Publish one fresh receipt

```bash
PY=/root/miniconda3/envs/smiles_pip118/bin/python
PROJECT=/root/autodl-tmp/worktrees/main-completion-fixed-budget-neurosed-v4-<reviewed>
MATRIX=/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/paper_matrix/four_methods_four_datasets_v1
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$MATRIX/adoptions/bace_ours_frozen_$TS"

PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT" "$PY" -I -B \
  "$PROJECT/scripts/autodl/adopt_bace_ours_frozen_cell.py" \
  adopt \
  --matrix-root "$MATRIX" \
  --output-root "$OUT"
```

The destination must not exist.  A successful terminal contains exactly:

```text
PASS
adoption_manifest.json
verification.json
```

The stdout marker and `PASS` bytes are exactly:

```text
[BACE_OURS_FREEZE_ADOPTION_PASS]
```

Publication first atomically installs the two receipt JSON files without a
terminal marker.  It reopens that final directory and repeats all source,
writer, and registry checks; only then is `PASS` created and fsynced last.  If
that post-publish verification fails, the fresh directory may remain for
diagnosis but it contains no `PASS` and is not reusable.

## Reopen the receipt

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT" "$PY" -I -B \
  "$PROJECT/scripts/autodl/adopt_bace_ours_frozen_cell.py" \
  validate \
  --output-root "$OUT"
```

Validation reopens every receipt file, rechecks the pinned source bytes, reruns
both live-writer audits, and reruns the ordinary registry candidate gate.

## Fresh matrix audit

Create a new audit root; never overwrite the existing six-cell audit.  Supply
the pinned BACE standardized root as an explicit `BACE/Ours` candidate (or as
a dedicated scan root), alongside the previously approved six roots and the
same frozen expectations.  The audit itself, not the adoption receipt, must
report `BACE/Ours = FROZEN_PASS` and exactly `7/16` before that progress can be
claimed.

The paired Slurm script is a static HPC refusal because this is an AutoDL
receipt route, not a GPU experiment.
