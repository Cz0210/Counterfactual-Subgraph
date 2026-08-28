# AutoDL TasteMolNet T9 COMRECGC smoke

## Status

The implementation candidate is intentionally **release disabled**.  The
tracked config has `release_enabled=false`, every mutable release pin is
`null`, and the AutoDL wrapper exits 78 before loading `common.sh`, querying a
GPU, opening a model, or creating an output.  Do not flip either gate until
the managed-execution core, T9 result dispatch, continuation-controller task
manifest, final clean commit/tree, storage authority, exact private output
parent, and fresh T2/T3/T4 predecessors have all received independent review.

The paired Slurm script is a static AutoDL-only refusal.  Its unreachable
command is CLI documentation and is not an HPC execution route.

## Scientific contract

T9 is a bounded native COMRECGC smoke, not a binary compatibility run and not
a paper result.

- dataset: TasteMolNet, three classes, source `Sweet=1`;
- importance used by the native random walk: `1 - p(Sweet)`;
- counterfactual membership: `argmax([p0,p1,p2]) != 1`;
- graph identity: SHA-256 of the canonical, attributed native graph;
- GINE hidden vectors: distance/recourse vectors only, never identity;
- repeated canonical graphs: the first GINE probability/hidden row is the
  canonical row reused by the official walk; every later real rescore must
  preserve validity, argmax/candidate semantics, shape, dtype, and remain
  within `rtol=1e-5, atol=1e-7`, otherwise the run fails;
- lineage: retained separately and excluded from graph identity;
- walk: serial official stateful heads, `M=500`, midpoint checkpoint after
  fully completed step 250, exact persisted reload (including the canonical
  GINE row cache), then steps 251–500;
- source cohort: first eight unique GINE-predicted Sweet graphs selected from
  a frozen train-only pool of 64;
- native parameters: heads 5, candidate capacity 2048, sample size 10000,
  teleport 0.1, theta 0.1, delta 0.02, DBSCAN min samples 3, common-recourse
  size 5, seed 7;
- recourse pairs retain the official `torch.where([source,candidate])`
  source-major order through DBSCAN, medoid ties, coverage, and greedy
  selection; selected cluster IDs are unique;
- full budget `M=50000` remains explicitly not run;
- RF, validation, calibration payload, test, dataset redistribution, and
  paper-result eligibility are all false.

Every numeric/type boundary is strict.  A smaller source pool, fewer heads,
easier clustering threshold, different seed, bool-as-int, NaN/Inf, embedding
hash, Python `hash()`, or parent metadata identity is rejected.

## Official source authority

The only upstream authority is COMRECGC commit
`122f9341a360e9f06bb58a2f5823bb596021f6bf`.  Seven executable source files
have fixed reviewed SHA-256 values in
`src.baselines.comrecgc.held_upstream.OFFICIAL_SOURCE_SHA256`.

At runtime each file is opened once with every ancestor retained and
`O_NOFOLLOW`.  Python loads that exact file descriptor through
`/proc/self/fd/N`; the external checkout is never inserted into `sys.path`.
The descriptors remain held through the scientific run and the terminal
commit callback.  A caller-provided self-consistent but nonreviewed file hash
mapping is rejected and cannot be labelled as the pinned commit.

## Predecessor and process boundary

The final runner must acquire
`hold_injected_active_execution(expected_receipt_kind="taste_t9_gpu2_v1",
expected_task_id="tastemolnet_t9_comrecgc_smoke",
expected_stage="T9_COMRECGC_SMOKE", ...)` before any scientific input.  It
retains that holder until its last fallible PASS callback.  The route is
exclusive physical GPU2 with strict validator `taste_t9_v1`.

T2 is consumed only through the receipt-only exact-five-file holder using the
adoption root plus gate/receipt/source-evidence SHA.  T3 and T4 are retained
through their public held stage consumers and must bind the same T2 receipt
and exact frozen checkpoint.  The checkpoint bundle and train CSV remain
descriptor-held; validation/calibration/test rows are never opened.

The task's managed predecessor list is T2, T3, T4.  GPU2 lane ordering after
T8 is a continuation-controller scheduling dependency, not a substitute
scientific authority and not a reason to rewrite the T9 scientific gate.

## Private terminal output

The only persistent output files are:

```text
input_hashes.json
state.json
manifest.json
comrecgc_smoke.json
gate.json
output_hashes.json
PASS
```

No SMILES, molecule ID, graph payload, checkpoint payload, or row-level
prediction is persisted.  `PASS` contains exactly
`[TASTE_T9_COMRECGC_SMOKE_PASS]\n`. The structured `marker` fields and stdout
use the identical already-bracketed string without a newline; code must not
add or remove a second pair of brackets. Publication uses the retained output tree,
prevalidates the complete prepared image with the same public strict consumer,
revalidates managed ACTIVE plus every immutable input inside the final
callback, and exposes the exact prepared marker inode with no replacement.
After PASS, only nonthrowing descriptor cleanup and return zero are allowed;
the managed parent then performs an independent held reopen and publishes
COMPLETION while its execution/GPU locks remain held.

## Future release invocation

The reviewed continuation controller, not a shell user, supplies all manifest
and predecessor values to `scripts/autodl/run_tastemolnet_comrecgc_smoke.sh`.
The scientific CLI remains:

```bash
python -B scripts/run_tastemolnet_comrecgc_smoke.py \
  --config configs/hpc.yaml \
  --stage T9_COMRECGC_SMOKE \
  --output-dir "$TASTEMOLNET_T9_OUTPUT" \
  --t2-adoption-root "$TASTEMOLNET_T2_ADOPTION_ROOT" \
  --t2-adoption-gate-sha256 "$TASTEMOLNET_T2_ADOPTION_GATE_SHA256" \
  --t2-adoption-receipt-sha256 "$TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256" \
  --t2-source-evidence-sha256 "$TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256" \
  --t3-output-root "$TASTEMOLNET_T3_OUTPUT_ROOT" \
  --t4-output-root "$TASTEMOLNET_T4_OUTPUT_ROOT" \
  --checkpoint-dir "$TASTEMOLNET_T2_BUNDLE" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --official-root "$COMRECGC_OFFICIAL_ROOT" \
  --set inference.fallback_to_heuristic=false
```

`--validate-only` may inspect an already published private output while the
science release remains disabled.  It does not launch or modify science.
