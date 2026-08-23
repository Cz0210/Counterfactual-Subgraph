# AutoDL BACE GlobalGCE exact stable-top-k route

This route is an opt-in replacement for the disk-heavy gSpan spill phase.  It
does not change the GlobalGCE rule objective, the frozen calibrated BACE GINE,
the minimum support, the 360-parent train cohort, or the official stable
support ordering.

## Exactness argument

The pinned GlobalGCE implementation sorts all reported frequent subgraphs by
support descending with Python's stable sort and consumes the first 20.  gSpan
projected support is anti-monotone: every descendant of a DFS code has support
no greater than its ancestor.  Once 20 earlier candidates exist with support
at least the current code's support, neither that code's later descendants nor
an equal-support later descendant can enter the stable top 20.  The opt-in
route therefore:

1. preserves root order and per-root DFS order;
2. ranks retained rows by support descending, root index ascending, and local
   report index ascending;
3. advances the native report counter for every visited report;
4. prunes only after reporting the current code and proving the stable cutoff;
5. keeps an atomic pre-root top-k snapshot, so a crash restarts the whole root
   without adopting a partial traversal;
6. fingerprints graph-list, node-insertion, and edge-traversal order rather
   than sorting away native traversal semantics;
7. atomically closes `checkpoint.json` with `stage=complete` before publishing
   terminal-PASS `exact_top_k_audit.json`, which binds the checkpoint and
   selected payload hashes.

The legacy SQLite-all-patterns route remains the default.  A new route must
pass monolithic-versus-pruned candidate/support/payload parity and crash-resume
tests before using `--gspan-exact-top-k-pruning`.

## Safety boundary

- Never enable the flag on an existing output root.  It has a distinct
  fingerprint/schema and requires a fresh root.
- Never stop the running BACE GlobalGCE v5 writer merely because its SQLite
  database is large.  A replacement is eligible only after the exact parity
  gate and a fresh immutable execution worktree are available.
- The bridge, classifier checkpoint, minimum support, top-k, train IDs, seed,
  epochs, and held-out isolation must remain identical.
- A storage-guard stop in the legacy route is recoverable evidence, not a
  license to lower `min_freq` or `top_k`.

Direct BACE frozen-GINE CLI opt-in (the historical RF pool builder deliberately
does not expose this flag):

```bash
python scripts/autodl/run_bace_baseline_gnn_route.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  globalgce-train-rules \
  --method GlobalGCE \
  --gnn-checkpoint /absolute/calibrated_bace_gine \
  --source-manifest /absolute/source_graph_manifest.jsonl \
  --native-train-csv /absolute/train.csv \
  --official-root /absolute/pinned_globalgce \
  --output-dir /absolute/fresh_output \
  --expected-parent-count 360 --seed 13 --epochs 100 \
  --top-k-native 20 --min-freq 7 --resume \
  --gspan-exact-top-k-pruning
```

The paired Slurm entrypoint is
`scripts/slurm/run_bace_baseline_gnn_route.sh`; pass the same
`globalgce-train-rules ... --gspan-exact-top-k-pruning` arguments after the
script name.  Every terminal manifest must assert `oracle_backend=gnn`,
`classifier_family=gine`, and `rf_oracle_used=false`.
