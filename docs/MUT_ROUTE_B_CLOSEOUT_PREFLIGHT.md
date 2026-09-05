# Mut Route B: conditional closeout preflight

This is a diagnostic, **not a runnable end-to-end successor or launch receipt**.
The existing A/B experiment must first produce a reopened, byte-bound genuine
scientific divergence. Engineering failure, missing logs and an incomplete
500-step comparison must not select Route B. No current owner is restarted.

Run on the immutable integration checkout with the existing environment:

```bash
python scripts/autodl/preflight_mut_route_b_closeout_v1.py \
  --config configs/hpc.yaml \
  --resource-path /autodl-fs/data/counterfactual-subgraph-runtime
```

The CLI reads source files and filesystem availability and emits JSON to
stdout. It opens no model, checkpoint, active SQLite or WAL, acquires no GPU,
and writes no output root or matrix. The paired Slurm script is intentionally
CPU-only; HPC output remains under the authorized `/share/home/u20526/czx`
submission directory. It is not an AutoDL science launcher.

## Actual scientific input gap

The pinned trace-off generation writes `frozen_payload_closure_audit.json` and
reloads `counterfactuals.pt`. With no `trace_recorder`, its selected-event input
is empty. It does not write the complete selected-action predecessor history
used by `run_mutagenicity_chemistry_audit` for exact replay and the fixed
deterministic repair. Final graph tensors and node-origin attributes do not
uniquely reconstruct that history. Old trace-on candidate lineage is not an
input for a new candidate universe.

A complete fix needs an observational causal-history producer that leaves
the trace-off algorithm state and RNG unchanged, demonstrated by the existing
500-step and checkpoint/reload gates. It must not secretly enable trace-on or
claim that a generation-only terminal closes a method cell. Until that input
exists, state remains `BLOCKED_SCIENCE_CRITICAL_LINEAGE_PRODUCER_MISSING`.

The real `run_common_recourse.py` already builds new pair chunks and exact
DBSCAN when no external source-pair/DBSCAN arguments are provided. Its identity
binds the **new** generation manifest, payload, ordered graph hashes, indices,
parents and distance checkpoint. `fresh_pair_command` constructs this CPU
command without executing it. Downstream still needs a fresh-generation
terminal validator and a narrow dispatch in the existing canonical publisher;
the historical-adoption terminal must not be impersonated.

## Compact storage estimate, not admission

For capacity 100,000, candidate batch 128 and 1,448 train parents, the upper
bound is 782 pair chunks / 1,564 npy files, rather than one file per candidate
or candidate-parent pair. Two keep-last-2 checkpoint stores plus one in-flight
publication per store are budgeted, with retention receipts and a conservative
fixed-file allowance. Known peak new inodes: 2,068. This excludes unmeasured
evaluation cache peaks and any future lineage-producer files.

The existing 100,000-free-inode guard is unchanged. At the read-only snapshot
of 2026-09-05 14:45:36 UTC, free inodes were 98,232: 1,768 below that guard,
or 3,836 below guard plus known compact new files. No files were deleted.

The worst-case candidate-parent count is 144,800,000, not the old universe's
observed pair count. Exact vector dimension and dtype are optional measured
inputs (`--vector-dimension`, `--vector-itemsize`); absent values stay unknown.
Consolidation doubles the pair-array peak while chunk files coexist. Generation
SQLite/checkpoint bytes, exact DBSCAN temporaries, chemistry and evaluation
caches remain unmeasured: this estimate cannot mark full storage admission PASS.
