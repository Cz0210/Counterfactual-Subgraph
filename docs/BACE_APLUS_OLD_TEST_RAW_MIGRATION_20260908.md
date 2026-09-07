# BACE A+ old Ours test raw-cost leaf — 2026-09-08

The active A+ scientific driver is unchanged by this addition. Its former test
`build_index` covered only the original corrected GNN source (614 parent-model
units), not newly calculated costs in the completed Ours2607 descriptive test.
This leaf prepares an explicit two-source raw graph-cost union. It is not a new
science owner, matrix publisher, selector, model evaluator or GPU lock.

## Scope and historical failure

The known source is AutoDL:
`/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/bace/ours_reach_v2/reach-71b82bdb-20260907`.
Its complete execution does not imply full independent scientific acceptance.
The actual `independent-audit-92b9e583/audit.json` is
`BLOCKED_FIRST_INDEPENDENT_EVIDENCE_CONFLICT`, first failing at
`FULL_REACH_WITNESS_CANNOT_BE_REAPPLIED`. That code compared Python tuple/list
containers directly; commit `04ddf03e47351ce7da59b334ad2cc6e03551d7e3` changes it
to canonical JSON comparison. The follow-up audit was deferred by the direction
change. No later full-audit PASS is assumed, and no old file is rewritten.

The known full-pool witness failure is *not* silently waived for a new whole-result
PASS. After the actual new freeze, the new leaf independently checks each finite
raw-cost source row: canonical candidate, original match index/atoms, valid
deletion and connected sanitized residual, original oracle identity, parent
self-hash, completed test binding, source graph schema/MolCLR/numerical contract
and four unchanged kernel source blobs. It does not adopt reported metrics,
predictions, flip masks or selected minima. Unknown historical audit failures or
actual raw graph/value conflicts stop with the first explicit error. The extra
full-pool negative/witness diagnostic is not a distance source.

## API (no migration executed before a supplied new freeze)

`src.experiments.bace_gin_reach_test_raw` exposes:

- `validate_aplus_freeze(freeze, spec=..., evidence_root=...)`: reads the actual
  copied new `contract.json`; calls the root driver's dynamic `require_freeze`;
  verifies self-hashes, spec, 66 calibration identities, three frozen ordered
  length20 sequences and the retained original-control order. The third control
  name is not hard-coded, allowing the separately frozen supplemented A+ pool.
- `export_test_raw(source, output, repo=..., new_freeze_path=...,
  new_freeze_sha=..., validate_new_freeze=...)`: AutoDL completed-source migration.
- `union_test_indexes(original_descriptor, ours_descriptor, output, same gate
  kwargs)`: HPC union of the newly freeze-bound original614 index and transferred
  Ours index. Both source costs/provenances are retained; conflicting numerical
  values for one exact directional graph key are rejected.

The callback must return the actual `ACTUAL_A_PLUS_FREEZE_VERIFIED` receipt, not
`True` or a preconfigured timestamp. It runs before any historical test receipt,
parent record or old index is opened. The output implements the existing
`bace_reach_v2_raw_graph_cost_adoption_v1` schema consumed by
`VerifiedRawGraphDistance`; the new evaluation still computes its own GIN
strict flips and minimum over its own legal matches.

## Explicit bindings

Export binding schema: `completed_ours2607_test_raw_source_v1`.
It contains `campaign`, `kernel_source_commit`, `independent_audit: {path,sha256}` and
`documents: {role: {path,sha256}}`. Required roles are exactly the leaf's
`SOURCE_FILES`: search contract, candidate freeze, selector freeze, train gate,
final binding, execution audit, raw reuse receipt and terminal. Their real file
SHA values must be captured when the approved freeze-bound migration is staged;
unresolved or invented receipt digests are not deployable bindings.

The final binding already pins the original test CSV, portable raw contract and
old selected20 source. Parent records are read once and newly inventoried with
file/self SHA. Original model weights and full packages are not rehashed.
Union binding has `original_index` and `ours_index`, each `{path,sha256}`.

`kernel_source_commit` identifies only the four unchanged numerical-kernel
source blobs, compared with the existing raw-index proof. It is not the
historical search or closeout worker execution commit. Those identities remain
in their own receipts; an absent worker commit stays unknown.

Thin CLI:
`scripts/experiments/migrate_bace_gin_reach_test_raw.py --help`.
It requires `--config`, `--action export|union`, `--binding`, `--binding-sha`,
the actual copied `--experiment-spec` and **file** `--experiment-spec-sha`,
`--evidence-root`, actual new `--new-freeze-sha`, and a fresh `--output`.
Paired `scripts/slurm/migrate_bace_gin_reach_test_raw.sh` is CPU-only
(2 CPU / 4 GiB / 30 min): explicit exception to the repository's GPU training
default because this work performs no oracle inference or OT. Submit with
`--chdir` bound to the immutable deployed leaf worktree; it does not use a
mutable global checkout or alter any conda environment.

## Delivery state

Code and synthetic/RDKit fixture tests only. No old test-record migration,
model inference, OT computation, Slurm submission or source transfer was run
for this leaf. Deployment/real export is waiting for the root-provided new A+
freeze and completed source descriptors. A narrow raw-record audit PASS from
this tool will not mean the old or new whole experiment has final acceptance.
