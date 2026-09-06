# Mut sealed-result K10 publication projection (2026-09-07)

The independent Mut ComRecGC scientific execution at
`fbefa4caff172453d42afd90f8518bc7e8bddf47` completed its unified evaluation,
full gate and freeze. Its native Figure 4 exporter selected maximum K=20, while
the existing approved Mut/Ours, GCFExplainer and GlobalGCE publication protocol
uses K=10. The native K20 curve must not be relabeled as matched K10.

The authoritative K10/601-point reference is enforced by
`src/eval/user_approved_frozen_v4.py` (source-bundle validation and evaluation
manifest) and the matched-K10 path in `src/eval/am_legacy_standardization.py`.
This publication repair neither selects a new grid nor uses Test results to
choose a budget or threshold.

## Narrow supported operation

`src/eval/mut_registry_k10_projection.py` first requires the existing full
independent terminal validator, including the explicit typed startup-repair
receipt. It then reads only sealed parent-distance summaries, prefix/table
aggregates, small manifest closures and the original approved reference.
It does not load a classifier, molecular dataset, checkpoint, OT pair store,
candidate selector, or chemistry implementation.

The fresh projection contains `projection.json` and `standardized/`. Figure 3,
Table 2 K10, prefix metrics and parent summaries are byte-for-byte copies.
Only Figure 4 is deterministically re-aggregated at K10 from the 217 sealed
parent-best distances. Its 601 threshold strings are copied exactly from the
approved original Mut/Ours Figure 4, whose ordered-string SHA256 is
`817968eb0260902205f9faedd634b7b6872fef8b12c46772d84423c9336102ae`.

The native frozen threshold contract and its SHA-bound predecessor must match
the same predeclared uniform grid, theta=0.05 and cap=0.0535. The verifier checks
every K10 parent decision against both historical floating-point serializations:
any threshold crossing rejects the projection. A numerical tolerance or equal
aggregate count is not sufficient. The legacy nested contract hash remains an
explicit unchanged historical label; it is not claimed to be a recomputed hash.

Conditional costs stay blank only where the sealed parent records prove zero
strict-flip parents and the authoritative prefix reports zero coverage. No zero
cost is imputed and no conditional cost is replaced by a capped cost. The ordinary
registry's narrow schema handling invokes full typed projection replay before
accepting these missing values; other cells retain their existing behavior.
Destination-distribution fields not present in the native export remain explicitly
unavailable, without invented counts. RF family and raw completeness come from
the validated outer terminal, with explicit hash-bound provenance.

All original files, including native K20 and the historical `FAILED.json`, remain
unchanged. `projection.json` records original/new file hashes, the actual preserved
publication-driver checkout, commit and module hash separately from scientific
commit fbef. A later reader's unrelated HEAD is not mistaken for the producer.
The flags truthfully state `aggregate_reexport=true`,
`figure_table_recomputed=true`, and inference/OT/selector/Test-dataset reruns
false. The parent cohort, original K20 aggregate, K10 aggregate, copied bytes,
reference protocol, producer identity and complete projected inventory are
reopened on validation. Missing, conflicting or additional files fail closed.

## Commands and publication binding

```bash
python -I -B /absolute/new-checkout/scripts/autodl/project_mut_registry_k10.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --terminal-root /absolute/original-mut-terminal \
  --reference-standardized-root /absolute/approved-mut-ours/standardized \
  --startup-repair-receipt /absolute/repair/runtime/control_adapter_receipt.json \
  --output-root /absolute/fresh-mut-k10-projection --proc-root /proc
```

Creation fsyncs the new files/directories and runs the ordinary registry gate.
A failed creation is not publishable; any partial output is preserved for
diagnosis, not deleted or reused. No original seal is overwritten.

`publish_mut_successor_v1.py` and `append_non_taste_matrix_authority.py` accept
`--registry-projection /absolute/fresh-mut-k10-projection` (the **root**, not
`projection.json`). The original completed stage-02 export receipt and original
terminal root remain required. Only the ordinary registry's standardized root is
redirected after strict validation; there is no reason-code suppression or identity
waiver. The canonical publication locator retains the original terminal root and
additionally identifies the projected standardized root. The matrix authority
row therefore points to matched K10, not the native K20 export.

Paired Slurm scripts retain the repository's A800/runtime contract, although this
projection itself is a CPU-only summary operation. No Slurm or remote scientific
execution is necessary for the repair.

Focused tests exercise exact source preservation, the real ordinary registry,
wrong K/grid/cohort/coverage and source-hash rejection, threshold-crossing
rejection, conditional-null proof, producer/consumer identity separation and CLI
propagation. A read-only replay against the actual sealed 2026-09-07 Mut summaries
passed: K10 Figure 4 SHA256
`c1f10fbc0a0502fe53ba172d50af35194430c64103794992955cf3318990fc30`,
601 rows, final threshold 0.0535 coverage 19/217, and undefined conditional-cost
prefixes exactly 1, 2 and 3. This replay wrote no remote artifacts.

## Approved legacy missing identities are not contradictory hashes

The first real publication of the validated b8d11337 projection reached the
cross-method gate and was rejected for oracle, dataset, split and MolCLR hash
conflicts. Read-only inspection found no pair of different nonempty hashes:
all three approved legacy Mut methods have these four identities explicitly
unavailable under the existing checksum-validated `USER_APPROVED_FROZEN_V4`
exception. The new ComRecGC cell has genuine, independently validated identities.

The cross-method check now excludes an empty field from comparison only for an
already validated AIDS/Mut Ours, GCFExplainer or GlobalGCE `ADOPTABLE_PASS` row,
with that exact exception, its identity-unavailable status, valid exception hash
and the corresponding missing-field waiver. It neither rewrites rows nor fills
missing hashes from ComRecGC. Nonempty hash conflicts, unapproved missing fields,
other datasets, other methods, thresholds and metric contracts retain fail-closed
behavior. This matches the existing append compatibility receipt's explicit
`reference_unavailable_fields`; it does not assert cross-method identity equality.
