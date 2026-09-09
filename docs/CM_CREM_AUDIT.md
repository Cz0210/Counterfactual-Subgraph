# CM-CReM BACE independent audit

This module is deliberately BACE-specific. It is not another controller, cache,
generation route, or portable-PASS shortcut. The existing independent `audit`
Slurm job calls two functions, in order:

```python
from src.baselines.cm_crem_audit import audit_bace_run, independent_spotcheck

# Existing record reconciliation writes this first, without claiming PASS.
experiment.put("test_evaluation.json", reconciled_evaluation.to_dict())
provenance = audit_bace_run(experiment.spec, experiment.root)
experiment.put("audit/provenance_review.json", provenance)
spotcheck = independent_spotcheck(experiment.spec, experiment.root, provenance)
experiment.put("audit/independent_spotcheck.json", spotcheck)
```

Every exception is a blocking audit failure, never an empty candidate set, an
infinite distance, or accepted science. The producer driver remains responsible
for writing the final audit only after these functions actually succeed. Export
visual QA and portable transfer/import validation remain separate gates.

## Saved-record verification

`audit_bace_run(spec, root, *, fixture=False)` reads the authoritative 386 train,
66 calibration, and 141 test parents through their pinned file hashes and ID
order. It validates the selection freeze before opening test scientific records.
The audit requires the actual official-database receipt, generation-unit database
content SHA, every original-train prediction, all source-parent attributions,
legal generation terminals/budgets, per-parent filter outputs, deterministic
train-only <=2000 candidate library, encoder/raw-OT records and producer identity,
exact calibration re-selection, and fixed-denominator test reduction.

The pilot's 64 nonself distances are checked against its saved actual encodings;
a `real_nonself_wnode_pairs` count alone is insufficient. Raw pair keys exclude
distance values by design. Each scientific input must additionally be bound by
the driver's fsynced `producer_receipts/JOB.jsonl` byte hash. Missing journals,
missing raw parent encodings, rewritten distances, changed selected columns,
NaN, and unresolved engineering failures are rejected.

Execution commits must match the resolved execution commit or the explicit
`execution.accepted_producer_commits` allowlist. This permits a reviewed older
immutable pilot deployment; matching science hash alone does not authorize an
unknown implementation. Producers must have terminal scheduler state and must
not be the current audit job. A successfully committed legal parent unit from
a subsequently FAILED/TIMEOUT job remains reusable; it is not regenerated just
to make the outer scheduler state COMPLETED. Active jobs are not closed evidence.

The result is `BACE_SAVED_RECORD_PROVENANCE_VERIFIED`, with
`scientific_pass_claimed=false`. Its first outstanding check is explicitly the
fixed-identity original-model/raw-OT recomputation, not a generic pending label.

## Independent scientific spotcheck

`independent_spotcheck(spec, root, provenance, *, fixture=False,
fixture_backend=None)` checks the exact saved evaluation file against the
reconciled result, then orders existing legal nonself frozen test pairs by:

```text
SHA256([science_hash, "audit_spotcheck_v1", parent_id, candidate_id])
```

The first at most 8 pairs are selected. Distance values do not enter that order;
test columns cannot extend beyond the <=20 frozen candidates. For each selected
pair the function reconstructs inputs from original SMILES, reruns original GINE
and its original temperature, recomputes original MolCLR node encodings without
using the saved node cache, and calls the original exact-WNode function anew.
Repeated OT is explicitly `audit_spotcheck_count`, never reported as production
cache reuse. No new generation, calibration selection, test cohort, or science
parameter is introduced.

Encoder and distance comparisons are exact. Forward comparisons use only the
already resolved original forward tolerance (zero by default; a nonzero value
requires the existing numerical-contract SHA binding). The auditor never chooses
or enlarges a tolerance after seeing a difference. A legitimate numerical mismatch
therefore remains an audit failure to diagnose; it is not silently accepted.

With no legal pair, the function verifies that no nonzero legal-pair claim exists
and independently rereads fixed-hash source/target prediction examples from the
already audited train/test ledgers (and frozen targets, when present). It records
whether the library is genuinely empty or the fixed test cohort has no source
parents. Empty sampling cannot certify nonzero results.

Production success is `CM_CREM_INDEPENDENT_SPOTCHECK_PASS`: this means the bounded
scientific spotcheck passed, not that the entire portable package passed. It binds
the implementation commit/source SHA, provenance SHA, exact source evaluation and
selection-freeze file hashes, sample identities and checked record count.
`scientific_pass_claimed` remains false at this intermediate gate; the final driver
and portable acceptance path own the combined final status.

## Tests and operational scope

Tests inject a backend only with `fixture=True`, emit visibly FIXTURE statuses,
and never load a real model, run real OT/generation, or create a final scientific
PASS directory. Production rejects injected fixture backends/evidence. Focused
tests cover full empty/nonempty synthetic provenance, raw-value byte tampering,
post-audit mutation, exact one-ULP mismatch, frozen deterministic sampling,
generation failure/timeout semantics, and genuine zero evidence.

No new CLI or Slurm script is introduced: the existing `run_cm_crem.py audit`
stage and paired `scripts/slurm/run_cm_crem.sh` remain the execution interface.
The official database GET403 is still an external generation blocker; code and
fixture tests are not a real pilot, scientific result, or portable acceptance.
