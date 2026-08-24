# AIDS ComRecGC Production-Subset Exact Equivalence Audit

## Scope

This CPU-only AutoDL audit is a release gate for the paper-faithful AIDS
ComRecGC route. It does not launch DBSCAN on the full production store and a
passing result must never be reported as a full-production DBSCAN PASS.

The command accepts only a terminal, SHA-pinned `close_pair_contract.json` and
the SHA-pinned physical `(parent, candidate)` pair array. Validation reopens
the close-view authority, its physical vectors, normalized distances, bitmap,
pair-semantics contract, and source pair-store manifest. A missing authority,
path/hash mismatch, source-stat drift, symlink, or pair-axis mismatch fails
before `PASS` is written.

## Subsets and comparisons

Five deterministic induced inputs are materialized in original logical-row
order:

1. first rows;
2. seeded random rows without replacement;
3. rows nearest a seeded recourse-vector pivot;
4. rows farthest from that pivot;
5. rows nearest the inclusive theta boundary by frozen normalized distance.

Here “hash-closed” means that every induced input byte and its authority are
cryptographically closed. These samples are not claimed to be closed under
the full production epsilon-neighbor graph; outside rows may have epsilon
edges into them.

Each subset records logical indices, physical rows, pair rows, vectors, every
SHA256, seed, pivot, selection semantics, and partition canonicalization.
`sklearn.cluster.DBSCAN(metric=euclidean, eps=0.02, min_samples=3)` is compared
with the general exact external-memory engine for core mask, noise mask, and
partition. The exact adaptive all-core certificate is attempted with general
fallback disabled; an inapplicable certificate is recorded as inconclusive,
while an applicable certificate must reproduce the same result.

Post-clustering comparison includes float32/stable-float64 centroids, strict
`distance < delta` membership, strict `centroid_norm < theta` eligibility,
covered parent sets, counterfactual sets, and greedy selection with canonical
cluster-ID tie-breaking.

## AutoDL command

Run from an immutable AutoDL execution checkout and a fresh output root:

```bash
python scripts/baselines/comrecgc/audit_aids_production_subsets.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --close-pair-contract /absolute/close_pair_contract.json \
  --expected-close-pair-contract-sha256 <sha256> \
  --physical-pairs /absolute/pair_indices.npy \
  --expected-physical-pairs-sha256 <sha256> \
  --expected-sklearn-version 1.7.2 \
  --output-dir /absolute/fresh/production_subset_audit
```

The paired Slurm file exists only to keep repository entrypoint inventory in
sync. It intentionally exits with a configuration error: this route is
AutoDL-only and must not be submitted to HPC.

Terminal success publishes `production_subset_equivalence.json`, five
hash-closed subset audit files, and `PASS` last. The terminal manifest always
contains `full_production_dbscan_equivalence_claimed=false`.
