# Taste GCFExplainer NeuroSED: official semantics with a fixed pair budget

Status: implementation and offline audit only. No real GED label, benchmark
PASS, pair-budget selection PASS, NeuroSED checkpoint, or T7 result is claimed
by this document.

This protocol supersedes the own-parent research adaptation described in
`TASTE_GCF_NEUROSED_PROTOCOL.md` for any production Taste GCFExplainer run.
That adaptation remains useful only as a named negative control. It cannot be
used to satisfy the official-semantics gate.

## 1. Scientific boundary

The following are retained official semantics:

- the query and target are independently selected graphs from one admitted
  split;
- the query is an official-topology sampled subgraph of its query-source
  graph, not a subgraph paired with its own parent as the target;
- interval labels are returned by real `pyged.sed` backed by GEDLIB;
- the training objective consumes the lower and upper bounds separately;
- training and validation follow the official batch-interleaved selector;
- GCF inference is directed from the generated candidate query to the original
  input target.

The only project resource-control extension is a deterministic finite number
of independent pairs. It replaces neither the pair roles nor the label
backend. In particular, it is not an upstream GREED default and it is not an
exhaustive `train x train` product.

These are prohibited:

- parent-to-own-subgraph or own-subgraph-to-parent training shortcuts;
- graph-class labels, a classifier, a neural proxy, deletion counts, or an
  average of the bounds as GED supervision;
- assigning a label to a timeout or GEDLIB error;
- reversing the runtime GCF distance direction;
- emitting a PASS marker because code compiled or a dependency was absent.

## 2. Pinned source authority

The offline audit used these pre-provisioned source trees:

| Authority | Immutable identity |
| --- | --- |
| GREED | `1c756f49625abb62c9f6de5b0059876a4c7499c1` |
| GREED experiments | `e85423dc943fda1979811e7449846efffec2a1e1` |
| GREED `neuro/datasets.py` | `aa1bab19394b2fcad4d6f1c45c5206f0485cc098dbd4742bf1396d229c0fa1ad` |
| GREED `neuro/train.py` | `8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28` |
| GREED `neuro/models.py` | `c5653dd9eeec1add8d6ae6253c30908df5ab8962ea0d9f9a6f25d32c393e0e70` |
| GREED `neuro/config.py` | `cb34333a497c9627ee2f728cf45734162b78a6924e596b7cde88ef2788f66050` |
| GREED `pyged/src/pyged.cpp` | `55b35f952ea4070fad430d0911d29bfca21b4e10926e9bd7d56d2515d6499b16` |
| GREED `pyged/CMakeLists.txt` | `597f2f23252b0681d8de0d4c48cd4d10fad59d5c9130262fe2e7d3753737a010` |
| GREED-expts AIDS training notebook | `49a7bc0095d879bf49454cd6c18e42bb687c149a32e425b59c2acbe6c2df0114` |
| vendored GCF `neurosed/models.py` | `8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60` |
| vendored GCF `distance.py` | `d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3` |

`official_neurosed_commit` is the pinned GREED commit above. The vendored GCF
directory has no independent Git metadata, so `official_gcf_commit` is
currently `UNAVAILABLE_FROM_VENDORED_SNAPSHOT`. Its critical file hashes are
the available project authority. A future release must either recover and pin
the upstream GCF commit or explicitly approve and authenticate the complete
vendored snapshot; this implementation does not invent a commit.

The audited official pair builder is:

```text
neuro.datasets.make_inner_dataset(
  graphs, n_pairs, n_hops_query, trav_prob_query,
  node_lim_query=None, n_hops_target=None, targets=None
)
```

It calls `make_queries(targets, ...)` and independently calls
`random.choices(targets, k=n_pairs)`, then sends the ordered query-target rows
to `inner_sed`. The fixed sampler preserves that role separation while adding
a seed-7 deterministic budget, size strata, and same/cross-class diversity
diagnostics. Class is never a distance target.

## 3. Real pyged/GEDLIB contract

The pinned wrapper uses method `f2` and an argument of the form
`--threads <n> --time-limit 1`. The project keeps `f2`; it does not switch to a
different method to avoid a dependency. Its SED costs are directional:

| Edit | Cost |
| --- | ---: |
| node insertion | 0 |
| node deletion | 1 |
| node relabel mismatch | 1 |
| edge insertion | 0 |
| edge deletion | 1 |
| edge relabel | 0 |

Consequently `(q, t)` and `(t, q)` are not interchangeable. A symmetric cache
is forbidden. Any later cache key must bind canonical query graph hash,
canonical target graph hash, GEDLIB configuration hash, feature-schema hash,
and the explicit direction.

The isolated builder accepts only already-provisioned source and dependencies.
It authenticates GREED and GREED-expts, authenticates an operator-supplied
full GEDLIB commit, creates a fresh build root, disables only the unused Gurobi
compile/link branch, retains F2, compiles one worker, imports only the produced
module, and verifies zero insertion versus positive deletion on tiny graphs.
It records Python/compiler/CMake versions and build flags. It never runs
`pip`, `conda`, `git clone`, or any network command, and never mutates
`smiles_pip118`.

The source snapshots available during the local audit contain no GEDLIB
dependency tree or pybind11 CMake package. The honest local result is therefore
`BLOCKED_GEDLIB_BUILD`, with a null marker. A real build PASS is possible only
after AutoDL has a reviewed, pre-provisioned GEDLIB checkout and pybind11 CMake
directory and the exact GEDLIB commit is supplied. No remote science was
started by this change.

## 4. Deterministic pair universe

Training pairs use only Taste train graphs for both roles. Validation pairs use
only Taste validation graphs for both roles. Every row enforces
`query_graph_id != target_graph_id`. Calibration and test are not accepted
split values.

The pair builder reads one normalized absolute, non-symlink CSV through a held
descriptor, verifies its SHA-256 before use, reconstructs the reviewed feature
schema, and writes unlabeled metadata. Each row records the required graph IDs,
split, sizes, scaffolds, seeds, stratum, and reconstruction hashes. The sampler
does not materialize a Cartesian product. A requested training/validation
budget must additionally have `ceil(1.10 * budget)` deterministic candidates;
successful rows are taken in sampler order, with no GED-result-based choice.

The 100-, 500-, and 1000-pair benchmark cohorts are disjoint slices of one
deterministic 1600-pair inventory. File hashes and the actual ordered pair-ID
inventories are carried into the reports so a summary cannot prove
disjointness from hashes alone.

## 5. Benchmark and worker selection

Each benchmark calls the authenticated isolated `pyged` module and real F2
backend. It writes:

```text
gedlib_benchmark_100.json
gedlib_benchmark_500.json
gedlib_benchmark_1000.json
gedlib_benchmark_summary.json
```

Raw observations distinguish `SUCCESS`, `TIMEOUT`, and `GEDLIB_ERROR`.
Timeout/error rows have null bounds and never enter training. Reports include
wall time, seconds/pair, pairs/hour, p50/p90/p95/p99 latency, timeout/failure
counts, child CPU time/utilization, maximum child RSS, load average, iowait,
and node/edge-count correlations.

Worker trials use 1, 2, 4, and 8 workers, subject to physical-core availability,
with at least 100 fresh pairs per trial. The selected count is the highest
healthy throughput after host-load, iowait, BACE legacy, and AIDS exact
contention gates. All GED workers set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `TOKENIZERS_PARALLELISM=false`. Protected jobs
are never stopped by this route.

## 6. Fixed-budget planner

Only train budgets 5000, 10000, and 20000 are legal. Their validation budgets
are respectively 1000, 2000, and 4000. Using the real 1000-pair report, the
planner computes:

```text
(train_pairs + validation_pairs) * p95_seconds_per_pair
-------------------------------------------------------- * 1.25
                    selected_workers
```

It chooses the largest tier whose projected label time is no more than 24
hours, timeout rate is no more than 0.05, and whose disk-reservation and CPU-
contention gates passed. If even 5000 fails, the result is
`BLOCKED_GEDLIB_THROUGHPUT` plus all three ETAs. It cannot select an unapproved
budget or fall back to own-parent or approximate labels.

The disk gate must reserve compact/columnar label storage while retaining at
least `MIN_FREE_AFTER_RESERVATIONS_GB=100`. The benchmark JSONL is diagnostic;
a full label pipeline must use Parquet/Arrow or NumPy binary rather than a
large per-pair JSON debug dump.

## 7. Remaining release blockers

This phase deliberately stops before scientific training. The following still
must be implemented and independently verified before any official fixed-
budget PASS or T7 run:

1. provision and pin GEDLIB plus pybind11 on AutoDL, then obtain the real build
   smoke and real 100/500/1000 reports;
2. implement compact directional label cache/output, reserve replacement, and
   official lower/upper/exact-bound representation;
3. compare a small fixture field-by-field with the pinned official builder;
4. bind train/validation split-isolation evidence against calibration/test
   membership without loading held-out scientific payloads;
5. replace the epoch-level research selector with GREED's batch-interleaved
   validation/checkpoint/stopping contract;
6. train and independently verify a fresh checkpoint and its official GCF
   loader/direction under the managed controller/resource gates.

Until all six complete, do not emit
`[TASTE_NEUROSED_PAIR_BUILDER_PASS]`,
`[TASTE_NEUROSED_OFFICIAL_FIXED_BUDGET_PASS]`, or any T7/T12 PASS.
