# Taste GCFExplainer NeuroSED: fixed-budget non-MIP GEDLIB release route

> **2026-08-29 main-table release override.** The historical F2/Gurobi and
> 100/500/1000 plus 5k/10k/20k planning gates below remain as provenance only.
> The active route uses pinned GEDLIB non-MIP candidates
> `branch`, tested twice on the same 100 real
> independent train pairs. It requires >=95% success, identical fixed-seed
> outcomes, finite `lower <= upper`, <=10 minutes per candidate and <=30
> minutes total, then selects the fastest passing backend. Manifests must say
> `GED_LABEL_BACKEND_VARIANT=NON_MIP_GEDLIB`, `F2_BLP_USED=false`, and
> `GUROBI_USED=false`.
>
> The active fixed budget is train=5000, validation=1000, seed=7. If the real
> 100-pair canary projects more than 24 hours for all 6000 labels, the only
> allowed reduction is train=2000, validation=500 and it must be recorded as
> resource-reduced. No 10k/20k search and no separate 500/1000 benchmark
> release blockers remain. Independent split-local pairs, no
> parent/own-subgraph shortcut, no calibration/test access, official GREED
> training/checkpoint semantics, and generated-query to original-target GCF
> direction remain unchanged.
> Pinned IPFP is excluded because its default initialization uses an unseeded
> C++ `random_device` path. `anchor_aware_ged` is also excluded because it
> invokes that IPFP path internally without a deterministic initializer.
> A separate verifier must reopen every candidate's two observations files and
> benchmark reports, check their digests, and recompute bounds, success rate,
> determinism, throughput, selected backend, and budget before its receipt can
> be referenced by a model card.

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

The project extensions are a deterministic finite number of independent pairs
and the explicitly selected pinned non-MIP GEDLIB label backend. They replace
neither the pair roles nor the interval-label representation. This route is
not the complete upstream F2/BLP configuration, is not an upstream GREED
default, and is not an exhaustive `train x train` product.

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
| GEDLIB v1.0 required by pinned GREED | `120856f670e013f080b116c0be4cc6bd72fc935d` |
| GREED `neuro/datasets.py` | `aa1bab19394b2fcad4d6f1c45c5206f0485cc098dbd4742bf1396d229c0fa1ad` |
| GREED `neuro/train.py` | `8e4d425d9d63e0aa56d5a1e6e25738f511ca7b52b08ac297fcf2c1678bdf9e28` |
| GREED `neuro/models.py` | `c5653dd9eeec1add8d6ae6253c30908df5ab8962ea0d9f9a6f25d32c393e0e70` |
| GREED `neuro/config.py` | `cb34333a497c9627ee2f728cf45734162b78a6924e596b7cde88ef2788f66050` |
| GREED `pyged/src/pyged.cpp` | `55b35f952ea4070fad430d0911d29bfca21b4e10926e9bd7d56d2515d6499b16` |
| GREED `pyged/CMakeLists.txt` | `597f2f23252b0681d8de0d4c48cd4d10fad59d5c9130262fe2e7d3753737a010` |
| GREED-expts AIDS training notebook | `49a7bc0095d879bf49454cd6c18e42bb687c149a32e425b59c2acbe6c2df0114` |
| official GCFExplainer | `cc7ca30eb2026c57f20cd6afe2ee621f486fcf2e` |
| vendored GCF `neurosed/models.py` | `8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60` |
| vendored GCF `distance.py` | `d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3` |
| vendored GCF `importance.py` | `5e364634fcf6fac9c5e16b5d9dc2f53837ab67508421e5076010c1e9cdac33be` |
| vendored GCF `vrrw.py` | `89ff1a9dbb9561d33dd4fbc1bffe84e60deeb069948778b39b75dc5c93a59fce` |
| vendored GCF `summary.py` | `371ca30b9672bd17b472d261327dc343b989b52150257de8a8ce1c868389af44` |

`official_neurosed_commit` is the pinned GREED commit above. The official
GCFExplainer repository was independently checked at the full commit
`cc7ca30eb2026c57f20cd6afe2ee621f486fcf2e`. A recursive byte comparison found
that every retained file under `baselines/gcfexplainer_official/` is identical
to that commit; the vendored tree only omits upstream dataset/model artifacts.
The release gate therefore pins the exact repository URL and commit in
addition to the critical executable-file hashes above. The readiness validator
descriptor-reopens all 17 retained files, rejects symlinks and extra files or
directories, and binds the complete inventory digest
`467205d647d8a1be55f129a936ace8be48904eeb2b802e909a8c62cc6088c606`.
It rejects another repository or commit rather than trusting self-reported
model-card metadata.

The audited official pair builder is:

```text
neuro.datasets.make_inner_dataset(
  graphs, n_pairs, n_hops_query, trav_prob_query,
  node_lim_query=None, n_hops_target=None, targets=None
)
```

It calls `make_queries(targets, ...)` and independently calls
`random.choices(targets, k=n_pairs)`, then sends the ordered query-target rows
to `inner_sed`. The fixed sampler preserves that role separation with separate
seed-7 deterministic query-source and target RNG streams drawing with
replacement from the complete same-split graph sequence. It rejects an
otherwise accepted draw only when the graph IDs are equal or the query-source
cannot yield a valid official-style query. Only after the ordered pair draws
are complete does it derive size bins and same/cross-class diagnostics. Those
diagnostics never select, filter, rebalance, reroll, or order pairs and are not
part of pair identity. Class is never a distance target.

## 3. Real pyged/GEDLIB contract

The pinned upstream wrapper uses method `f2` and an argument of the form
`--threads <n> --time-limit 1`, but that path requires the unavailable Gurobi
build/runtime. The active route deliberately switches to the authenticated
GEDLIB `branch` method with `--threads 1` and records the switch in every
manifest. It never claims F2/BLP use. The retained SED edit costs are
directional:

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

The cache-policy helper first inspects the complete scalar edit-cost contract.
A synthetic symmetric contract shares forward/reverse keys only when insertion
and deletion costs match. The pinned SED contract is proven asymmetric, keeps
query and target order in the key, and sets reverse sharing to false.

The isolated builder accepts only already-provisioned source and dependencies.
It authenticates GREED and GREED-expts, authenticates an operator-supplied
GEDLIB checkout at the exact official v1.0 commit above, creates a fresh build
root, removes the wrapper's Gurobi-only F2/BLP exposure, exposes only the
deterministic `branch` candidate, compiles one worker, imports only the produced
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

### Canonical feature-schema producer

Before either split-local pair inventory is built, materialize the NeuroSED
feature schema with the dedicated AutoDL CLI:

```bash
python -B scripts/autodl/build_tastemolnet_neurosed_feature_schema.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --train-csv /absolute/private/tastemolnet/splits/train.csv \
  --expected-train-sha256 eac05f7003c37a24554aa2c22e1051edb90eb4a12f9b62ae6fd47d73efa59564 \
  --validation-csv /absolute/private/tastemolnet/splits/validation.csv \
  --expected-validation-sha256 eedb06c6997652113f234f085135acd4f6dafb10f0d5d4d8e3f432473712a016 \
  --output-json /absolute/fresh/tastemolnet-neurosed/feature_schema.json
```

The command accepts only the tracked `configs/hpc.yaml` bytes and the one
fail-closed inference override shown above. It calls the shared held-descriptor
split loader for `train.csv` and `validation.csv`, verifies each file's SHA and
declared role, rejects cross-split molecule-ID overlap, and calls the shared
`derive_feature_schema`. The output file is a fresh atomic-no-replace canonical
`tastemolnet_gcf_neurosed_feature_schema_v1` object; producer audit fields are
not mixed into that downstream schema. The single aggregate JSON receipt on
stdout records `opened_payload_splits=["train","validation"]`, both split
roles/counts/hashes, and explicit false/empty evidence for calibration/test
payload access. It contains no molecule ID or SMILES.

The paired `scripts/slurm/build_tastemolnet_neurosed_feature_schema.sh` is
syntax/CLI-parity evidence only and exits 78 before the Python command because
this data route is AutoDL-only.

The pair builder reads one normalized absolute, non-symlink CSV through a held
descriptor, verifies its SHA-256 before use, reconstructs the reviewed feature
schema, and writes unlabeled metadata. Each row records the required graph IDs,
split, sizes, scaffolds, seeds, post-sampling diagnostic stratum, and
reconstruction hashes. The source/target draw path never reads class labels,
scaffolds, or size bins and does not materialize a Cartesian product. A
top-level seed other than exactly `7` is rejected by the sampler, manifest
builder, and CLI. Each final sampler manifest carries a canonical self-hash. A
requested training/validation budget must additionally have
`ceil(1.10 * budget)` deterministic candidates; successful rows are taken in
sampler order, with no GED-result-based choice.

For the present Taste project pair inventories, the preregistered query
sampler values are `n_hops_query=5` and
`traversal_probability_query=0.5` (the pinned upstream argument is named
`trav_prob_query`). These are project sampling choices. The upstream function
requires the caller to supply both values, and the pinned GREED-experiments
AIDS material consumes prebuilt pairs rather than establishing `5`/`0.5` as
official defaults. Therefore manifests and prose must not describe either
number as an upstream AIDS or GREED claim.

The 100-, 500-, and 1000-pair benchmark cohorts are disjoint slices of one
deterministic 1600-pair inventory. File hashes and the actual ordered pair-ID
inventories are carried into the reports so a summary cannot prove
disjointness from hashes alone.

## 5. Benchmark and worker selection

Each active selector replay calls the authenticated isolated `pyged` module
and real GEDLIB `branch` backend. The historical tiered F2 filenames below are
retained only as provenance for the superseded planning route:

```text
gedlib_benchmark_100.json
gedlib_benchmark_500.json
gedlib_benchmark_1000.json
gedlib_benchmark_summary.json
gedlib_worker_selection.json
```

Raw observations distinguish `SUCCESS`, `TIMEOUT`, and `GEDLIB_ERROR`.
Timeout/error rows have null bounds and never enter training. Reports include
wall time, seconds/pair, pairs/hour, p50/p90/p95/p99 latency, timeout/failure
counts, child CPU time/utilization, maximum child RSS, load average, iowait,
and node/edge-count correlations.

The local label contract retains both pyged bounds and the exact-versus-bound
flag. It records the pyged `float64` return and reproduces upstream
`torch.empty` storage as finite `float32`; it never averages the interval or
selects only one endpoint. A timeout/error row cannot be converted. The
reserve selector consumes rows only in deterministic sampler order and either
reaches the requested success count or returns
`BLOCKED_GEDLIB_LABEL_YIELD`.

Worker trials use every member of `1, 2, 4, 8` that does not exceed the
runtime-detected physical-core count. Every candidate is mandatory and uses a
fresh, mutually disjoint cohort of at least 100 real pairs; these cohorts are
also disjoint from the 100/500/1000 planning cohorts. A missing candidate,
duplicate pair, backend/config drift, or unreproducible throughput blocks the
selection rather than permitting an operator choice.

`gedlib_worker_selection.json` embeds and canonical-hashes all candidate
reports. It excludes any candidate with a timeout, GEDLIB error, unhealthy
host-load/iowait gate, BACE legacy throughput drop above 10%, or AIDS exact
throughput drop above 10%, then chooses the remaining highest measured
pairs/hour (lower worker count breaks an exact tie). Its validator rebuilds
the whole manifest and selection. The budget planner consumes that validated
manifest and has no `--selected-workers` or manual CPU-contention override.
All GED workers set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `TOKENIZERS_PARALLELISM=false`. Protected jobs
are never stopped by this route.

There is currently no reviewed producer that authenticates the protected
BACE/AIDS process generations and samples their progress plus load/iowait
before and during each GEDLIB trial. Consequently the benchmark CLI no longer
accepts operator-supplied throughput-drop percentages or health flags. It
binds a self-hashed
`tastemolnet_neurosed_gedlib_worker_resource_evidence_v1` blocker containing
its own process identity and host sample, then exits 78 before importing pyged
or starting any worker process and without a benchmark PASS marker. Worker
selection replays that blocker and returns
`BLOCKED_GEDLIB_RESOURCE_EVIDENCE` with `selected_gedlib_workers=null`.
A self-authored resource-evidence `PASS` is rejected while the reviewed
producer source SHA remains null. The future producer must add authenticated
BACE/AIDS identities, timestamped pre/during progress counters, periodic
load/iowait samples, and recomputed drop percentages before this gate can
select any worker count. After that producer exists, every required worker
candidate must carry authenticated resource evidence: one missing or
unauthenticated candidate blocks the complete selection, while a fully
authenticated candidate with timeout/error or measured unhealthy contention
is merely excluded from ranking so another authenticated candidate may win.

The checked-in pair builder also emits only the unique 1600-pair planning
inventory partitioned as 100/500/1000. It cannot yet emit four additional
worker-trial cohorts that are mutually disjoint and disjoint from those 1600
pairs. Therefore `WORKER_TRIAL_COHORT_BUILDER_NOT_IMPLEMENTED` is an
independent machine blocker. The tracked blocker document is
`configs/autodl/tastemolnet_neurosed_worker_trial_blockers_v1.json`; its marker
is null and `safe_to_select_workers=false`. The paired Slurm refusal prints
both infrastructure blockers and deliberately contains no example four-report
command that could be mistaken for a runnable release route.

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
hours, timeout rate is no more than 0.05, whose disk reservation passed, and
whose machine-replayed worker-selection manifest is PASS. The selected count
must equal the worker count of the real 1000-pair report and all backend pins
must match. If even 5000 fails, the result is
`BLOCKED_GEDLIB_THROUGHPUT` plus all three ETAs. It cannot select an unapproved
budget or fall back to own-parent or approximate labels.

The disk gate must reserve compact/columnar label storage while retaining at
least `MIN_FREE_AFTER_RESERVATIONS_GB=100`. The benchmark JSONL is diagnostic;
a full label pipeline must use Parquet/Arrow or NumPy binary rather than a
large per-pair JSON debug dump.

## 7. Selector, GCF direction, and pre-release health contracts

The pure selector state machine mirrors pinned `neuro.train.train_full`:

- one validation batch is observed immediately before every permitted train
  batch;
- only a strictly lower validation interval loss creates a checkpoint
  candidate; equality is a non-improvement and has no tie break;
- the counter is not reset at epoch boundaries;
- stopping occurs before the paired training update when consecutive non-
  improvements become greater than
  `cycle_patience * (step_size_up + step_size_down)`;
- every permitted training update records AdamW completion, one CyclicLR step,
  and gradient clipping at `0.1`.

Each candidate binds checkpoint bytes captured at that pre-update event.  The
production trainer uses the pinned GREED-expts AIDS call: train batch 200,
validation batch 1000, both shuffled, AdamW `lr=weight_decay=1e-3`,
`cycle_patience=5`, CyclicLR steps 2000/2000, and clipping at `0.1`.  It writes
`best.pt` from the selected pre-update candidate and makes `model.pt` an exact
byte copy of the same selected state.  The worker stops at
`READY_FOR_INDEPENDENT_VERIFICATION`; a second CLI process reopens and replays
the selector before it can write the scientific PASS marker.  For a controller
that owns only one command argv, `--train-and-verify` performs the training and
then launches that verifier as a child Python process; the parent command is
successful only after reopening the verifier's PASS-last file.

The direction binding calls `embed_targets(original_inputs)` first and then
exposes only `predict_outer_with_queries(generated_candidates)`. Every matrix
entry records query and target graph hashes with roles
`generated_counterfactual_candidate` and `original_input_graph`. A reversed
API or unexpected matrix shape is rejected. This binding is tested with an
in-memory model but is not yet wired into T7.

The production writer accepts the frozen inventory shape actually present on
AutoDL: exactly 5000 train rows and 1000 validation rows, with zero surplus.
It never claims the historical 10% reserve for those files.  If any exact-
inventory GED call fails, the run ends as `BLOCKED_GEDLIB_LABEL_YIELD`; it may
not resample or manufacture a replacement.  A future inventory that physically
contains the preregistered 10% reserve remains accepted and uses first-success
sampler order.  The two independently verified selected-backend canary replays
are reopened, compared field-by-field, proven to be the exact train prefix,
and their successful directional rows are adopted as cache rather than rerun.

The fixed-budget model-card/readiness validator cross-binds train/validation
sampler manifests, label manifests, the selector trace, and the direction
trace. It revalidates seed 7, the unstratified independent-draw contract, the
declared exact-budget or physically present reserve count, and each sampler
self-hash rather than trusting model-card fields. It requires real
pyged labels, exact approved budgets, compact storage, no held-out data,
the retained directional SED costs, the explicit non-MIP backend identity,
successful reload/batch checks, and all source/checkpoint hashes. Its output is
only
`READY_FOR_MANAGED_INDEPENDENT_VERIFICATION` with `marker=null`. It also
descriptor-reopens the complete retained GCF tree and binds it to the
authenticated upstream repository/commit. The former missing-GCF-identity
blocker is closed; this does not close any GEDLIB, training, or T7 blocker.

## 8. Remaining release blockers

The writer/trainer/verifier path is implemented but this code-only change does
not claim that AutoDL labels or a checkpoint exist.  The following runtime
work remains before T7:

1. deploy this immutable code on AutoDL and run the writer against the already
   verified build, selection/receipt, feature schema, and exact 5000/1000 pair
   roots;
2. run the official trainer and its separate verifier on a released GPU;
3. bind the resulting PASS/checkpoint into T7 and its managed method verifier.

Until all six complete, do not emit
`[TASTE_NEUROSED_FIXED_BUDGET_PASS]` or any T7/T12 PASS merely because this
code compiles; the marker is emitted only by the independent artifact reopen.
