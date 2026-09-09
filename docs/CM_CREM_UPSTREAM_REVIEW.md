# CM-CReM source review — 2026-09-10

This is a source / fixture review, not evidence of ChEMBL22 generation or a
completed BACE experiment. The original main-table oracle and outputs are not
changed. Source: `https://github.com/gmum/counterfactual-masking`, full commit
`b5816b502cde00ee24c652a02cbc54664583f773` (MIT, copyright2025 GMUM).

## Source and dependencies

The entire `source/linksGenerator.py` was read. Original byte SHA256:
`b5e485195ede009c560ee397f458794dc22272c14fb17faf3313fbdc64f39e49`.
The review copy under `patches/cm_crem/upstream_linksGenerator.py.txt` adds one
terminal LF because the original file has none; the test removes that single LF
and checks the exact original SHA. Production reads the actual pinned checkout
file and accepts no whitespace-normalized substitute. The upstream license is
retained alongside the review copy.
`patches/cm_crem/native_scope.patch` is the mechanical import/definition-scope
diff: the retained source segments were asserted AST-identical to the runtime
selected node set. It is not a standalone module without the explicitly
documented dependency namespace, nor an altered CM algorithm.

Actual author requirements: Python3.11.5, CReM0.2.14, RDKit2023.9.6,
NumPy1.26.4. The exact CReM0.2.14 wheel was downloaded for read-only source
review. Its `crem/crem.py` uses `import sqlite3` and a single
`sqlite3.connect(db_name)` at line320; the selected single-core path contains
no exception handler that suppresses database exceptions. CReM is BSD3-Clause.
Database is the author's separate ChEMBL22 SA2 external prior; no license,
accessibility, lack of train/test overlap, or database download success is
inferred from the code license. Its actual byte identities belong in the
campaign asset receipt, not this source review.

## Actual attribution and mask path

Read `source/explainability.py`, the complete counterfactual generation script,
and `helpers_counterfactual.py`. Original GradCAM uses the actual `c3` activation,
`model.predict(...).backward()`, per-node channel-mean gradients multiplied by
activations and summed over channels, then min-max normalization (no ReLU).
The scalar author predictor is adapted to the frozen project's source-class
score by the separate oracle module; this does not adopt the author's new GIN.

The CF script sorts importances descending, retains `max(1, n//5)` atoms
(floor, not ceil), and unions every `AtomRings()` ring intersecting that INITIAL
mask. It does not recursively expand rings intersecting newly added atoms.
Integer-mask sets and component iteration use the fixed process hash seed0.
The generation input is the same ordered parent Mol, not a reparse of an
unrelated canonical-SMILES metadata field. MolBlock exchange explicitly carries
SMILES noImplicit/explicit-H and bond slash flags omitted by V3000; restoring
them must recover both the complete ordered atom/bond digest and the original
canonical-isomeric graph. Chirality, isotope, charge, H, stereo atoms and bond
endpoints are checked. No atom-index guessing is allowed.

The task's explicit empty-context exception is `NO_REPLACEABLE_CONTEXT`, with
zero native calls and no database query when ring expansion masks the complete
parent. It is not an infrastructure error or a replacement by another molecule.
Other nonempty-context parents make exactly one top-level official CM call.

## Native generation and exact adaptations

`_load_native` checks the full original file bytes, then compiles only the
unchanged FunctionDef/ClassDef AST nodes in `NATIVE_SYMBOLS`. It injects the
standard library/RDKit dependencies and the monitored `mutate_mol` callable.
No algorithm statement is rewritten. The returned provenance includes original
source SHA, executed AST SHA, deterministic unparsed executed-source SHA, and
the exact selected dependency names. The import namespace omits the unrelated
OpenBabel/DiffLinker code and never imports their modules or models.

Preserved native behavior:

- component discovery and exact anchor-list-based component merging;
- radius1; up to64 replacements PER COMPONENT, not per full parent;
- same-size symmetry-on query with native60s alarm, symmetry-off fallback;
- size-increase1,2,3 with the same symmetry/fallback order;
- single-component direct return and native attachment reconstruction for
  multiple components (including its explicit SINGLE attachment bonds);
- full Cartesian combination when <=500, else Python random.sample500;
- original chemistry sanitization / disconnected rejection behavior.

No Morgan/local similarity-diversity selection is invoked. Global prototype
selection is a separate authorized adaptation. The native returned raw strings
are exact-string-deduplicated, hashed with the independent parent seed, and
limited to128 before any candidate oracle/distance filtering. Graph-level
canonical dedup occurs later in the unchanged evaluation environment.

Database transport adapts only `crem.crem`'s SQLite namespace, not global
`sqlite3.connect`: the one bound absolute static database is opened `mode=ro`
and `query_only=ON`. Per-query receipts record actual fallback/component calls,
returned counts and elapsed times; SELECT calls are counted. All query boundary
records are flushed to the dedicated parent log, including before exceptions.
OSError/SQLite errors become an explicit BaseException sentinel to cross the
author's broad Exception handler, then an INFRASTRUCTURE_FAILED outer receipt.
Unexpected non-timeout errors suppressed by the author are detected from the
query record and become ENGINEERING_FAILED. Neither class becomes scientific
empty output. Original inner TimeoutException remains the native fallback.

Production `generate_parent` requires the isolated fixed environment, starts
one spawn worker, and supervises900s from a separate process. Native SIGALRM
cannot replace that timer. On expiry only the verified worker's own process
group receives SIGTERM; there is no killall/pkill/SIGKILL. A surviving group is
an engineering/blocking condition, not another dispatched parent. Partial
results are never adopted. A timeout is a budgeted outcome, not a correctness
or zero-recourse proof. No other project process or environment is modified.
The dedicated generation interpreter uses `-s -B`, not `-I`: `-I` implies `-E`
and would ignore PYTHONHASHSEED even while os.environ prints0. Both inherited
ignore-environment flags and the actual child's hash-randomization flag are
checked. The original GINE/WNode isolated launch policy is not changed.

## Focused validation and limitations

Tests execute the exact author function bodies with real RDKit molecules and
controlled mock fragment queries. They cover stable seed/truncation, chiral /
isotopic / aromatic-NH transport, mask/ring/context, real native same-anchor
merge and multiple-component attachment, all8 fallback calls,64 budget,
source pin drift, failure propagation and real spawn process-group timeout.
A tiny SQLite fixture proves read-only access and module-only patching. The
optional fixed CReM wheel integration executes the actual library against an
explicitly EMPTY TEST database; it is NOT a ChEMBL22 test or evidence of an
official nonzero replacement. Local RDKit fixtures run in the existing Mac
test environment and do not claim the pinned production environment passed.
Production preflight and the32-parent ChEMBL22 pilot must pass separately.

## Isolated HPC installation

`scripts/cm_crem_generator_env.sh` uses a fresh task-specific prefix under
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1`.
The existing Conda executable is read-only. All Conda package/env directories,
pip/XDG caches and temporary files are explicitly within that task. Crucially,
the installed Conda24.9.2 source was checked to honor `CONDA_REGISTER_ENVS=false`,
so it does not append to the shared user's `~/.conda/environments.txt`.
No HOME variable is repurposed and no shared package/env/cache is changed.
The existing GINE Python3.10.19 is not reused as the generation interpreter;
there was no Python3.11.5 archive in the inspected existing package cache.
Install is one bounded1200s Conda stage and600s pinned pip stage, with exact
actual version/import readback and an immutable terminal. Failed attempts are
not overwritten or automatically reinstalled. This is environment setup, not
scientific generation; no ChEMBL database is replaced by its import fixture.
