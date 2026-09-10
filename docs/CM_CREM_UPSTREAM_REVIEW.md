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

### Deployed environment and bounded transport repair

The first deployment created Python3.11.5 successfully, but its online pip
stage exited124 at the600s total transport deadline, after the34.9MB RDKit
download and during NumPy download. `installation_terminal.json` and both
original logs remain unchanged. The second, explicitly authorized engineering
attempt did not repeat Conda create. Mac downloaded the same pinned Linux
cp311 wheels from the official PyPI index, with Pillow12.3.0 (the original
resolved RDKit dependency). Every wheel passed full ZIP CRC and one content
SHA at Mac, then one destination transfer SHA. Offline pip installed those
four wheels into the existing project-only prefix and succeeded.

Actual prefix:
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1/environment-20260910-v1/python3115-crem0214`.
Actual readback is Python3.11.5, CReM0.2.14, RDKit2023.9.6,
NumPy1.26.4, Pillow12.3.0. `-s -B` with PYTHONHASHSEED=0 was read back as
ignore_environment=0 and hash_randomization=0. This proves dependency imports,
not ChEMBL22 replacement or scientific pilot success.

The immutable attempt2 records under the environment root are
`installation_attempt2_intent.json`, `installation_attempt2_terminal.json`,
`actual_versions-attempt2.json`, and the pilot-spec-compatible derived
`offline-attempt-2/environment_manifest.json` (binding the actual readback
path and SHA). `scripts/cm_crem_generator_env.sh` accepts the explicit
`--offline-repair-attempt2` action only when the unchanged attempt1 terminal
is FAILED_pinned_pip/124 and no attempt2 intent exists. Its offline install
has a1800s bound; it rejects repeated repair rather than silently using a
third attempt. No shared Conda environment was modified.

The author subset is deployed at
`/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1/assets/upstream-b5816b502cde00ee24c652a02cbc54664583f773`.
`reviewed_source_manifest.json` binds the four reviewed source files and MIT
license extracted from `git archive` of that full commit. All archived members
were checked; no symlink, unknown file or existing destination was accepted.
This is explicitly a reviewed source subset, not a claim that DiffLinker or
the whole author runtime has been installed. ChEMBL22 remains a separate
required asset; no substitute database has been obtained.

## Pilot repair1: preserve original bond endpoints across V3000

The real oracle pilot2654380 failed during input serialization, before native
CReM generation: `make_parent_request` / `load_parent_mol` rejected an ordered
graph mismatch. A bounded HPC-only audit of the frozen386 train parents found
47 affected parents and51 reversed bonds. The only differing fields were bond
begin/end atom indices; every canonical chemical identity was unchanged.
V3000 can orient a bond toward a chiral atom when writing wedge stereochemistry.
Neither the GINE model nor Grad-CAM caused this transport failure.

The explicit `cm_crem_parent_v2` schema adds the original bond endpoint vector.
Loading first proves that each same-index bond joins the same two atom indices,
then restores its original orientation and retains atom/bond order. Reversal
of direction-sensitive bond types is rejected. Original H flags and bond
directions are still restored, followed by the unchanged full ordered SHA and
canonical-isomeric identity checks. No atom or selected-mask index is guessed
or remapped. Valid legacy v1 records remain accepted under their existing strict
checks. The CM algorithm, oracle, attribution formula and chemical budget are
unchanged.

The minimal public synthetic fixture `C[C@@H](O)CC` reproduces the same
V3000 endpoint reversal without exporting dataset molecules. Eighteen focused
transport/native tests pass. On HPC, all386 frozen train parents pass the v2
roundtrip in the original RDKit2025.09.3; the exact same saved requests also
pass in isolated RDKit2023.09.6, retaining full ordered hashes and canonical
identity. Test data, GNN inference and OT were not used by this regression.
The original failed pilot and its four existing attribution-unit files remain
untouched. Evidence is under the original run's
`repair1-transport-diagnostic/{redacted_summary,produce_regression,consume_regression}.json`;
full diagnostic molecule records remain on HPC and were not exported to Mac.

## Static database compatibility helper

`src/baselines/cm_crem_database_compat.py` exposes
`verify_static_database_compatibility(database: Path) -> dict` for the dedicated
computation-node asset preparation process. The caller owns the job timeout
and durable receipt. It is not an additional CM pilot parent or oracle call.

Review of actual CReM0.2.14 `__get_replacements_rowids` and `_get_replacements`
establishes `radius1`, `rowid`, `env`, `freq`, `core_num_atoms`, `core_smi`, and
`core_sma` as the required mutation SQL interface. `dist2` is only consumed
when a link/distance argument is supplied; the radius1 mutation probe does not
silently require that unrelated column. The installed `crem/crem.py` must
match reviewed wheel source SHA
`e47fb661e318378e370e12507370f2576c28e7e6ebcf83dd7355a75763568d89`.

The helper checks the SQLite header, rejects existing WAL/journal/shm, opens
`mode=ro&immutable=1`, enables connection-only `query_only`, checks the actual
table/columns, and performs a real nonempty SELECT. It then calls the actual
`mutate_mol` once on the fixed public molecule CCO, radius1, same-size fragment,
replace_ids=[0], symmetry_fixes=true, max_replacements=4 and ncores=1. This is
a narrowly bounded asset compatibility fixture, not altered scientific CM
budget. At least one different valid connected complete molecule and actual
SELECT activity during that native call are required. A query with no output
does not become a false replacement PASS.

All connections are closed, the CReM-only SQLite namespace and Python RNG are
restored, and inode/size/mtime/ctime plus sidecars must be unchanged. There is
no VACUUM, journal-mode change, database recovery or database full rehash; the
outer static asset receipt supplies the once-verified complete content digest.
EIO/SQLite failures remain explicit infrastructure failures. Exceptions expose
`DatabaseCompatibilityError.stage` and `.receipt` to the outer task.

Twelve focused tests pass, including real CReM0.2.14 code performing CCO→NCO
against a synthetic radius1 fixture, denied writes, source-change detection,
missing schema, empty table, empty replacement, and EIO propagation. This is
not yet a claim that the official ChEMBL asset passed its computation-node
check; only that job's real receipt can establish that status.
