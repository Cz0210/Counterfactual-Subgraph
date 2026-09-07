# BACE A+ GlobalGCE attachment and oracle adaptation

This is a new version, not a rewrite of the original80 BLOCKED materialization
or the paused GINE repair at epoch35. The prior source is cd051072. The old
joint categorical NONE/bond formula and native LHS tensor mapping are retained.

The prior materializer reconstructed all parent atoms and bonds. Its native
tensor schema stores atom chirality but does not store bond E/Z direction;
reconstructing unchanged bonds therefore cannot preserve all identity graphs.
The A+ materializer keeps a copy of the actual input parent molecule in its
original tensor order and edits only changed atom/bond slots. It never reparses
canonical SMILES as if they retained the old global slot numbering. The original
parent remains immutable.

Boundary nodes incident to an outside atom are explicit fixed original-atom
anchors. This constraint is applied before both differentiable hard training
and final validation/materialization. Outside bonds are not synthesized or
searched. A disconnected local RHS may be legal via the original outside graph;
an actually disconnected complete product remains rejected. No arbitrary
single bonds or test-dependent attachment choices are added.

The frozen corrected GIN is a different oracle from the old GINE. Therefore
epoch35 may initialize one new seed7 GIN-aligned warm-start, not a claimed
trajectory-preserving resume. No old optimizer/epoch/ledger is rewritten. A
separate hard-forward GIN bridge matches the ordinary deployed GIN messages
(without GINE's message ReLU), normalization/readout and frozen temperature.
Only decoder inputs receive straight-through estimated gradients. Invalid
complete products are not passed to the classifier.

Identity and positive-control fixtures are engineering evidence, not generated
counterfactuals, train coverage, or scientific test results. A+ results remain
in an independent experiment registry; they do not replace the original GINE
main row or add a matrix cell.
# CPU closeout (2026-09-08)

The separate `run_bace_globalgce_aplus_evaluation.py` leaf accepts a completed
validation-selected native pool, evaluates calibration, freezes its own original
GlobalGCE selector, then opens test. It retains all real LHS mappings and uses
the minimum exact raw graph cost among current-GIN strict flips. A missing cost
is a failed parent boundary, not a zero result. Native selector fingerprints
remain the original aligned LHS/RHS transition bits, not invented deletion SMILES.

The original Global selector provenance is the `selection-shared` directory in
`bace_baseline_merge_closeout_0e5d31f_20260901T114200Z/globalgce`: its actual
`variant_configs.json` and `frozen_selection_manifest.json` bind A1–A4, seed13,
two local swap passes, top20/table10, and the original B12 frozen thresholds.
The new leaf verifies these source files before selection; it never adopts the
old GINE winner or flips. Its saved-record audit explicitly does not claim a
second model, chemistry, or OT execution. Output is an independent A+ registry,
not the original main-matrix authority.
# CPU stage-boundary handoff and split binding (2026-09-08)

The future CPU leaf hashes each split CSV against the original portable bundle's
frozen manifest at that split's entry. Test bytes are first opened only after this
GlobalGCE pool's own selector freeze. Calibration matrix files are adopted on
resume only when their complete saved content matches the already completed
parent checkpoints. GlobalGCE retains its original four-variant selector and
threshold source; the authorized at-most-K cap affects only a short available
pool, never pads or regenerates candidates.

`cpu-handoff` is a single dataset-specific stage handoff in the existing GlobalGCE
owner module. It waits for the exact original owner terminal and bound pool,
then calls calibration, freeze, test and aggregate in order. It never edits the
live training owner/spec or acquires a GPU lease. Resource pauses resume only
sealed parent checkpoints; engineering errors stop with their evidence retained.

The existing AutoDL raw-test index has no available previously sealed file-SHA
descriptor. Its fixed path is therefore not opened during planning: the Global
leaf first reads/hashes it after its own selector freeze, validates its internal
source/numerical/kernel bindings, and only then atomically writes a fresh adoption
receipt. That receipt explicitly states the file hash was not previously bound.
Later resume rejects changed bytes; old teacher flip masks/minima are not adopted.
