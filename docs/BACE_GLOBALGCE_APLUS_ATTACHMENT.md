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
