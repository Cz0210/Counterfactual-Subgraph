# BACE Ours disconnected-residual postmortem

## Incident

The frozen BACE Ours v2 Figure 3 reported 107 of 116 eligible test parents
covered at K=1 by the fragment `CCN`. Independent exact RDKit matching
reproduced the matches, but 106 of the 107 winning residuals had multiple
connected components. The evaluator's legacy `hard_delete_all_matches_v1`
path sanitized the residual and accepted it without requiring one component.

## Root cause

`hard_delete_substructure_any_match` intentionally preserved all components
and marked a sanitized dot-separated residual as deletion-valid. That legacy
behavior was then consumed by the RF teacher and MolCLR distance provider.
The resulting low distances and strict flips were internally consistent rows,
but they represented disconnected graph collections rather than one residual
molecule. The old q30 threshold was fitted from the same action semantics and
is therefore contaminated as well.

## Correction

The corrected primitive removes one exact unique atom match and fails closed
unless the residual is nonempty, sanitized, and single-component. All matrix,
evaluation, aggregation, selector, threshold, cache, and artifact contracts
carry the action and match-policy versions. A separate cache namespace prevents
legacy disconnected distances from being reused.

The corrected outputs are versioned under:

- `outputs/hpc/optimization/bace_ours_connected_residual_v3`
- `outputs/hpc/selectors/bace_ours_connected_residual_v3`
- `outputs/hpc/eval/paper/bace_common3_connected_residual_v3`

The original v2 outputs are preserved read-only. Candidate selection remains
calibration-only, GCF results are not selector inputs, and the corrected test
evaluation is allowed exactly once after the connected selection gate passes.
