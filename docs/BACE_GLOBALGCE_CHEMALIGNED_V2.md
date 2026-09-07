# GlobalGCE-ChemAligned v2 (2026-09-07)

This is an explicitly authorized new molecular adaptation, not retroactive
validation of the historical zero. Pinned upstream 157e65c emits sigmoid
adjacency probabilities and unrestricted affine bond logits (its apparent
sigmoid was a Linear bias argument). The new contract uses softmax on the
bond class axis and p(NONE)=1-a+a*q(NONE), p(b)=a*q(b). Joint argmax with
lowest-index ties determines both adjacency and bond labels. Padding masks
remove edges incident to absent atoms. No conflicting edge is forced SINGLE.

The LHS tensor index, not Python mapping insertion order, binds RHS atoms to
the parent. Unchanged external attachments and source atom attributes are
preserved. Validation is of the complete replaced molecule: a disconnected
local RHS may be valid when the untouched parent connects its atoms. Empty,
disconnected or unsanitizable complete products are rejected, without feature
fallback or altered chemistry. This adaptation does not modify active Taste
code or old frozen rule files.

The opt-in bridge supplies actual sanitized full-graph GINE features (including
parent formal charge/chirality). Its forward is the existing frozen GINE; an
explicit straight-through embedding estimator propagates generator gradients.
Invalid products are not sent to GINE. No differentiability claim is made for
the discrete chemistry operation.

First execute only the bound train and validation rematerialization of all80
saved raw rule tensors, reusing original mining, generator checkpoint, GINE,
360 source-train IDs and the official validation split. It cannot load test or
calibration. A later single seed7 repair-finetune, maximum100epochs, is allowed
only if train feasibility fails; its exact objective and execution entrypoint
must be sealed before launch. Rematerialization is not training or matrix PASS.
The existing141 denominator, threshold/cap and K1..20 remain fixed for later
calibration/frozen-pool evaluation, with at-most-K and no padding.

CLI: `python -I -B scripts/run_bace_globalgce_chemaligned.py --config
configs/hpc.yaml --repair-config /absolute/contract.json --action rematerialize
--output-root /absolute/fresh/root`. CPU action requires empty
CUDA_VISIBLE_DEVICES. Paired Slurm wrapper is kept in sync but is not submitted
for this CPU preparation. Status uses the same CLI with `--action status`.
