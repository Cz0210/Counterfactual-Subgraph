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

## One preregistered repair-finetune objective

The frozen `TRAINING_CONTRACT` is serialized into a fresh training contract
before the GPU canary or formal campaign. It does not replace the active
rematerialization receipt. Initialization is the saved official generator,
not a fresh teacher or newly mined LHS. Seed is 7. One epoch is exactly one
optimizer update over at most five logical batches of 500 applications,
drawn by a separate seed7 `torch.randperm` from the complete source-eligible
train match universe. There are at most 100 epochs and exactly 100 optimizer
updates in a completed budget; this is not 100 updates per epoch. Summed small
backward calls do not step the optimizer or alter the sampled decoder state.

The fixed objective is 10 times mean node L2 reconstruction, 10 times mean
joint-state L2 reconstruction, 100 times binary cross entropy of joint
non-NONE adjacency, 100000 times the pinned original Gaussian KL expression,
and 10 times the mean valid complete-product target-0 NLL. Adam uses learning
rate0.1 and weight decay1e-5; StepLR uses step_size10 and gamma0.9. The saved
generator is already post-warmup, so the same full objective applies at every
repair epoch. Node sigmoid weights are normalized for the node reconstruction
term. Edge reconstruction and adjacency consume the single joint state, not
incompatible separate hard outputs. This is an explicitly new adaptation,
not a claim of numerical equivalence to the old objective/trajectory.

Invalid complete products do not enter GINE. They receive only reconstruction
gradients, with a fixed `-log(float32 epsilon)` diagnostic penalty in the
reported mean NLL; no fabricated oracle probability is optimized. Real valid
products use the frozen GINE and an explicitly straight-through estimator.
We record actual match, legality, strict-flip, unique-product and parent counts,
plus train witness probabilities and mapping IDs.

Validation evaluates all bound validation matches every fifth epoch. Checkpoint
selection is lexicographic: most strict-flip parents, then most valid parents,
then lowest mean target NLL, then earliest epoch. This rule is fixed before
training and never uses calibration or test. The first update and every later
epoch atomically save model, optimizer, scheduler, complete RNG, sampler state,
input and joint-contract references. The bounded GPU canary additionally loads
a fresh independent generator/optimizer and executes the next saved-state step.
It also checks a known legal train identity fixture through the actual saved
generator's differentiable decoder and real frozen GINE. Identity is explicitly
forced only for that synthetic engineering fixture, never for actual generation
or training; it is not reported as a generated recourse. Model buffers and RNG
are restored after this fixture.

At budget completion the validation-selected weights undergo a train-only
existence search. A first actual flip ends this feasibility search (not a train
coverage estimate). A missing witness after its complete search is
RESEARCH_TARGET_UNMET, not an engineering PASS or invented zero-coverage cell.
An actual witness allows native rule export with stable semantic dedup and all
genuine LHS source indices. The shared existing evaluator explicitly opts into
this adapter, retaining full-parent execution, fixed BACE GINE, old thresholds,
native calibration selection, test-after-freeze and K1..20 saturation. For m<20,
weights of saturated prefixes are folded into the final effective prefix;
requested Table2 K10 uses min(10,m), without padded candidates.

Training CLI actions are `train-canary` and `train`, both requiring
`--rematerialization-root`; `--resume` only resumes the same bound latest
checkpoint. Formal quota is consumed once by an exclusive campaign ledger.
`export` takes the completed `--training-root` (or a rematerialized original
checkpoint with actual train recourse). Automatic training closeout exports a
new candidate universe only after real train feasibility. No matrix writer is
included. CPU/GPU resource and original lease admission remain launcher duties.
# Existing-owner stage continuation

`scripts/run_bace_globalgce_chemaligned.py --action owner --owner-spec ...`
uses one canonical owner root bound in the fresh training contract. It waits
for the exact Ours `TRAIN_ONLY_POOL_FROZEN` receipt and then delegates every
canary/formal GPU stage to the already tested AutoDL owner and UUID/FD locks.
No GPU reservation is changed here. At an expired resource receipt it pauses
before the next optimizer update; formal recovery uses only the original
same-root checkpoint and the original one-shot ledger. Successful training
exports a train-witness-gated pool, followed by explicit CPU calibration,
single selector freeze, test, and final-freeze commands. Matrix supersession
requires the original publisher's independent corrected-version interface;
evaluation completion alone is not PASS/publication.
