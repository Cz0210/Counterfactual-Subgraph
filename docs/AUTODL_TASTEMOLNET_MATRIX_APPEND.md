# AutoDL TasteMolNet strict matrix append

`scripts/autodl/append_tastemolnet_matrix_authority.py` is a CPU-only,
read-only consumer for completed TasteMolNet T11--T14 paper cells.  It does
not run science, select candidates, open a split, or accept smoke/canary
markers.  Invoke it once whenever one or more new full-cell terminal roots
exist; the next invocation must use the just-published authority as its
predecessor.

The publisher reopens the predecessor's combined hash closure, the managed T3
GINE/temperature publication and the scoped Taste policy receipt.  It then
requires the exact method stage, terminal schema and PASS bytes, an independent
verifier, immutable inventories, no live writer, calibration freeze before
test, no test selection/threshold fitting, no RF oracle, three-class source
label 1, destinations 0/2, and identical dataset/split/test-parent/MolCLR/
threshold identities across every passing Taste row.  Non-target matrix rows
must remain byte-for-byte equal as JSON objects.  Publication uses a fresh
staging directory, a no-replace atomic rename, and an independent reopen.

T11 must first be published by its separate final verifier; its science root
is deliberately not accepted.  T12 production is not released yet.  Its
consumer contract is reserved exactly as
`tastemolnet_t12_final_run_manifest_v1`,
`tastemolnet_t12_terminal_verification_v1`, and
`[TASTE_GCF_PASS]\n`; replay-canary output cannot match it.

Direct AutoDL example for one newly completed method:

```bash
export PYTHONPATH="$PWD"
python scripts/autodl/append_tastemolnet_matrix_authority.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --prior-authority-root /absolute/current/hash-closed-matrix-authority \
  --taste-cell ComRecGC=/absolute/t14/final-attempt-uuid \
  --t3-root /absolute/managed/tastemolnet/gine/seed7/calibrated-publication \
  --taste-policy-receipt /absolute/tastemolnet_policy_receipt.json \
  --prepared-root /absolute/tastemolnet/prepared \
  --graph-cache-root /absolute/tastemolnet/graph-cache \
  --output-root /absolute/fresh/matrix-authority-after-t14
```

Repeat `--taste-cell METHOD=ROOT` to append several terminals that became
ready together.  The existing continuation controller should enqueue the
same command independently after each real cell PASS; no new controller is
required.  A successful invocation prints only the matrix count it actually
published, for example `[MATRIX_9_OF_16_PASS]`.
