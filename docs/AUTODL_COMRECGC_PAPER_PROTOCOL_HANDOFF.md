# AutoDL COMRECGC paper-protocol continuation handoff

Status timestamp: 2026-08-25 Asia/Shanghai. This file is a live handoff and
must be refreshed from the persistent controller before release.

## Frozen protocol and source

- Paper: Fournier and Medya, *COMRECGC: Global Graph Counterfactual Explainer
  through Common Recourse*, ICML 2025 / PMLR 267.
- Official implementation: `ssggreg/COMRECGC`, commit
  `122f9341a360e9f06bb58a2f5823bb596021f6bf`.
- The source audit is in `docs/COMRECGC_ORIGINAL_PROTOCOL_AUDIT.md`.
- Official clustering input is the recourse-vector subset whose normalized
  frozen GREED/NeuroSED distance is `<= theta`; the complete Cartesian store is
  only a physical cache.
- AIDS DBSCAN contract is Euclidean, `eps=0.02`, `min_samples=3`, inclusive
  epsilon, with the sample itself counted as a neighbour.
- Downstream centroid-radius and centroid-norm filters use strict `<`.
- Paper/code differences are preserved: paper teleportation is `0.05`, while
  the completed official-code-style generation used `0.1`; paper writes
  `>0.5`, while the implementation uses `>=0.5`; upstream `--cf_size` is
  declared but unused.

## Current authoritative matrix

The scientific matrix is `7/16`: the six user-approved frozen-v4 AIDS and
Mutagenicity Ours/GCFExplainer/GlobalGCE cells plus frozen BACE Ours. The four
TasteMolNet cells remain `BLOCKED_LICENSE_REVIEW` and `RUN_TASTEMOLNET=0`.
No adopted PASS cell may be rerun or regressed because an older registry file
is stale.

## AIDS COMRECGC

- Physical snapshot rows: `91,916,686 = 71,642 candidates x 1,283 parents`.
- Physical order: candidate-major, parent-minor; pair columns are
  `[parent_index, candidate_index]`.
- Physical snapshot vectors SHA256:
  `68072364166c20364b8d079a08fd67f5008447db54f51b338f3f541eb54b39e5`.
- The old brute worker PID is `273939`; it remains protected until the new
  route has a protocol PASS, a stable exact-science checkpoint, and a
  materially acceptable ETA.
- Latest observed old progress: `24,480 / 91,916,686`; health class
  `RUNNING_UNVIABLE`; rolling ETA about `96,214 h`.
- The reviewed-but-unbound `2b5d3d42` route must not be deployed because it
  did not bind a theta-close logical view to the Cartesian snapshot.
- Frozen GREED full close-pair scan, logical input count, exact certificates,
  cluster/noise count, centroid/radius/greedy result, and final standardized
  output: pending the fresh paper-protocol controller.

## Mutagenicity COMRECGC

The existing Mutagenicity controller only waits on an older AIDS identity. A
fresh Mut controller must be built after AIDS PASS and bind the exact new AIDS
controller manifest SHA, task ID, and attempt-0 output. Mutagenicity must run
its own close-pair audit and exact clustering; it may not reuse the AIDS
bitmap, cluster certificate, centroid, or greedy result.

## BACE baselines

- GCFExplainer Quick-50, Quick-100, and M500 are strict-equivalence PASS.
  Patched M500 was 6.32% slower, so the decision is `KEEP_LEGACY_FULL`.
  Latest protected full PID `139725`: `16,846 / 50,000`, rolling ETA about
  `115.43 h`.
- ComRecGC legacy M500 is terminal complete. Optimized M500 PID `517784` was
  last observed at `200 / 500`; its exact equivalence audit had not yet run.
  Protected full PID `169008` was last observed at `5,425 / 50,000`.
- GlobalGCE v6 controller PID `450901`, owner PID `451683`, science PID
  `451688`; latest completed epoch was `17`, next epoch `18`. It remains on
  frozen GINE with RF disabled.

## GPU ownership

- GPU0: protected BACE GCFExplainer full (`139720 -> 139725`).
- GPU1: BACE GlobalGCE v6 (`451683 -> 451688`).
- GPU2: BACE ComRecGC optimized M500 (`456429 -> 517784`).
- GPU3: protected BACE ComRecGC full (`168949 -> 169008`).

All four observed UUID locks matched live workers. AIDS protocol audit and
exact clustering are CPU-only and must not acquire a GPU lock.

## Pending controller and commands

The immutable execution commit, new controller ID/PID/heartbeat, close-pair
throughput/ETA, exact DBSCAN checkpoint, graceful old-route supersession,
fresh Mut controller, matrix refresh, and three-dataset exports are pending.
The final deployment section must record exact status and restart commands
after the manifest has been built no-clobber and launched.

Only one manual intervention is allowed to remain: explicit TasteMolNet
exact-data licence or research-reuse evidence.
