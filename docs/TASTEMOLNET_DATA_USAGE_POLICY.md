# TasteMolNet research/reporting data-use policy

## Status of this commit

This repository records two facts separately:

1. the fixed TasteMolNet upstream snapshot does not explicitly state dataset
   licence terms (`NOT_EXPLICITLY_STATED`); and
2. a scoped project authorization may permit private research computation and
   reporting of aggregate paper results without permitting redistribution of
   the dataset.

Neither fact is a licence conclusion. The project must never emit
`LICENSE_PASS`, claim that the upstream licence was resolved, or use an
open-access paper licence as a substitute for data terms.

The checked-in machine-readable policy records the project owner's **active,
scoped research authorization**:

```text
authorization_state=ACTIVE_SCOPED_AUTHORIZATION
research_compute_allowed=true
paper_result_reporting_allowed=true
RUN_TASTEMOLNET=1
data_redistribution_allowed=false
```

This is project-level permission for private computation and aggregate paper
reporting, not an upstream licence conclusion. Every execution must bind the
policy's exact raw and canonical hashes, a fresh read-only authority receipt,
and fresh execution roots. Activating the policy does not itself deploy, start,
download, prepare, rebuild, or redistribute anything.

The authoritative policy path is:

```text
configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml
```

The historical `LICENSE_REVIEW_REQUIRED` artifact remains immutable provenance.
It is superseded only as an execution decision by the scoped policy
receipt; it is never deleted, rewritten, or converted into a PASS marker.

## Fixed private-data authority

The policy binds one existing local snapshot and does not permit another
download or preparation pass:

```text
upstream commit:
  16af8ead8a17b6bd3941d9eb5879c5be75c14114
source CSV SHA-256:
  b7308b3277fd07ed6af4b861c0d2ce2d843f92cc81a9e5e4efd65cf4040a291b
prepared output_manifest.json SHA-256:
  36aaf17bf45e0a092a96a0379fab31d9e6bfcd719b87cb4ffa4e57a6642bb645
split_manifest.json SHA-256:
  841f3b911e5d353c1e00f010bafcc8a6f7b3433082dba8a8979fab1b558251af
rows:
  train=9437, validation=1328, calibration=1328, test=1328
```

The read-only authority validator also requires the existing graph-cache
manifest and all four train/validation/calibration/test cache files to match
their source hashes and counts. It does not deserialize held-out rows during
the policy audit. Data preparation, source copying, network download, and
graph-cache rebuild are forbidden by this route.

## Scientific contract

TasteMolNet remains a real three-class task:

```text
0 = Bitter
1 = Sweet
2 = Tasteless
source class = Sweet (1)
strict counterfactual flip = pred_before == 1 and pred_after != 1
oracle_backend = gnn
classifier_family = gine
num_classes = 3
rf_oracle_used = false
```

Both Sweet-to-Bitter and Sweet-to-Tasteless count as strict flips. A binary
projection, an RF oracle, or a `1-label` target is invalid. Model fitting uses
train; checkpoint selection and temperature calibration use validation only.
The held-out test is not loaded during training; the full GINE fragment exposes
only its path and SHA until the frozen-oracle evaluation gate.

The AutoDL route is dedicated, CPU-controller/GPU-worker scoped,
exclusive to physical GPU 1 for the formal GINE, and requires fresh controller
and science roots. Physical GPU 2 is a separate classifier-independent READY
lane and is not silently used by the classifier.
HPC is forbidden for this campaign. Paired Slurm files exist only to satisfy
repository CLI parity and intentionally exit before running the command.

### Downstream T3/T4/T6 supplemental authority

The base T2 policy intentionally freezes classifier fitting to train and
validation. A second exact, machine-readable policy narrows the later access
rather than broadening the trainer:

```text
configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json
```

It permits only the following typed stages (T6 authority does not itself
implement or launch T6):

- `T3_GINE_CALIBRATED`: verify and adopt the temperature already fitted by T2
  on validation logits inside the immutable bundle. It reads the bundle's
  hash-closed `validation_predictions.csv` as calibration evidence, but opens
  no external validation CSV or graph payload; no optimizer/fitter is called,
  and no checkpoint is copied or rewritten. T3 is CPU-only and claims no GPU
  ownership. Its required fresh `calibrated-<timestamp>-<pid>` root is a
  hash-closed adoption/reference bundle, not a second classifier copy; the
  `checkpoint_id` in its gate is the one common calibrated oracle identity
  consumed by all four methods.
- `T4_ORACLE_SMOKE`: open the authenticated graph-cache `manifest.json` and
  `calibration.pt` only, load the selected `model.pt` oracle once on physical
  GPU 1, and run a deterministic bounded sixteen-parent three-class interface
  smoke with exactly four real connected deletions per selected parent and
  observed strict flips to both non-Sweet classes.
- `T6_OURS_SMOKE`: use the frozen prepared train CSV only for a bounded real PPO smoke
  whose reward is the same frozen three-class GINE. It requires at least five
  optimizer steps, an immutable T5 clean-policy input, no RF oracle, and no
  validation, calibration, or test payload access. A later implementation must
  still pass the separately reviewed T5/T3/T4/T2 held authorities; this policy
  entry alone is not an execution implementation.

The supplemental policy still sets `research_compute_allowed=true`,
`paper_result_reporting_allowed=true`, and
`data_redistribution_allowed=false`. T4 never opens `train.pt`,
`validation.pt`, `test.pt`, or a CSV. Its evidence root contains aggregate
metrics and provenance hashes only—no SMILES, molecule identifiers, or
per-example predictions. `last.pt` remains terminal checkpoint/reload evidence;
downstream inference uses the selected `model.pt`.

Both policy files, all T2 checkpoint children, the T3 evidence closure,
`model.pt`, and the cache children are read relative to retained root-to-leaf
physical descriptors. T3/T4 output may be created only as a direct fresh child
of `$AUTODL_ARTIFACT_ROOT/gnn_oracles/tastemolnet/gine/seed7`, with the exact
`calibrated-*` or `t4-oracle-smoke-*` basename. Output parent/root FDs remain
held while all documents and their future-marker hash are prepared. The
complete input/output closure is revalidated while the marker is still absent;
the PASS marker is then created and fsynced as the final commit operation.

Later consumers can retain `hold_taste_stage_output(...)`, bind its exact
`checkpoint_dir`, checkpoint ID, full-byte inventory, stat inventory, and
`sha256sums.txt` SHA through `hold_taste_checkpoint_bundle(...)`, and retain
the exact T6 policy with `hold_tastemolnet_downstream_policy(...)`. This closes
the path reopen window while a downstream model or reward adapter loads.

The thin AutoDL entrypoints are:

```bash
RUN_TASTEMOLNET=1 \
TASTEMOLNET_T2_BUNDLE=/absolute/immutable/t2-bundle \
TASTEMOLNET_GRAPH_CACHE_ROOT=/absolute/immutable/graph-cache-root \
scripts/autodl/run_tastemolnet_gnn_calibration_adoption.sh

RUN_TASTEMOLNET=1 \
TASTEMOLNET_T2_BUNDLE=/absolute/immutable/t2-bundle \
TASTEMOLNET_T3_OUTPUT=/absolute/passed/t3-evidence \
TASTEMOLNET_GRAPH_CACHE_ROOT=/absolute/immutable/graph-cache-root \
scripts/autodl/run_tastemolnet_gnn_oracle_smoke.sh
```

Both commands require fresh outputs. The second waits for an idle physical GPU
1 and binds its UUID before the visible `cuda:0` process is launched.

Full training also requires a private `--training-state-dir` outside the
immutable classifier output and outside every prepared/cache root.  At the end
of each epoch it atomically checkpoints the current model, AdamW state, best
validation state, early-stop counter, metric history, and Python/NumPy/Torch
CPU/CUDA RNG states.  The JSON checkpoint manifest is published only after the
state file is fsynced, and only the current and previous epochs are retained;
each safe deletion is recorded in `checkpoint_cleanup.json`.  A resumed worker
must match the complete canonical configuration, every config-file SHA,
dotlist/CLI overrides, clean Git commit/tree/source hashes, Python/Torch/CUDA
runtime, physical GPU-1 UUID, and original input/policy contract exactly.  The
output parent is held by an inode-bound directory descriptor, lock, sentinel,
and contract claim for the full training/finalization lifetime.  The classifier
bundle uses the single deterministic sibling
`.<output>.finalizing-<contract-sha>`; an empty mkdir-before-claim crash may be
reclaimed, a nonempty unclaimed root is rejected, partial owned contents are
inventoried and cleanup-receipted, and a completed inventory is published with
Linux `renameat2(RENAME_NOREPLACE)`.  Claim/completion sidecars remain
hash-bound to the terminal training receipt.  Thus a process loss cannot turn
a partial bundle into a scientific PASS or overwrite an existing output.

One dedicated persistent Taste controller owns that worker.  Its immutable
spec binds the clean project commit/tree, reviewed worker wrapper and SHA,
Python, all three config SHAs, exact argv, frozen scientific environment,
policy receipt, private prepared/cache authority, controller CID, and the
output/state roots.  A durable exec-startup barrier closes the register/release
window; controller restart adopts only the recorded live PID generation, and
at most one genuine process-loss retry may reuse the same state root.  GPU-1
waiting stays inside the same worker generation with a fixed deadline and
bounded controller event/worker logs.  Terminal publication holds and
revalidates the state root, named writer lock, output parent, finalization
claim, complete bundle/policy audit, and file SHA/stat inventories through the
final `PASS` no-replace write.

## Redistribution boundary

Private research roots may contain the source CSV, prepared rows, graph-cache
payloads, per-example predictions, and model artifacts. They must never be
used as a public artifact root.

Public release is limited to a separate fresh, manifest-closed, sanitized root
containing approved aggregate metrics, aggregate tables/figures, method
configuration, and provenance hashes. The following remain forbidden:

- raw or cleaned dataset tables;
- full or reconstructable SMILES/label tables;
- graph-cache payloads;
- molecule identifiers, SMILES, or per-example predictions;
- archives or opaque bundles that could hide any of the above;
- trained model release under this policy.

Every candidate public root must pass the read-only audit at the exact public
entrypoint:

```bash
PYTHONPATH=$PWD python scripts/audit_public_artifact_no_dataset_redistribution.py \
  --config configs/hpc.yaml \
  --policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --expected-policy-sha256 <active-policy-sha256> \
  --public-root /absolute/fresh/sanitized-public-root \
  --prepared-root /absolute/existing/prepared-root \
  --graph-cache-root /absolute/existing/graph-cache-root \
  --output /absolute/fresh/public-artifact-audit.json
```

The candidate root itself contains the exact
`public_release_manifest.json`; the audit output must be outside that
manifest-closed root.

The audit rejects symlinks, special files, hardlinks, unregistered files,
case-colliding names, protected data/cache hashes even when renamed, private
paths, molecule-level fields, and unsupported roles. Its success marker means
only that the inspected public artifact contains no detected redistributable
dataset material; it is not a licence marker.

## Active policy audit and controller-fragment commands

The policy/data authority audit is read-only with respect to the existing data
and cache roots and writes only to a fresh audit root:

```bash
PYTHONPATH=$PWD python scripts/audit_tastemolnet_research_policy.py \
  --config configs/hpc.yaml \
  --policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --expected-policy-sha256 b370ed9655f0a566b3615fc321c547945dd73fcee27d637110b801a766e1ca1b \
  --prepared-root /absolute/existing/prepared-root \
  --graph-cache-root /absolute/existing/graph-cache-root \
  --output-dir /absolute/fresh/policy-audit-root \
  --require-active
```

With the checked-in policy this emits
`TASTE_RESEARCH_POLICY_V2_PASS` and
`TASTE_NO_DATA_REDISTRIBUTION_GUARD_PASS`. Neither marker is a licence PASS.

The runnable GINE fragment requires the existing private authority and its
fresh policy receipt:

```bash
PYTHONPATH=$PWD python scripts/autodl/build_tastemolnet_gine_research_tasks.py \
  --config configs/hpc.yaml \
  --policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --expected-policy-sha256 b370ed9655f0a566b3615fc321c547945dd73fcee27d637110b801a766e1ca1b \
  --prepared-root /absolute/existing/prepared-root \
  --graph-cache-root /absolute/existing/graph-cache-root \
  --policy-receipt /absolute/fresh/policy-audit-root/tastemolnet_policy_receipt.json \
  --expected-output-root /absolute/fresh/future-science-root \
  --output /absolute/fresh/tastemolnet-gine-active.json \
  --require-active
```

The resulting task has `enabled=true`, `run_tastemolnet=1`, a validated policy
receipt, exact existing prepared/cache roots, physical-GPU-1 exclusivity, and
a fresh science root.  It launches the dedicated controller wrapper rather
than the scientific worker directly:

```bash
bash scripts/autodl/run_tastemolnet_gine_controller.sh

PYTHONPATH=$PWD python scripts/autodl/run_tastemolnet_gine_controller.py status \
  --controller-root /absolute/existing/taste-controller-root
```

The wrapper requires the fragment-provided
`TASTEMOLNET_GINE_CONTROLLER_CID`, `TASTEMOLNET_GINE_CONTROLLER_ROOT`,
`TASTEMOLNET_GNN_FULL_OUTPUT`, and
`TASTEMOLNET_GNN_TRAINING_STATE_ROOT`, sets the audited four-GPU scheduler
ceiling explicitly, and resumes only that same physical controller root.  The
paired `scripts/slurm/run_tastemolnet_gine_controller.sh` is deliberately a
static HPC refusal and never starts Taste science. The legacy binary
licence audit remains historical and can emit only
`BLOCKED_LICENSE_REVIEW`; it cannot authorize this scoped route.

The fresh main-table launcher is
`scripts/autodl/launch_tastemolnet_main_v1.sh`. It creates the policy adoption
receipt under `control/tastemolnet-main-v1`, records the old block as
`SUPERSEDED_POLICY_V1`, initializes the T0--T16 evidence skeleton, and starts
the persistent GINE controller through `nohup`. It freezes
`RUN_GNN_ABLATION=0`, `MAX_CONCURRENT_TASTE_FULL=2`, a 20-GiB planning
reservation, and `MIN_FREE_AFTER_RESERVATIONS_GB=100`. It does not modify the
historical controller or the main-result matrix.
