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

The checked-in machine-readable policy is an **inactive activation template**:

```text
authorization_state=PENDING_ROOT_ACTIVATION
research_compute_allowed=false
paper_result_reporting_allowed=false
RUN_TASTEMOLNET=0
```

It therefore cannot start a worker. A separate, independently reviewed root
activation must bind the explicit user direction, change the policy to its
exact active state, pin its raw and canonical hashes, and create only fresh
execution roots. This commit performs no activation, deployment, SSH action,
data preparation, cache rebuild, or experiment run.

The authoritative policy path is:

```text
configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml
```

The historical `LICENSE_REVIEW_REQUIRED` artifact remains immutable provenance.
It is superseded only as an execution decision by a future scoped policy
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
projection, an RF oracle, or a `1-label` target is invalid. Selection may use
train and validation only. Calibration and held-out test loading remain
disabled until later typed gates explicitly release them; the inactive full
GINE fragment exposes only test metadata hashes.

The future AutoDL route is dedicated, CPU-controller/GPU-worker scoped,
exclusive to physical GPU 2, and requires fresh controller and science roots.
HPC is forbidden for this campaign. Paired Slurm files exist only to satisfy
repository CLI parity and intentionally exit before running the command.

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

## Inactive audit and controller-template commands

The policy/data authority audit is read-only with respect to the existing data
and cache roots and writes only to a fresh audit root:

```bash
PYTHONPATH=$PWD python scripts/audit_tastemolnet_research_policy.py \
  --config configs/hpc.yaml \
  --policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --prepared-root /absolute/existing/prepared-root \
  --graph-cache-root /absolute/existing/graph-cache-root \
  --output-dir /absolute/fresh/policy-audit-root
```

With the checked-in policy this emits
`TASTEMOLNET_POLICY_READY_EXECUTION_DISABLED`, never PASS. Supplying
`--require-active` fails before creating the output root.

The disabled GINE fragment can be generated without data paths or a policy
receipt:

```bash
PYTHONPATH=$PWD python scripts/autodl/build_tastemolnet_gine_research_tasks.py \
  --config configs/hpc.yaml \
  --policy configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml \
  --expected-output-root /absolute/fresh/future-science-root \
  --output /absolute/fresh/tastemolnet-gine-disabled.json
```

The resulting task has `enabled=false`, `command=null`,
`run_tastemolnet=0`, and no live data authority. A future activation must use
`--require-active`, a validated policy receipt, exact existing prepared/cache
roots, and the independently reviewed dedicated controller. The legacy binary
licence audit remains historical and can emit only
`BLOCKED_LICENSE_REVIEW`; it cannot authorize this scoped route.
