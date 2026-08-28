# Managed Taste release v3 authority foundation

## Scope and naming

The compatibility module and launch filenames retain `taste_main_v2`, but the
security contract implemented here is the managed Taste release-v3 foundation.
It is an authority/monitor and a T4 foreground-runner substrate. It is **not**
the complete `main_completion_v4` scheduler and it does not authorize T5, T6,
T8, or T9 execution. Machine-readable scope and blockers live in
`docs/MANAGED_TASTE_RELEASE_V3_BLOCKERS.json`.

The only fixed GPU bindings in this commit are:

- `T4_ORACLE_SMOKE` -> physical GPU 1;
- `TASTE_GCF_NEUROSED` -> physical GPU 2.

GPU 0 and GPU 3 are never eligible. The NeuroSED name reserves only authority;
it does not implement or authorize official fixed-budget NeuroSED training.

## External launch trust root

`launch_taste_main_v2.sh launch` starts a Python supervisor. The supervisor,
not the controller child, performs the hardened clean-Git and policy audit. It
spawns the child with two inherited one-way pipes:

1. the child reports its process snapshot;
2. the supervisor independently checks the Linux `/proc` generation, writes an
   immutable launcher receipt, and sends its exact path/SHA back;
3. only then may the child create its immutable controller receipt.

The launcher and controller must be distinct parent/child generations. A
self-signed launcher receipt is rejected. Before reporting success, the
supervisor holds the launcher receipt, controller receipt, full heartbeat
chain, fresh terminal heartbeat, and live controller generation, then writes
`launcher_ready.json`. It exposes the immutable sequence-1 anchor separately
from the fresh terminal heartbeat, so a later generation cannot change T4's
H1 input. The shell treats raw file existence as insufficient.
Supervisor, controller child, and T4 science children run with `python -I -B`;
each absolute script then adds only its reviewed checkout root, so inherited
`PYTHONPATH` and user-site packages cannot precede the audit.

Launch requires a clean immutable checkout, at least 100 GiB persistent free
space, and these exact policy gates:

```text
RUN_TASTEMOLNET=1
TASTE_RESEARCH_COMPUTE_ALLOWED=1
TASTE_PAPER_RESULTS_ALLOWED=1
TASTE_DATA_REDISTRIBUTION_ALLOWED=0
PRIMARY_TASTE_SOURCE_LABEL=1
MIN_FREE_AFTER_RESERVATIONS_GB=100
SCHEDULER_POLL_SECONDS=60
AUTODL_MAX_GPUS=4
MAX_CONCURRENT_TASTE_FULL=2
AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
RUN_GNN_ABLATION=0
```

`AUTODL_RUNTIME_ROOT` must equal
`$AUTODL_DATA_ROOT/counterfactual-subgraph-runtime`, and
`AUTODL_CONTROL_ROOT` must equal `$AUTODL_RUNTIME_ROOT/control`. The receipt
records that data/runtime/control closure and the exact canonical
`$AUTODL_RUNTIME_ROOT/locks` namespace; alternate lock roots are rejected.

Status is read-only and is dispatched before the launch policy, disk, and
clean-checkout gates:

```bash
TASTE_MAIN_V2_CONTROLLER_ROOT=/absolute/controller/root \
  bash scripts/autodl/launch_taste_main_v2.sh status
```

No production controller or science workload is deployed or launched by this
repository change; only bounded protocol tests spawn synthetic controller
children.

## Immutable evidence

Each UUID controller root contains append-only evidence:

```text
controller_receipt.json
heartbeats/<sequence>-<uuid>.json
gpu_leases/<task>-<lease_uuid>.json
gpu_lease_activations/<lease>-<sequence>-<uuid>.json
gpu_lease_renewals/<lease>-<sequence>-<uuid>.json
gpu_lease_releases/<lease>-<uuid>.json
gpu_locks/gpu<index>-<lease>-<activation>.json
.publication_staging/
```

Every public JSON is written to private staging, fsynced, and atomically
published without replacement. Heartbeats form a complete chain anchored at
sequence 1. Historical generations are validated for bytes, schema, sequence,
and predecessor SHA; only the latest terminal generation is subject to
freshness and live-PID checks. This permits long workers without treating an
old, valid anchor heartbeat as stale. The holder retains only the trust root
and terminal generation, so ten-second heartbeats do not leak one FD each.

Consumers call:

```python
hold_taste_main_v2_controller_authority(
    receipt_path,
    anchor_heartbeat_path,
    expected_controller_id,
    expected_git_commit,
    expected_git_tree,
    max_age_seconds,
    expected_launcher_receipt_path=...,
    expected_launcher_receipt_sha256=...,
    expected_receipt_sha256=...,
    expected_heartbeat_sha256=...,  # immutable historical anchor SHA
    expected_task_id=...,           # None for a CPU/no-GPU consumer
    expected_gpu_index=...,
    expected_gpu_uuid=...,
    expected_lease_uuid=...,
    expected_lease_sha256=...,
    expected_attempt_id=...,
    expected_generation_token=...,
    expected_activation_phase=...,
    expected_worker_process=...,      # outer runner's captured PID generation
)
```

A CPU/no-lease caller sets `expected_task_id=None` and must omit every GPU,
lease, attempt, generation, phase, and worker pin. This holder API is frozen
for later T5 integration, but this commit intentionally adds no T5 CLI.

`.evidence` separates `anchor_heartbeat_*` from the current terminal
`heartbeat_*`. `.revalidate()` rescans the complete chain to a newly fresh
terminal generation and revalidates all held authorities. Method verifiers
must call it immediately before terminal publication.

## GPU ownership and phases

The registered outer managed runner holds the existing canonical project
`GPUFileLock` keyed by the exact physical GPU UUID. It holds that FD across the
scientific worker, `SEALED` handoff, independent verifier, and final publish.
This lock remains held if the controller dies. The controller separately holds
a canonical UUID-keyed Taste coordination lock, preventing a second Taste
controller from acknowledging the same GPU.

The append-only activation phases are:

```text
WORKER_ACTIVE -> WAITING_VERIFIER -> VERIFIER_ACTIVE -> RELEASE_REQUESTED
```

Science children wait for a fresh heartbeat acknowledging their exact
attempt/generation, live runner process, child lineage, phase, lease, physical
index/UUID, and both locks. Phase order cannot jump, attempt/runner identity
cannot change across the chain, and the controller refuses a transition while
any prior scientific child generation remains live. Renewals are append-only predecessor chains and
become effective only when a controller heartbeat includes them. A clean
release is requested by the runner only after the verifier has exited and the
final publication exists; the controller independently proves all prior child
generations have exited, then writes an immutable release ACK before either
lock is released. Audit generations remain after unlock, permitting a later
lease to reuse the GPU without erasing history.

Expiry, identity drift, broken chains, or heartbeat loss never sends a signal.
The controller publishes `QUARANTINED` and retains its coordination lock. A
managed runner waits for any already-spawned child to exit naturally before it
can return an error; it never kills that child.

## AutoDL commands

After exporting the policy gates and canonical runtime roots:

```bash
bash scripts/autodl/launch_taste_main_v2.sh launch
bash scripts/autodl/launch_taste_main_v2.sh request-t4-lease
bash scripts/autodl/launch_taste_main_v2.sh request-neurosed-lease
```

`request-t4-lease` requires `TASTE_T4_PHYSICAL_GPU_UUID` for GPU 1;
`request-neurosed-lease` requires `TASTE_NEUROSED_PHYSICAL_GPU_UUID` for GPU 2.
The paired Slurm files always refuse execution because Taste science is
AutoDL-only.
