# Existing BACE LLM owner repair (2026-09-06)

This changes dispatch only: L1 off-the-shelf 7B, L2 the existing 300-update
PPO LoRA with matched regeneration, then L3 off-the-shelf 2B. No training,
SFT, borrowing, main-matrix count gate, or secondary-seed dependency is added.
The corrected GNN core remains mandatory. No main owner or reservation changes.

## CPU preparation

Run the existing `run_bace_native_llm.py prepare` in the deployed immutable
commit first; old sealed task specifications bound to another commit cannot
execute. Then use `run_bace_llm_successor.py --seal-dispatch-spec <fresh.json>`
with its existing readiness/output/GNN-archive arguments and `--resource-config`.
For the already independently accepted import, also pass the exact paths and
file SHAs in `--adopt-corrective-overlay[-sha256]` and
`--adopt-corrective-audit[-sha256]`. This reopens the two small proofs and hashes
the outer archive once; it does **not** unpack or replay the GNN evaluation.
It seals an owner acceptance with archive stat identity. Later invocations use
`--gnn-acceptance` and its SHA; changed archive identity requires re-adoption.

The resource config has exactly these fields (absolute paths, not old values):

```json
{
  "main_registry_path": "/autodl-fs/data/counterfactual-subgraph-runtime/control/final16-owner-registry/current.json",
  "main_ready_sources": ["<actual main successor/READY heartbeat path>"],
  "proc_root": "/proc",
  "cgroup_memory_root": "/sys/fs/cgroup/memory",
  "persistent_root": "/autodl-fs/data/counterfactual-subgraph-runtime",
  "gpu_lock_root": "<existing runtime layout locks_dir>",
  "minimum_gpu_free_mb": 40000,
  "maximum_idle_utilization_percent": 5,
  "minimum_memory_headroom_bytes": 68719476736,
  "minimum_persistent_free_bytes": 107374182400,
  "checkpoint_resume_pass": true
}
```

The thresholds shown are a conservative example, **not measured admission**.
Bind reviewed resource requirements and all current main READY sources when
preparing the real config. The sampler additionally reads every current task
and publisher heartbeat from the canonical registry. A missing publisher/READY
source, stale heartbeat, PID/start-tick mismatch, or failed primary owner blocks
launch. In particular a dead T13 with its declared GPU1 reservation is not idle.
No missing queue is treated as an empty queue.

## Existing queue hook

```bash
scripts/autodl/launch_llm_after_gnn_v1.sh --owner-dispatch \
  --gpu-index <physical-index> --gpu-uuid <physical-UUID> \
  --llm-dispatch-spec <sealed-dispatch.json> \
  --llm-dispatch-spec-sha256 <file-SHA> \
  --owner-output-root <fresh-owner-root> --wait-seconds 86400
```

This is a bounded waiting owner for **one next variant**, not a new dispatcher.
The existing outer queue invokes the same sealed dispatch again with a fresh
owner root after natural completion. PAUSED_AT_CALL_CHECKPOINT resumes the
same variant/root; it is never mistaken for a completed pool. The outer queue
must not advance after code 75 (paused/waiting) or a failure. A deployed live
outer-queue binding is an operational requirement, not claimed by these tests.

Admission requires 1200 seconds of consecutive real idle samples. GPU inventory,
heartbeats, PID generations, cgroup memory and filesystem headroom are reread
at most 60 seconds apart. Heartbeat values older than 120 seconds cannot be
restamped. The owner takes the existing GPUFileLock and a single existing
ProjectGPUSlotLock under `locks/llm-ablation`, samples again, and passes both
actual descriptors plus a one-use binding pipe. The physical UUID is the sole
CUDA_VISIBLE_DEVICES entry, mapping to logical `cuda:0`.

The child validates independent lock contention, owner/child PID generations,
inodes and nonce. Both descriptors are retained in that child but closed in
forked descendants and marked CLOEXEC before helper execution. Main READY or
reservation changes make the next generation boundary pause. SIGTERM to the
owner requests pause; it does not kill the child or unlock early. No bounded
120-second *GPU release latency* is claimed; 120 seconds is evidence freshness.

Owner artifacts: heartbeat.json, initial_resource_evidence.json (immutable),
resource_evidence.json (actual current observations), and terminal.json.
CPU-only subprocess tests exercise contention, all exit states, fork/exec FD
non-leakage and cooperative pause. They do not constitute a real GPU smoke.
