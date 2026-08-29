# AutoDL main-table continuation sidecar

This is a bounded controller for the current four-method/four-dataset
continuation only. It is not a managed-execution successor and has no dynamic
task-registration API.

## Fixed responsibilities

The queue contains exactly these entries:

- `NEUROSED`: launch an explicitly supplied trainer only after the fixed-budget
  label manifest satisfies the assertions in the spec. The trainer is wrapped
  by the existing physical-UUID `gpu_lock.py` and uses the first idle and
  unlocked GPU in `[0, 1]`.
- `T9`: launch only
  `scripts/autodl/run_tastemolnet_comrecgc_smoke.sh`, on physical GPU1, with a
  newly generated UUIDv4 in the stage root, final root, and run ID. Return code
  75 terminally abandons all three identities; the next attempt allocates a
  different UUID.
- `T6`, `T7`, and `T8`: remain `BLOCKED_RELEASE` and are never invoked by this
  sidecar.
- `T10`: remains `WAITING_SMOKES`; aggregation is outside this sidecar.

It also observes, without modifying, the exact AIDS blocked route, BACE GCF
final markers, the BACE GlobalGCE PID generation, and the BACE ComRecGC PID
generation/progress document. The BACE ComRecGC step-17500 registration is
persisted immediately as `DURABLE_PENDING`. At the trigger it becomes
`READY_FOR_EXTERNAL_CONVERGENCE_CHECK`. A caller may inject the separately
reviewed read-only convergence function; the hook result is persisted, but
even `PASS` becomes only `AUDIT_PASS_AWAITING_SEPARATE_HANDOVER`.

The sidecar contains no process-termination operation and no matrix writer.
The AIDS handover is always recorded as forbidden while the exact route is
`BLOCKED`. `RUN_GNN_ABLATION` is required to be JSON `false` and is reset to
`0` in every launched child environment.

## Persistent files

The fresh `state_root` contains:

```text
controller_receipt.json
controller.lock
queue.json
state.json
heartbeat.json
events.jsonl
bace_comrecgc_convergence_registration.json
attempts/
  t9/<uuid>/attempt.json
  t9/<uuid>/terminal.json
  neurosed/<uuid>/attempt.json
  neurosed/<uuid>/terminal.json
```

JSON snapshots use temporary-write, file `fsync`, atomic replace, and parent
directory `fsync`. Events are append-locked and `fsync`ed. The hidden child
mode writes the exact wrapper/trainer return code to `terminal.json`; this is
what makes T9 return-code-75 abandonment survive a sidecar restart.

## Spec

The launch spec is deliberately explicit. The shortened example below shows
all structural fields; production values must use the exact live UUIDs, PID
start ticks, authority hashes, and paths from the same audit.

```json
{
  "schema_version": "autodl_main_table_continuation_spec_v1",
  "controller_id": "main-table-continuation-<uuid>",
  "state_root": "/autodl-fs/data/counterfactual-subgraph-runtime/control/main-table-continuation/<uuid>",
  "project_root": "/absolute/immutable/e07625e9/worktree",
  "runtime_root": "/autodl-fs/data/counterfactual-subgraph-runtime",
  "data_root": "/autodl-fs/data",
  "python": "/root/miniconda3/envs/smiles_pip118/bin/python",
  "config": "/absolute/immutable/e07625e9/worktree/configs/hpc.yaml",
  "entrypoint": "/absolute/immutable/e07625e9/worktree/scripts/autodl/run_main_table_continuation_sidecar.py",
  "poll_seconds": 60,
  "run_gnn_ablation": false,
  "lock_root": "/autodl-fs/data/counterfactual-subgraph-runtime/locks",
  "gpus": [
    {"index": 0, "uuid": "GPU-..."},
    {"index": 1, "uuid": "GPU-..."}
  ],
  "aids_exact": {
    "state": "BLOCKED",
    "handover_allowed": false,
    "controller_id": "exact-controller-id",
    "controller_pid": 1,
    "science_pid": 2,
    "checkpoint": "/absolute/exact/checkpoint.json",
    "blocker": "exact blocker from the current audit"
  },
  "observers": {
    "bace_gcf": {
      "pid": 3,
      "start_ticks": 4,
      "final_markers": ["/absolute/bace/gcf/final/PASS.json"]
    },
    "bace_globalgce": {"pid": 5, "start_ticks": 6},
    "bace_comrecgc": {
      "pid": 7,
      "start_ticks": 8,
      "trigger_step": 17500,
      "progress_json": "/absolute/bace/comrec/progress.json",
      "progress_pointer": "/last_checkpoint_step",
      "convergence_audit": {
        "resolved_config_path": "/absolute/bace/comrec/_native_aux/resolved_config.json",
        "trace_chunks_dir": "/absolute/bace/comrec/_native_aux/trace/selected_action_trace_chunks",
        "local_checkpoint_root": "/absolute/bace/comrec/_native_aux/checkpoints",
        "mirror_checkpoint_root": "/absolute/bace/comrec/_native_aux/checkpoint_mirror",
        "audit_parent": "/absolute/fresh/convergence/audits",
        "expected_config_sha256": "<sha256>",
        "expected_parent_ids_sha256": "<sha256>"
      }
    }
  },
  "blocked_taste": {
    "T6": {"state": "BLOCKED_RELEASE", "reason": "release authority false"},
    "T7": {"state": "BLOCKED_RELEASE", "reason": "release authority false"},
    "T8": {"state": "BLOCKED_RELEASE", "reason": "release authority false"}
  },
  "t9": {
    "enabled": true,
    "wrapper": "/absolute/immutable/e07625e9/worktree/scripts/autodl/run_tastemolnet_comrecgc_smoke.sh",
    "stage_parent": "/absolute/fresh-stage-parent",
    "final_parent": "/absolute/fresh-final-parent",
    "run_id_prefix": "taste-t9-comrecgc-m500",
    "fixed_environment": {
      "RUN_TASTEMOLNET": "1",
      "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
      "TASTE_PAPER_RESULTS_ALLOWED": "1",
      "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
      "RUN_GNN_ABLATION": "0",
      "AUTODL_PYTHON": "/root/miniconda3/envs/smiles_pip118/bin/python",
      "AUTODL_DATA_ROOT": "/autodl-fs/data",
      "TASTEMOLNET_T2_ADOPTION_ROOT": "/absolute/T2",
      "TASTEMOLNET_T2_ADOPTION_GATE_SHA256": "<sha256>",
      "TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256": "<sha256>",
      "TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256": "<sha256>",
      "TASTEMOLNET_T3_OUTPUT_ROOT": "/absolute/T3",
      "TASTEMOLNET_T4_OUTPUT_ROOT": "/absolute/T4",
      "TASTEMOLNET_TRAIN_CSV": "/absolute/train.csv",
      "COMRECGC_OFFICIAL_ROOT": "/absolute/COMRECGC"
    }
  },
  "neurosed": {
    "label_manifest": null,
    "trainer_argv": null
  }
}
```

When the trainer contract is ready, use a fresh sidecar spec/root and replace
the last object with an exact argv. The placeholders are substituted without a
shell:

```json
{
  "label_manifest": "/absolute/ged_label_manifest.json",
  "label_manifest_sha256": "<optional-sha256>",
  "label_assertions": {
    "/train_success_count": 5000,
    "/validation_success_count": 1000,
    "/ged_backend": "branch",
    "/state": "PASS",
    "/calibration_loaded": false,
    "/test_loaded": false
  },
  "trainer_argv": [
    "/root/miniconda3/envs/smiles_pip118/bin/python",
    "-B",
    "/absolute/train_fixed_budget_neurosed.py",
    "--output-dir",
    "{attempt_root}"
  ],
  "fixed_environment": {"RUN_GNN_ABLATION": "0"},
  "science_process_token": "train_fixed_budget_neurosed.py",
  "attempt_parent": "/absolute/fresh/neurosed/parent",
  "success_marker_template": "{attempt_root}/PASS"
}
```

## Commands

Run one validation/scheduling tick:

```bash
python scripts/autodl/run_main_table_continuation_sidecar.py \
  --config configs/hpc.yaml once --spec /absolute/continuation.json
```

Run the 60-second persistent loop:

```bash
python scripts/autodl/run_main_table_continuation_sidecar.py \
  --config configs/hpc.yaml run --spec /absolute/continuation.json
```

Read status without joining the controller lock:

```bash
python scripts/autodl/run_main_table_continuation_sidecar.py \
  --config configs/hpc.yaml status --state-root /absolute/state-root
```

The paired Slurm file is a static refusal because this controller is bound to
AutoDL procfs identities and physical UUID locks.
