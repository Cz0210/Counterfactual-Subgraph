#!/usr/bin/env python3
"""Seal the authorized GPU1 T12 accelerated owner and successor binding."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import (  # noqa: E402
    TASK_SPEC_PATH_TOKEN,
    atomic_json,
    canonical_bytes,
    file_sha256,
    materialize_task_spec_path,
    seal_spec,
)
from src.utils.tastemolnet_t12_accelerated_from250 import (  # noqa: E402
    build_scientific_source_equivalence,
    build_prebound_continuation,
    build_promotion_blocker,
    validate_reference_step250,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--reference-task-spec", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--repo-root", type=_absolute, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--python", type=_absolute, required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-lease", type=_absolute, required=True)
    parser.add_argument("--owner-control-root", type=_absolute, required=True)
    parser.add_argument("--accelerated-root", type=_absolute, required=True)
    parser.add_argument("--full-root", type=_absolute, required=True)
    parser.add_argument("--postprocess-root", type=_absolute, required=True)
    parser.add_argument("--publisher-root", type=_absolute, required=True)
    parser.add_argument("--matrix-authority-root", type=_absolute, required=True)
    parser.add_argument("--disposable-index-root", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("T12 accelerated spec root must be fresh")
    try:
        args.disposable_index_root.relative_to(Path("/root/autodl-tmp"))
    except ValueError as exc:
        raise ValueError(
            "T12 disposable history index must use /root/autodl-tmp local scratch"
        ) from exc
    if args.gpu_index != 1:
        raise ValueError("T12 accelerated branch is authorized only on GPU1")
    evidence = validate_reference_step250(task_spec_path=args.reference_task_spec)
    reference = json.loads(args.reference_task_spec.read_text(encoding="utf-8"))
    reference_contract = reference["science_contract"]
    current_official_root = args.repo_root / "baselines/gcfexplainer_official"
    source_equivalence = build_scientific_source_equivalence(
        repo_root=args.repo_root,
        reference_commit=evidence["reference_execution_commit"],
        current_commit=args.execution_commit,
    )
    source_equivalence_path = args.output_root / "scientific_source_equivalence.json"
    source_equivalence_file_sha256 = hashlib.sha256(
        canonical_bytes(source_equivalence) + b"\n"
    ).hexdigest()
    dispatch_uuid = str(uuid4())
    task_id = f"t12-accelerated-from250-{dispatch_uuid[:8]}"
    owner_runtime = args.owner_control_root / task_id / "runtime"
    spec_path = args.output_root / f"{task_id}.json"
    input_roots = dict(reference["input_roots"])
    input_hashes = dict(reference["input_hashes"])
    reference_official_root = Path(reference_contract["official_root"])
    replaced_official_role = False
    for role, value in list(input_roots.items()):
        if Path(value) == reference_official_root:
            input_roots[role] = str(current_official_root)
            replaced_official_role = True
    if not replaced_official_role:
        input_roots["current_official_gcf_root"] = str(current_official_root)
        input_hashes["current_official_gcf_root"] = source_equivalence[
            "current_inventory"
        ]["inventory_sha256"]
    additions = {
        "reference_task_spec": args.reference_task_spec,
        "reference_checkpoint_250": Path(evidence["checkpoint_manifest"]),
        "reference_checkpoint_250_payload": Path(evidence["checkpoint_payload"]),
        "reference_generation_receipt_250": Path(evidence["generation_receipt"]),
        "reference_run_identity": Path(evidence["reference_root"]) / "run_identity.json",
        "reference_history_prefix": Path(evidence["history_segment"]),
        "reference_first_seen_prefix": Path(evidence["first_seen_segment"]),
        "scientific_source_equivalence": source_equivalence_path,
    }
    for name, path in additions.items():
        input_roots[name] = str(path)
        input_hashes[name] = (
            source_equivalence_file_sha256
            if name == "scientific_source_equivalence"
            else file_sha256(path)
        )
    raw = {
        "schema_version": "ignored-until-sealed",
        "task_id": task_id,
        "task_kind": "T12_ACCELERATED_FROM_CHECKPOINT250",
        "attempt_uuid": dispatch_uuid,
        "repo_root": str(args.repo_root),
        "execution_commit": args.execution_commit,
        "python": str(args.python),
        "entrypoint": str(
            args.repo_root / "scripts/autodl/run_t12_accelerated_from250_v1.py"
        ),
        "config_path": str(args.repo_root / "configs/hpc.yaml"),
        "config_sha256": "0" * 64,
        "manifest_path": str(Path(evidence["reference_root"]) / "run_identity.json"),
        "manifest_sha256": file_sha256(
            Path(evidence["reference_root"]) / "run_identity.json"
        ),
        "input_roots": input_roots,
        "input_hashes": input_hashes,
        "output_root": str(args.accelerated_root),
        "gpu_request": {
            "index": args.gpu_index,
            "uuid": args.gpu_uuid,
            "lease_path": str(args.gpu_lease),
            "lease_scope": "PROJECT_GPU_LOCK",
            "selection_policy": "authorized_gpu1_parallel_with_reference_gpu3",
        },
        "cpu_request": {"workers": 1, "planner": "serial", "writer": "single"},
        "memory_request": {"minimum_parent_headroom_bytes": 64 * 1024**3},
        "required_environment": {
            **reference["required_environment"],
            "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW": "1",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
        },
        "matrix_authority_root": str(args.matrix_authority_root),
        "expected_owner_command_sha256": "0" * 64,
        "expected_heartbeat_path": str(owner_runtime / "heartbeat.json"),
        "expected_pid_file": str(owner_runtime / "owner_pid.json"),
        "expected_terminal_path": str(owner_runtime / "terminal.json"),
        "resume_policy": "fork_exact_250_then_500_then_reload_510",
        "single_writer_policy": "fail_if_live_owner_or_output_writer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": "0" * 64,
        "arguments": ["owner", "--task-spec", TASK_SPEC_PATH_TOKEN],
        "owner_timeout_seconds": 120,
        "owner_probe": {"expected_cwd": str(args.repo_root), "max_age_seconds": 120},
        "science_contract": {
            "source_reference_task_spec": str(args.reference_task_spec),
            "source_reference_root": evidence["reference_root"],
            "source_checkpoint_250": evidence["checkpoint_manifest"],
            "source_checkpoint_250_manifest_sha256": evidence[
                "checkpoint_manifest_sha256"
            ],
            "source_checkpoint_250_payload_sha256": evidence[
                "checkpoint_payload_sha256"
            ],
            "source_checkpoint_250_state_sha256": evidence[
                "checkpoint_state_sha256"
            ],
            "source_checkpoint_250_rng_sha256": evidence["checkpoint_rng_sha256"],
            "source_first_seen_prefix_sha256": evidence[
                "first_seen_prefix_sha256"
            ],
            "source_first_seen_committed_bytes": evidence[
                "first_seen_committed_bytes"
            ],
            "source_history_prefix_sha256": evidence["history_prefix_sha256"],
            "source_history_committed_bytes": evidence["history_committed_bytes"],
            "source_science_attempt_id": evidence["reference_attempt_id"],
            "generation_token": evidence["generation_token"],
            "accelerated_root": str(args.accelerated_root),
            "accelerated_checkpoint_250": str(
                args.accelerated_root / "checkpoints/checkpoint-00000250.manifest.json"
            ),
            "accelerated_checkpoint_500": str(
                args.accelerated_root / "checkpoints/checkpoint-00000500.manifest.json"
            ),
            "accelerated_checkpoint_510": str(
                args.accelerated_root / "checkpoints/checkpoint-00000510.manifest.json"
            ),
            "reference_checkpoint_500": reference_contract["reference_checkpoint_500"],
            "reference_checkpoint_510": str(
                Path(evidence["reference_root"])
                / "checkpoints/checkpoint-00000510.manifest.json"
            ),
            "managed_neurosed_root": reference_contract["managed_neurosed_root"],
            "t3_root": reference_contract["t3_root"],
            "official_root": str(current_official_root),
            "reference_official_root": reference_contract["official_root"],
            "scientific_source_equivalence_receipt": str(
                source_equivalence_path
            ),
            "scientific_source_equivalence_file_sha256": (
                source_equivalence_file_sha256
            ),
            "scientific_source_equivalence_receipt_sha256": (
                source_equivalence["receipt_sha256"]
            ),
            "reference_execution_commit": source_equivalence[
                "reference_commit"
            ],
            "reference_execution_tree": source_equivalence["reference_tree"],
            "current_execution_commit": source_equivalence["current_commit"],
            "current_execution_tree": source_equivalence["current_tree"],
            "threshold_authority": reference_contract["threshold_authority"],
            "replay_gate": reference_contract["replay_gate"],
            "disposable_index_root": str(args.disposable_index_root),
            "dispatch_authorization": (
                "ALLOW_T12_ACCELERATED_FROM_CHECKPOINT250_NOW"
            ),
            "source_parent_count": 3778,
            "sample_size": 10000,
            "candidate_capacity": 100000,
            "seed": 7,
            "calibration_loaded": False,
            "test_loaded": False,
            "fresh_root_single_writer_reference_read_only": True,
            "acceleration_profile": "gpu1_local_nvme_ordered_single_writer_v2",
            "buffered_journal_enabled": True,
            "pure_executor_max_workers": 4,
            "ordered_collector_enabled": True,
            "single_writer_enabled": True,
            "warning_aggregation_enabled": True,
            "lightweight_progress_interval": 50,
            "durable_checkpoint_cursors": [500, 510],
            "intermediate_checkpoint_blocker": (
                "SOURCE_CHECKPOINT_AUTHENTICATES_250_500_510_SCHEDULE"
            ),
        },
    }
    raw["config_sha256"] = file_sha256(Path(raw["config_path"]))
    spec = seal_spec(materialize_task_spec_path(raw, spec_path))
    continuation = build_prebound_continuation(
        accelerated_spec_path=spec_path,
        accelerated_root=args.accelerated_root,
        full_root=args.full_root,
        postprocess_root=args.postprocess_root,
        publisher_root=args.publisher_root,
        matrix_authority_root=args.matrix_authority_root,
    )
    promotion_blocker = build_promotion_blocker()
    args.output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    atomic_json(source_equivalence_path, source_equivalence)
    atomic_json(spec_path, spec)
    atomic_json(args.output_root / "reference_step250_evidence.json", evidence)
    atomic_json(args.output_root / "continuation_prebinding.json", continuation)
    atomic_json(args.output_root / "promotion_blocker.json", promotion_blocker)
    atomic_json(
        args.output_root / "task_specs_manifest.json",
        {
            "schema_version": "tastemolnet_t12_accelerated_task_bundle_v1",
            "status": "READY_AUTHORIZED_GPU1",
            "task_spec": str(spec_path),
            "task_spec_sha256": file_sha256(spec_path),
            "continuation_prebinding": str(
                args.output_root / "continuation_prebinding.json"
            ),
            "reference_owner_must_not_be_restarted": True,
            "reference_owner_must_not_be_signaled": True,
            "gpu1_parallel_authorized": True,
            "gpu3_reference_must_continue": True,
            "scientific_source_equivalence": str(source_equivalence_path),
            "scientific_source_equivalence_file_sha256": (
                source_equivalence_file_sha256
            ),
            "scientific_source_equivalence_receipt_sha256": (
                source_equivalence["receipt_sha256"]
            ),
            "official_root": str(current_official_root),
            "promotion_state": promotion_blocker["status"],
            "promotion_blocker": str(args.output_root / "promotion_blocker.json"),
            "promotion_allowed": False,
        },
    )
    print(json.dumps({"status": "PASS", "task_spec": str(spec_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
