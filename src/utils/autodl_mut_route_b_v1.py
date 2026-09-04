"""Fail-closed task specification for fresh Mut ComRecGC trace-off Route B.

Route B is expensive.  A spec can be sealed only when a same-contract
trace-on/off audit has classified a real scientific-state divergence and a
separate launch receipt explicitly authorizes the fresh 50k generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping
from uuid import UUID

from .autodl_mut_first_divergence_v1 import file_sha256, stable_sha256


SCHEMA = "mut_traceoff_route_b_task_spec_v1"
AUTHORIZATION_SCHEMA = "mut_traceoff_route_b_launch_authorization_v1"
ROUTE_B_EXECUTION_COMMIT = "66487c062c86d53ef2f762ce04d0fb965af5af08"
M_MAX = 50_000
CANDIDATE_CAPACITY = 100_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class MutRouteBSpecError(RuntimeError):
    """A Route-B launch contract is incomplete, stale, or unauthorized."""


def _absolute(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise MutRouteBSpecError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise MutRouteBSpecError(f"{field} must be normalized and absolute")
    return path


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MutRouteBSpecError(f"{field} is missing or indirect: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutRouteBSpecError(f"{field} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MutRouteBSpecError(f"{field} must be one JSON object")
    return value


def _validate_self_hash(value: Mapping[str, Any], *, field: str, key: str) -> None:
    observed = value.get(key)
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise MutRouteBSpecError(f"{field}.{key} is not SHA-256")
    expected = stable_sha256({name: item for name, item in value.items() if name != key})
    if observed != expected:
        raise MutRouteBSpecError(f"{field} self hash changed")


def validate_route_b_evidence(
    value: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    audit = dict(value)
    if audit.get("schema_version") != "mut_post_same_contract_ab_decision_v1":
        raise MutRouteBSpecError(
            "Route B requires the sealed same-contract post-A/B decision"
        )
    _validate_self_hash(audit, field="route_b_evidence", key="decision_sha256")
    if audit.get("classification") != "SCIENTIFIC_STATE_DIVERGENCE":
        raise MutRouteBSpecError("Route B requires SCIENTIFIC_STATE_DIVERGENCE")
    if (
        audit.get("status") != "READY"
        or audit.get("branch") != "ROUTE_B_AUTHORIZATION_REQUIRED"
        or audit.get("route_b_evidence_eligible") is not True
        or audit.get("route_b_permitted") is not False
        or audit.get("historical_adoption_permitted") is not False
        or audit.get("requires_immutable_deployed_consumer") is not True
        or audit.get("fresh_50k_started") is not False
    ):
        raise MutRouteBSpecError("Route-B evidence is not the fail-closed A/B decision")
    first = audit.get("first_semantic_divergence_step")
    if not isinstance(first, int) or isinstance(first, bool) or not 1 <= first <= 500:
        raise MutRouteBSpecError("Route B requires a causal step-1..500 divergence")
    gate_path = _absolute(audit.get("same_contract_gate"), field="same_contract_gate")
    spec_path = _absolute(
        audit.get("same_contract_ab_spec"), field="same_contract_ab_spec"
    )
    if file_sha256(gate_path) != audit.get("same_contract_gate_sha256"):
        raise MutRouteBSpecError("Route-B same-contract gate bytes changed")
    if file_sha256(spec_path) != audit.get("same_contract_ab_spec_sha256"):
        raise MutRouteBSpecError("Route-B same-contract spec bytes changed")
    from .autodl_mut_post_ab_continuation_v1 import validate_same_contract_gate

    checked_gate = validate_same_contract_gate(
        _load_json(gate_path, field="same_contract_gate"), gate_path=gate_path
    )
    if checked_gate.get("summary_sha256") != audit.get(
        "same_contract_gate_summary_sha256"
    ):
        raise MutRouteBSpecError("Route-B gate summary binding changed")
    if check_files:
        from .autodl_mut_same_contract_ab_v1 import validate_same_contract_ab_spec

        validate_same_contract_ab_spec(
            _load_json(spec_path, field="same_contract_ab_spec"), check_files=True
        )
    return audit


def validate_launch_authorization(
    value: Mapping[str, Any],
    *,
    evidence_file_sha256: str,
    execution_commit: str,
    task_id: str,
    attempt_uuid: str,
    output_root: str,
    gpu_index: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    receipt = dict(value)
    if receipt.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise MutRouteBSpecError("Route-B authorization schema changed")
    _validate_self_hash(receipt, field="launch_authorization", key="receipt_sha256")
    if receipt.get("allow_fresh_traceoff_50k") is not True:
        raise MutRouteBSpecError("fresh trace-off 50k is not explicitly authorized")
    if receipt.get("evidence_file_sha256") != evidence_file_sha256:
        raise MutRouteBSpecError("Route-B authorization binds different evidence")
    if receipt.get("execution_commit") != execution_commit:
        raise MutRouteBSpecError("Route-B authorization binds different code")
    expected = {
        "task_id": task_id,
        "attempt_uuid": attempt_uuid,
        "output_root": output_root,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
    }
    changed = [key for key, item in expected.items() if receipt.get(key) != item]
    if changed:
        raise MutRouteBSpecError(f"Route-B authorization target changed: {changed}")
    return receipt


def build_route_b_spec(template: Mapping[str, Any], *, check_files: bool = True) -> dict[str, Any]:
    raw = dict(template)
    required = {
        "task_id",
        "attempt_uuid",
        "execution_commit",
        "repo_root",
        "python",
        "config_path",
        "upstream_root",
        "dataset_dir",
        "gnn_checkpoint",
        "distance_checkpoint",
        "evidence_path",
        "launch_authorization_path",
        "output_root",
        "checkpoint_root",
        "checkpoint_mirror_root",
        "lease_path",
        "gpu_lock_root",
        "gpu_uuid",
        "owner_runtime_root",
        "gpu_index",
    }
    if set(raw) != required:
        raise MutRouteBSpecError(
            f"Route-B template keys differ: missing={sorted(required - set(raw))}, "
            f"extra={sorted(set(raw) - required)}"
        )
    try:
        attempt = UUID(str(raw["attempt_uuid"]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise MutRouteBSpecError("attempt_uuid must be canonical UUIDv4") from exc
    if attempt.version != 4 or str(attempt) != raw["attempt_uuid"]:
        raise MutRouteBSpecError("attempt_uuid must be canonical UUIDv4")
    commit = str(raw["execution_commit"])
    if commit != ROUTE_B_EXECUTION_COMMIT:
        raise MutRouteBSpecError("Route B must use the A/B execution commit")
    paths = {
        field: _absolute(raw[field], field=field)
        for field in required
        if field
        not in {
            "task_id",
            "attempt_uuid",
            "execution_commit",
            "gpu_index",
            "gpu_uuid",
        }
    }
    evidence = _load_json(paths["evidence_path"], field="evidence")
    validate_route_b_evidence(evidence, check_files=check_files)
    evidence_sha = file_sha256(paths["evidence_path"])
    authorization = _load_json(
        paths["launch_authorization_path"], field="launch_authorization"
    )
    validate_launch_authorization(
        authorization,
        evidence_file_sha256=evidence_sha,
        execution_commit=commit,
        task_id=str(raw["task_id"]),
        attempt_uuid=str(raw["attempt_uuid"]),
        output_root=str(paths["output_root"]),
        gpu_index=int(raw["gpu_index"]),
        gpu_uuid=str(raw["gpu_uuid"]),
    )
    if raw["gpu_index"] != 0 or isinstance(raw["gpu_index"], bool):
        raise MutRouteBSpecError("Mut Route B is pinned to physical GPU0")
    if _GPU_UUID.fullmatch(str(raw["gpu_uuid"])) is None:
        raise MutRouteBSpecError("gpu_uuid must be one physical GPU UUID")
    if paths["lease_path"] == paths["gpu_lock_root"] / f"gpu-{raw['gpu_uuid']}.lock":
        raise MutRouteBSpecError("owner lease and global GPU UUID lock must differ")
    if check_files:
        for field in (
            "repo_root",
            "upstream_root",
            "dataset_dir",
        ):
            if not paths[field].is_dir() or paths[field].is_symlink():
                raise MutRouteBSpecError(f"{field} is absent or indirect")
        for field in (
            "python",
            "config_path",
            "gnn_checkpoint",
            "distance_checkpoint",
        ):
            if not paths[field].is_file():
                raise MutRouteBSpecError(f"{field} is absent")
        for field in (
            "output_root",
            "checkpoint_root",
            "checkpoint_mirror_root",
            "owner_runtime_root",
        ):
            if paths[field].exists() or paths[field].is_symlink():
                raise MutRouteBSpecError(f"{field} must be fresh")
        try:
            repo_head = subprocess.run(
                ["git", "-c", f"safe.directory={paths['repo_root']}", "-C", str(paths["repo_root"]), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
            upstream_head = subprocess.run(
                ["git", "-c", f"safe.directory={paths['upstream_root']}", "-C", str(paths["upstream_root"]), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise MutRouteBSpecError("cannot bind Route-B Git inputs") from exc
        if repo_head != commit:
            raise MutRouteBSpecError("Route-B execution worktree HEAD changed")
        if paths["config_path"] != paths["repo_root"] / "configs/hpc.yaml":
            raise MutRouteBSpecError("Route-B config escaped execution worktree")
        for field in ("repo_root", "upstream_root"):
            status = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={paths[field]}",
                    "-C",
                    str(paths[field]),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
            if status:
                raise MutRouteBSpecError(
                    f"Route-B Git input has dirty or shadow source: {field}"
                )
        dataset_files = {
            name: file_sha256(paths["dataset_dir"] / name)
            for name in ("dataset_summary.json", "generation_source_graphs.pt")
        }
    else:
        repo_head = commit
        upstream_head = "NOT_CHECKED"
        dataset_files = {}
    contract = {
        "dataset": "mutagenicity",
        "method": "comrecgc",
        "route": "fresh_trace_off_route_b",
        "M_MAX": M_MAX,
        "candidate_capacity": CANDIDATE_CAPACITY,
        "trace_enabled": False,
        "fresh_generation": True,
        "checkpoint_interval_steps": 500,
        "convergence_early_stop_allowed": False,
        "owner_stop": "RUN_TO_EXACT_M_MAX",
        "test_loaded": False,
        "calibration_loaded": False,
        "pair_store_reuse_allowed": False,
        "dbscan_reuse_allowed": False,
        "pair_store_recompute_after_candidate_universe_freeze": True,
        "dbscan_recompute_after_pair_store": True,
    }
    spec: dict[str, Any] = {
        "schema_version": SCHEMA,
        **raw,
        "evidence_file_sha256": evidence_sha,
        "launch_authorization_file_sha256": file_sha256(
            paths["launch_authorization_path"]
        ),
        "gnn_checkpoint_sha256": file_sha256(paths["gnn_checkpoint"])
        if check_files
        else str(raw.get("gnn_checkpoint_sha256", "NOT_CHECKED")),
        "distance_checkpoint_sha256": file_sha256(paths["distance_checkpoint"])
        if check_files
        else str(raw.get("distance_checkpoint_sha256", "NOT_CHECKED")),
        "config_sha256": file_sha256(paths["config_path"])
        if check_files
        else "NOT_CHECKED",
        "dataset_file_sha256s": dataset_files,
        "repo_head": repo_head,
        "upstream_commit": upstream_head,
        "contract": contract,
        "required_environment": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RUN_LLM_ABLATION": "0",
            "RUN_GNN_ABLATION": "0",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    spec["spec_sha256"] = stable_sha256(spec)
    return validate_route_b_spec(spec, check_files=check_files)


def validate_route_b_spec(
    raw: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != SCHEMA:
        raise MutRouteBSpecError("Route-B spec schema changed")
    _validate_self_hash(value, field="route_b_spec", key="spec_sha256")
    contract = value.get("contract")
    expected = {
        "dataset": "mutagenicity",
        "method": "comrecgc",
        "route": "fresh_trace_off_route_b",
        "M_MAX": M_MAX,
        "candidate_capacity": CANDIDATE_CAPACITY,
        "trace_enabled": False,
        "fresh_generation": True,
        "checkpoint_interval_steps": 500,
        "convergence_early_stop_allowed": False,
        "owner_stop": "RUN_TO_EXACT_M_MAX",
        "test_loaded": False,
        "calibration_loaded": False,
        "pair_store_reuse_allowed": False,
        "dbscan_reuse_allowed": False,
        "pair_store_recompute_after_candidate_universe_freeze": True,
        "dbscan_recompute_after_pair_store": True,
    }
    if contract != expected:
        raise MutRouteBSpecError("Route-B scientific contract changed")
    for field in (
        "repo_root",
        "python",
        "config_path",
        "upstream_root",
        "dataset_dir",
        "gnn_checkpoint",
        "distance_checkpoint",
        "evidence_path",
        "launch_authorization_path",
        "output_root",
        "checkpoint_root",
        "checkpoint_mirror_root",
        "lease_path",
        "gpu_lock_root",
        "owner_runtime_root",
    ):
        _absolute(value.get(field), field=field)
    if value.get("execution_commit") != ROUTE_B_EXECUTION_COMMIT:
        raise MutRouteBSpecError("Route B must use the A/B execution commit")
    if value.get("gpu_index") != 0 or isinstance(value.get("gpu_index"), bool):
        raise MutRouteBSpecError("Mut Route B is pinned to physical GPU0")
    if _GPU_UUID.fullmatch(str(value.get("gpu_uuid") or "")) is None:
        raise MutRouteBSpecError("Route-B physical GPU UUID changed")
    environment = value.get("required_environment")
    if not isinstance(environment, Mapping) or environment.get("PYTHONHASHSEED") != "0":
        raise MutRouteBSpecError("Route-B deterministic environment changed")
    if check_files:
        evidence_path = Path(value["evidence_path"])
        authorization_path = Path(value["launch_authorization_path"])
        if file_sha256(evidence_path) != value.get("evidence_file_sha256"):
            raise MutRouteBSpecError("Route-B evidence bytes changed")
        evidence = validate_route_b_evidence(
            _load_json(evidence_path, field="evidence")
        )
        if evidence.get("route_b_gate", {}).get("fresh_50k_started") is not False:
            raise MutRouteBSpecError("Route-B evidence already records a launch")
        if file_sha256(authorization_path) != value.get(
            "launch_authorization_file_sha256"
        ):
            raise MutRouteBSpecError("Route-B authorization bytes changed")
        validate_launch_authorization(
            _load_json(authorization_path, field="launch_authorization"),
            evidence_file_sha256=value["evidence_file_sha256"],
            execution_commit=value["execution_commit"],
            task_id=str(value["task_id"]),
            attempt_uuid=str(value["attempt_uuid"]),
            output_root=str(value["output_root"]),
            gpu_index=int(value["gpu_index"]),
            gpu_uuid=str(value["gpu_uuid"]),
        )
        for field in ("gnn_checkpoint", "distance_checkpoint"):
            if file_sha256(Path(value[field])) != value.get(f"{field}_sha256"):
                raise MutRouteBSpecError(f"{field} bytes changed")
        if file_sha256(Path(value["config_path"])) != value.get("config_sha256"):
            raise MutRouteBSpecError("Route-B config bytes changed")
        dataset_hashes = value.get("dataset_file_sha256s")
        if not isinstance(dataset_hashes, Mapping):
            raise MutRouteBSpecError("Route-B dataset hashes are absent")
        for name in ("dataset_summary.json", "generation_source_graphs.pt"):
            if file_sha256(Path(value["dataset_dir"]) / name) != dataset_hashes.get(name):
                raise MutRouteBSpecError(f"Route-B dataset input changed: {name}")
        try:
            repo_head = subprocess.run(
                ["git", "-c", f"safe.directory={value['repo_root']}", "-C", value["repo_root"], "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
            upstream_head = subprocess.run(
                ["git", "-c", f"safe.directory={value['upstream_root']}", "-C", value["upstream_root"], "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise MutRouteBSpecError("cannot revalidate Route-B Git inputs") from exc
        if repo_head != value.get("repo_head") or repo_head != value["execution_commit"]:
            raise MutRouteBSpecError("Route-B execution worktree HEAD changed")
        if upstream_head != value.get("upstream_commit"):
            raise MutRouteBSpecError("Route-B upstream COMRECGC HEAD changed")
        if Path(value["config_path"]) != Path(value["repo_root"]) / "configs/hpc.yaml":
            raise MutRouteBSpecError("Route-B config escaped execution worktree")
        for field in ("repo_root", "upstream_root"):
            status = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={value[field]}",
                    "-C",
                    str(value[field]),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
            if status:
                raise MutRouteBSpecError(
                    f"Route-B Git input has dirty or shadow source: {field}"
                )
    return value


def route_b_generation_command(spec: Mapping[str, Any]) -> list[str]:
    value = validate_route_b_spec(spec, check_files=False)
    output = Path(value["output_root"])
    return [
        str(value["python"]),
        "-I",
        "-B",
        str(Path(value["repo_root"]) / "scripts/baselines/comrecgc/run_generation.py"),
        "--config",
        str(value["config_path"]),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--route",
        "project",
        "--dataset",
        "mutagenicity",
        "--mode",
        "full",
        "--project-root",
        str(value["repo_root"]),
        "--upstream-root",
        str(value["upstream_root"]),
        "--dataset-dir",
        str(value["dataset_dir"]),
        "--gnn-checkpoint",
        str(value["gnn_checkpoint"]),
        "--distance-checkpoint",
        str(value["distance_checkpoint"]),
        "--output-dir",
        str(output),
        "--parent-limit",
        "1448",
        "--device",
        "cuda:0",
        "--batch-size",
        "128",
        "--graph-state-dir",
        str(output / "graph_state"),
        "--storage-guard-root",
        str(output),
        "--storage-check-every-steps",
        "500",
        "--storage-min-free-gib",
        "50",
        "--storage-min-free-ratio",
        "0.02",
        "--storage-min-free-inodes",
        "100000",
        "--checkpoint-root",
        str(value["checkpoint_root"]),
        "--checkpoint-mirror-root",
        str(value["checkpoint_mirror_root"]),
        "--checkpoint-interval-steps",
        "500",
        "--checkpoint-keep-last",
        "2",
        "--progress-interval-steps",
        "25",
    ]


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "CANDIDATE_CAPACITY",
    "M_MAX",
    "MutRouteBSpecError",
    "ROUTE_B_EXECUTION_COMMIT",
    "SCHEMA",
    "build_route_b_spec",
    "route_b_generation_command",
    "validate_launch_authorization",
    "validate_route_b_evidence",
    "validate_route_b_spec",
]
