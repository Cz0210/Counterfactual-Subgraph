"""Typed release and retained-input authority for TasteMolNet T7 GCF smoke.

This is an execution boundary, not a controller.  It cannot allocate a GPU,
change a lock, repair a predecessor, verify itself, or publish a terminal
result.  It accepts only SHA-pinned managed-v2/NeuroSED/controller authority
and retains every scientific input until worker staging is ready to seal.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from src.baselines.tastemolnet_gcf_smoke import (
    DATASET,
    DISABLED_RELEASE_STATE,
    RELEASE_CONFIG_PATH,
    RELEASE_KEYS,
    RELEASE_PIN_FIELDS,
    RELEASE_SCHEMA,
    REPO_ROOT,
    SMOKE_ALPHA,
    SMOKE_CANDIDATE_CAPACITY,
    SMOKE_GPU_INDEX,
    SMOKE_PARENT_COUNT,
    SMOKE_SAMPLE_SIZE,
    SMOKE_SEED,
    SMOKE_SOURCE_POOL_LIMIT,
    SMOKE_STEPS,
    SMOKE_TELEPORT,
    STAGE,
    TasteGCFSmokeError,
    TasteGCFSmokeReleaseDisabled,
)


EXTERNAL_AUTHORITY_SCHEMA = "tastemolnet_t7_gcf_external_authority_v1"
CONTROLLER_RECEIPT_SCHEMA = "tastemolnet_t7_controller_receipt_v1"
GPU_LEASE_RECEIPT_SCHEMA = "tastemolnet_t7_gpu_lease_receipt_v1"
RELEASED_STATE = "RELEASED_BY_EXTERNAL_T7_EXECUTION_AUTHORITY"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")

IMPLEMENTATION_CRITICAL_BLOBS = frozenset(
    {
        "docs/AUTODL_TASTEMOLNET_T7_GCF_SMOKE.md",
        "scripts/run_tastemolnet_gcf_smoke.py",
        "scripts/autodl/run_tastemolnet_gcf_smoke.sh",
        "scripts/slurm/run_tastemolnet_gcf_smoke.sh",
        "src/baselines/tastemolnet_gcf_smoke.py",
        "src/utils/tastemolnet_t7_gcf_release.py",
        "src/utils/tastemolnet_gine_pass_adoption_v1.py",
        "src/eval/tastemolnet_gnn_stages.py",
        "src/utils/managed_execution_v2.py",
        "src/utils/terminal_publisher_v2.py",
        "src/utils/process_identity_v2.py",
        "src/utils/tastemolnet_t7_managed_v2.py",
        "src/utils/retained_readonly_file.py",
        "src/baselines/tastemolnet_multiclass_adapters.py",
        "src/baselines/gcfexplainer_mutagenicity_adapter.py",
        "src/baselines/gcfexplainer_mutagenicity_runtime.py",
        "baselines/gcfexplainer_official/importance.py",
        "baselines/gcfexplainer_official/vrrw.py",
    }
)


def _json(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGCFSmokeError(f"{field} is malformed JSON") from exc
    if type(value) is not dict:
        raise TasteGCFSmokeError(f"{field} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _HEX_64.fullmatch(value) is None:
        raise TasteGCFSmokeError(f"{field} must be one lowercase SHA-256")
    return value


def _sha1(value: Any, *, field: str) -> str:
    if type(value) is not str or _HEX_40.fullmatch(value) is None:
        raise TasteGCFSmokeError(f"{field} must be one full lowercase Git SHA-1")
    return value


def _absolute(value: Any, *, field: str) -> Path:
    if type(value) is not str or not value:
        raise TasteGCFSmokeError(f"{field} must be one absolute path string")
    requested = Path(value).expanduser()
    normalized = Path(os.path.abspath(requested))
    if not requested.is_absolute() or requested != normalized:
        raise TasteGCFSmokeError(f"{field} must be normalized and absolute")
    return normalized


def _native_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TasteGCFSmokeError(f"{field} must be one native JSON integer")
    return value


def assert_execution_released() -> dict[str, Any]:
    release = _json(RELEASE_CONFIG_PATH.read_bytes(), field="T7 release config")
    if set(release) != RELEASE_KEYS:
        raise TasteGCFSmokeError("TASTE_T7_RELEASE_CONFIG_KEYS_CHANGED")
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or type(release.get("release_enabled")) is not bool
        or type(release.get("release_state")) is not str
        or release.get("gpu_index") != SMOKE_GPU_INDEX
    ):
        raise TasteGCFSmokeError("TASTE_T7_RELEASE_CONFIG_INVALID")
    if release["release_enabled"] is not True:
        if release["release_state"] != DISABLED_RELEASE_STATE or any(
            release.get(field) is not None for field in RELEASE_PIN_FIELDS
        ):
            raise TasteGCFSmokeError("TASTE_T7_DISABLED_RELEASE_CONFIG_DRIFTED")
        raise TasteGCFSmokeReleaseDisabled(
            "TASTE_T7_GCF_EXECUTION_NOT_RELEASED"
        )
    if release["release_state"] != RELEASED_STATE:
        raise TasteGCFSmokeError("TASTE_T7_RELEASE_STATE_INVALID")
    _sha1(release.get("implementation_commit"), field="implementation_commit")
    _sha1(release.get("implementation_tree"), field="implementation_tree")
    for field in (
        "external_authority_sha256",
        "t2_receipt_sha256",
        "t2_gate_sha256",
        "t2_source_evidence_sha256",
        "t3_gate_sha256",
        "t3_root_inventory_sha256",
        "t4_gate_sha256",
        "t4_root_inventory_sha256",
        "controller_receipt_sha256",
        "gpu_lease_receipt_sha256",
        "managed_execution_v2_pass_sha256",
        "taste_gcf_neurosed_pass_sha256",
        "taste_gcf_neurosed_gate_sha256",
        "taste_gcf_neurosed_verification_sha256",
        "taste_gcf_neurosed_checkpoint_sha256",
        "taste_gcf_neurosed_feature_schema_sha256",
        "taste_gcf_neurosed_sha256s_sha256",
    ):
        _sha256(release.get(field), field=field)
    _absolute(
        release.get("external_authority_path"), field="external_authority_path"
    )
    _absolute(release.get("output_parent"), field="output_parent")
    for field in (
        "managed_execution_v2_pass_path",
        "taste_gcf_neurosed_final_root",
        "taste_gcf_neurosed_pass_path",
        "taste_gcf_neurosed_checkpoint_path",
        "managed_stage_root",
    ):
        _absolute(release.get(field), field=field)
    return release


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(REPO_ROOT),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CEILING_DIRECTORIES": str(REPO_ROOT.parent),
        },
    )
    return completed.stdout.strip()


def _reject_hidden_index_flags() -> None:
    tagged = _git_output("ls-files", "-v", "-z")
    for record in tagged.split("\0"):
        if not record:
            continue
        if len(record) < 3 or record[1] != " ":
            raise TasteGCFSmokeError("Taste T7 Git index inventory is malformed")
        tag = record[0]
        if tag == "S" or tag.islower():
            raise TasteGCFSmokeError(
                "Taste T7 execution checkout has skip-worktree/"
                "assume-unchanged entries"
            )


def verify_execution_checkout(release: Mapping[str, Any]) -> dict[str, str]:
    _reject_hidden_index_flags()
    if _git_output(
        "status", "--porcelain", "--untracked-files=all", "--ignored=matching"
    ):
        raise TasteGCFSmokeError(
            "Taste T7 execution checkout is not completely clean"
        )
    lineage = _git_output("rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(lineage) != 2 or lineage[1] != release["implementation_commit"]:
        raise TasteGCFSmokeError(
            "Taste T7 release is not a one-parent implementation successor"
        )
    if (
        _git_output("rev-parse", f"{lineage[1]}^{{tree}}")
        != release["implementation_tree"]
    ):
        raise TasteGCFSmokeError("Taste T7 implementation tree pin changed")
    changed = set(
        filter(
            None,
            _git_output(
                "diff", "--name-only", "--no-renames", lineage[1], "HEAD"
            ).splitlines(),
        )
    )
    if changed != {
        "configs/autodl/tastemolnet_t7_gcf_smoke_release_v1.json",
        "scripts/autodl/run_tastemolnet_gcf_smoke.sh",
    }:
        raise TasteGCFSmokeError(
            "Taste T7 release commit changed non-release files"
        )
    wrapper = (
        REPO_ROOT / "scripts/autodl/run_tastemolnet_gcf_smoke.sh"
    ).read_text(encoding="utf-8")
    assignments = [
        line.strip()
        for line in wrapper.splitlines()
        if line.strip().startswith("TASTE_T7_GCF_WRAPPER_RELEASED=")
    ]
    if assignments != ["TASTE_T7_GCF_WRAPPER_RELEASED=1"]:
        raise TasteGCFSmokeError("Taste T7 AutoDL wrapper is not released")
    return {
        "execution_commit": lineage[0],
        "execution_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "implementation_commit": lineage[1],
        "implementation_tree": release["implementation_tree"],
    }


def _verify_critical_blobs(authority: Mapping[str, Any]) -> None:
    blobs = authority.get("critical_blobs_sha256")
    if type(blobs) is not dict or set(blobs) != IMPLEMENTATION_CRITICAL_BLOBS:
        raise TasteGCFSmokeError("Taste T7 critical blob inventory changed")
    for relative, digest in blobs.items():
        _sha256(digest, field=f"critical_blobs_sha256.{relative}")
        source = REPO_ROOT / relative
        if (
            source.is_symlink()
            or not source.is_file()
            or _sha256_file(source) != digest
        ):
            raise TasteGCFSmokeError(f"Taste T7 critical blob changed: {relative}")
    official_root = _absolute(
        authority.get("official_root"), field="Taste T7 official root"
    )
    if official_root != REPO_ROOT / "baselines/gcfexplainer_official":
        raise TasteGCFSmokeError("Taste T7 official root is not integrated source")


def _hold_json(
    stack: ExitStack,
    path: Path,
    digest: str,
    *,
    field: str,
) -> tuple[Any, dict[str, Any]]:
    from src.utils.retained_readonly_file import hold_readonly_file

    held = stack.enter_context(
        hold_readonly_file(path, expected_sha256=digest)
    )
    return held, _json(held.read_bytes(), field=field)


def hold_external_authority(
    stack: ExitStack,
    release: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    path = _absolute(
        release["external_authority_path"], field="external authority"
    )
    held, authority = _hold_json(
        stack,
        path,
        release["external_authority_sha256"],
        field="external authority",
    )
    expected_keys = {
        "schema_version",
        "status",
        "stage",
        "dataset",
        "implementation_commit",
        "implementation_tree",
        "critical_blobs_sha256",
        "t2",
        "t3",
        "t4",
        "checkpoint",
        "train",
        "official_root",
        "controller_receipt_path",
        "controller_receipt_sha256",
        "gpu_lease_receipt_path",
        "gpu_lease_receipt_sha256",
        "gpu_index",
        "gpu_uuid",
        "cuda_visible_devices",
        "managed_execution_v2",
        "neurosed",
        "output_parent",
        "expected_output_root",
        "policy",
        "smoke_contract",
    }
    if set(authority) != expected_keys:
        raise TasteGCFSmokeError("Taste T7 external authority keys changed")
    exact = {
        "schema_version": EXTERNAL_AUTHORITY_SCHEMA,
        "status": "RELEASE_AUTHORIZED",
        "stage": STAGE,
        "dataset": DATASET,
        "implementation_commit": release["implementation_commit"],
        "implementation_tree": release["implementation_tree"],
        "controller_receipt_sha256": release["controller_receipt_sha256"],
        "gpu_lease_receipt_sha256": release["gpu_lease_receipt_sha256"],
        "gpu_index": SMOKE_GPU_INDEX,
        "cuda_visible_devices": "1",
        "output_parent": release["output_parent"],
    }
    if any(
        type(authority.get(key)) is not type(value)
        or authority.get(key) != value
        for key, value in exact.items()
    ):
        raise TasteGCFSmokeError("Taste T7 external authority values changed")
    if _GPU_UUID.fullmatch(str(authority.get("gpu_uuid") or "")) is None:
        raise TasteGCFSmokeError("Taste T7 external authority lacks a GPU UUID")
    if authority.get("managed_execution_v2") != {
        "pass_path": release["managed_execution_v2_pass_path"],
        "pass_sha256": release["managed_execution_v2_pass_sha256"],
        "stage_root": release["managed_stage_root"],
        "auto_terminate_uncontrolled_children": False,
    }:
        raise TasteGCFSmokeError(
            "Taste T7 managed execution v2 authority changed"
        )
    neurosed = authority.get("neurosed")
    if (
        type(neurosed) is not dict
        or set(neurosed)
        != {
            "final_root",
            "pass_path",
            "pass_sha256",
            "gate_sha256",
            "verification_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "feature_schema_sha256",
            "sha256s_sha256",
            "distance_threshold",
        }
        or neurosed.get("final_root")
        != release["taste_gcf_neurosed_final_root"]
        or neurosed.get("pass_path")
        != release["taste_gcf_neurosed_pass_path"]
        or neurosed.get("pass_sha256")
        != release["taste_gcf_neurosed_pass_sha256"]
        or neurosed.get("checkpoint_path")
        != release["taste_gcf_neurosed_checkpoint_path"]
        or neurosed.get("checkpoint_sha256")
        != release["taste_gcf_neurosed_checkpoint_sha256"]
        or neurosed.get("gate_sha256")
        != release["taste_gcf_neurosed_gate_sha256"]
        or neurosed.get("verification_sha256")
        != release["taste_gcf_neurosed_verification_sha256"]
        or neurosed.get("feature_schema_sha256")
        != release["taste_gcf_neurosed_feature_schema_sha256"]
        or neurosed.get("sha256s_sha256")
        != release["taste_gcf_neurosed_sha256s_sha256"]
        or isinstance(neurosed.get("distance_threshold"), bool)
        or not isinstance(neurosed.get("distance_threshold"), (int, float))
        or not math.isfinite(float(neurosed["distance_threshold"]))
        or float(neurosed["distance_threshold"]) < 0
    ):
        raise TasteGCFSmokeError("Taste T7 NeuroSED authority changed")
    _verify_critical_blobs(authority)
    if authority.get("policy") != {
        "research_compute_allowed": True,
        "paper_result_reporting_allowed": True,
        "data_redistribution_allowed": False,
    }:
        raise TasteGCFSmokeError("Taste T7 scoped data policy changed")
    if authority.get("smoke_contract") != {
        "parent_count": SMOKE_PARENT_COUNT,
        "source_pool_limit": SMOKE_SOURCE_POOL_LIMIT,
        "steps": SMOKE_STEPS,
        "sample_size": SMOKE_SAMPLE_SIZE,
        "candidate_capacity": SMOKE_CANDIDATE_CAPACITY,
        "alpha": SMOKE_ALPHA,
        "teleport": SMOKE_TELEPORT,
        "seed": SMOKE_SEED,
        "neurosed_status": "PASS_INPUT_REVALIDATED",
        "distance_status": "EVALUATED",
        "selector_status": "NOT_EVALUATED",
        "full_route_status": "NOT_EVALUATED",
        "paper_result_eligible": False,
    }:
        raise TasteGCFSmokeError("Taste T7 bounded smoke contract changed")
    return held, authority


def _proc_start_ticks(pid: int) -> int:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise TasteGCFSmokeError(
            "Taste T7 controller process is not live"
        ) from exc
    _prefix, separator, tail = value.rpartition(")")
    fields = tail.strip().split()
    if not separator or len(fields) <= 19:
        raise TasteGCFSmokeError("Taste T7 controller proc stat is malformed")
    return int(fields[19])


def _verify_controller_process(controller: Mapping[str, Any]) -> None:
    pid = _native_int(
        controller.get("controller_pid"), field="controller_pid", minimum=2
    )
    ticks = _native_int(
        controller.get("controller_start_ticks"),
        field="controller_start_ticks",
        minimum=1,
    )
    if _proc_start_ticks(pid) != ticks:
        raise TasteGCFSmokeError("Taste T7 controller PID identity changed")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise TasteGCFSmokeError(
            "Taste T7 controller argv identity is unavailable"
        ) from exc
    if not cmdline or hashlib.sha256(cmdline).hexdigest() != _sha256(
        controller.get("controller_cmdline_sha256"),
        field="controller_cmdline_sha256",
    ):
        raise TasteGCFSmokeError("Taste T7 controller argv identity changed")
    if _proc_start_ticks(pid) != ticks:
        raise TasteGCFSmokeError("Taste T7 controller PID changed during argv check")


def _gpu_runtime_evidence(expected_uuid: str) -> dict[str, Any]:
    if (
        os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != "1"
        or os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
    ):
        raise TasteGCFSmokeError(
            "Taste T7 GPU1 environment differs from typed authority"
        )
    completed = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        index, separator, uuid = line.partition(",")
        if not separator:
            raise TasteGCFSmokeError("Taste T7 GPU inventory is malformed")
        inventory[int(index.strip())] = uuid.strip()
    if inventory.get(SMOKE_GPU_INDEX) != expected_uuid:
        raise TasteGCFSmokeError("Taste T7 physical GPU1 UUID changed")
    return {
        "physical_gpu_index": SMOKE_GPU_INDEX,
        "gpu_uuid": expected_uuid,
        "cuda_visible_devices": "1",
    }


def hold_controller_gpu_receipts(
    stack: ExitStack,
    authority: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    controller_file, controller = _hold_json(
        stack,
        _absolute(
            authority["controller_receipt_path"], field="controller receipt"
        ),
        authority["controller_receipt_sha256"],
        field="controller receipt",
    )
    controller_keys = {
        "schema_version",
        "status",
        "stage",
        "dataset",
        "controller_cid",
        "controller_root",
        "run_id",
        "controller_pid",
        "controller_start_ticks",
        "controller_cmdline_sha256",
        "expected_output_root",
        "gpu_index",
        "gpu_uuid",
    }
    if set(controller) != controller_keys or any(
        controller.get(key) != value
        for key, value in {
            "schema_version": CONTROLLER_RECEIPT_SCHEMA,
            "status": "RELEASE_AUTHORIZED",
            "stage": STAGE,
            "dataset": DATASET,
            "expected_output_root": authority["expected_output_root"],
            "gpu_index": SMOKE_GPU_INDEX,
            "gpu_uuid": authority["gpu_uuid"],
        }.items()
    ):
        raise TasteGCFSmokeError("Taste T7 controller receipt changed")
    _verify_controller_process(controller)
    pid = _native_int(controller["controller_pid"], field="controller_pid", minimum=2)
    ticks = _native_int(
        controller["controller_start_ticks"],
        field="controller_start_ticks",
        minimum=1,
    )
    gpu_file, gpu = _hold_json(
        stack,
        _absolute(
            authority["gpu_lease_receipt_path"], field="GPU lease receipt"
        ),
        authority["gpu_lease_receipt_sha256"],
        field="GPU lease receipt",
    )
    gpu_keys = {
        "schema_version",
        "status",
        "stage",
        "run_id",
        "owner_pid",
        "owner_start_ticks",
        "gpu_index",
        "gpu_uuid",
        "lock_mode",
        "expected_output_root",
    }
    if set(gpu) != gpu_keys or any(
        gpu.get(key) != value
        for key, value in {
            "schema_version": GPU_LEASE_RECEIPT_SCHEMA,
            "status": "ACTIVE",
            "stage": STAGE,
            "run_id": controller["run_id"],
            "owner_pid": pid,
            "owner_start_ticks": ticks,
            "gpu_index": SMOKE_GPU_INDEX,
            "gpu_uuid": authority["gpu_uuid"],
            "lock_mode": "exclusive",
            "expected_output_root": authority["expected_output_root"],
        }.items()
    ):
        raise TasteGCFSmokeError("Taste T7 GPU lease receipt changed")
    _gpu_runtime_evidence(authority["gpu_uuid"])
    return controller_file, controller, gpu_file, gpu


def hold_t2_adoption(
    stack: ExitStack,
    authority: Mapping[str, Any],
    release: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from src.utils.tastemolnet_gine_pass_adoption_v1 import (
        hold_t2_gine_pass_adoption,
    )

    value = authority.get("t2")
    keys = {
        "adoption_root",
        "receipt_sha256",
        "gate_sha256",
        "source_evidence_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise TasteGCFSmokeError("Taste T7 T2 adoption authority keys changed")
    root = _absolute(value["adoption_root"], field="T2 adoption root")
    receipt_sha = _sha256(value["receipt_sha256"], field="T2 receipt SHA")
    gate_sha = _sha256(value["gate_sha256"], field="T2 gate SHA")
    source_sha = _sha256(
        value["source_evidence_sha256"], field="T2 source-evidence SHA"
    )
    if (
        receipt_sha != release["t2_receipt_sha256"]
        or gate_sha != release["t2_gate_sha256"]
        or source_sha != release["t2_source_evidence_sha256"]
    ):
        raise TasteGCFSmokeError("Taste T7 T2 release pins changed")
    held = stack.enter_context(
        hold_t2_gine_pass_adoption(
            root,
            expected_gate_sha256=gate_sha,
            expected_receipt_sha256=receipt_sha,
            expected_source_evidence_sha256=source_sha,
        )
    )
    result = held.revalidate()
    expected = {
        "status": "PASS",
        "state": "T2_GINE_FULL_PASS_ADOPTED",
        "adoption_root": str(root),
        "receipt_sha256": receipt_sha,
        "gate_sha256": gate_sha,
        "source_evidence_sha256": source_sha,
    }
    if any(
        result.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise TasteGCFSmokeError(
            "Taste T7 T2 adoption result differs from release"
        )
    return held, result


def hold_stages_and_checkpoint(
    stack: ExitStack,
    authority: Mapping[str, Any],
    release: Mapping[str, Any],
    t2_evidence: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, bytes], dict[str, Any]]:
    from src.eval.tastemolnet_gnn_stages import (
        hold_taste_checkpoint_bundle,
        hold_taste_stage_output,
    )

    held: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for label, expected_stage in (
        ("t3", "T3_GINE_CALIBRATED"),
        ("t4", "T4_ORACLE_SMOKE"),
    ):
        value = authority.get(label)
        if type(value) is not dict or set(value) != {
            "stage",
            "root",
            "gate_sha256",
            "root_inventory_sha256",
        }:
            raise TasteGCFSmokeError(
                f"Taste T7 {label.upper()} authority keys changed"
            )
        if value["stage"] != expected_stage:
            raise TasteGCFSmokeError(f"Taste T7 {label.upper()} stage changed")
        held[label] = stack.enter_context(
            hold_taste_stage_output(
                _absolute(value["root"], field=f"{label} root")
            )
        )
        evidence[label] = held[label].revalidate()
        if (
            evidence[label]["stage"] != expected_stage
            or evidence[label]["gate_sha256"] != value["gate_sha256"]
            or evidence[label]["root_inventory_sha256"]
            != value["root_inventory_sha256"]
            or evidence[label]["gate_sha256"]
            != release[f"{label}_gate_sha256"]
            or evidence[label]["root_inventory_sha256"]
            != release[f"{label}_root_inventory_sha256"]
        ):
            raise TasteGCFSmokeError(
                f"Taste T7 {label.upper()} held authority changed"
            )
    t2_binding_sha = _canonical_sha256(t2_evidence)
    for label in ("t3", "t4"):
        if (
            evidence[label]["t2_adoption_gate_sha256"]
            != t2_evidence["gate_sha256"]
            or evidence[label]["t2_adoption_receipt_sha256"]
            != t2_evidence["receipt_sha256"]
            or evidence[label]["t2_adoption_binding_sha256"]
            != t2_binding_sha
        ):
            raise TasteGCFSmokeError(
                f"Taste T7 {label.upper()} differs from held T2 receipt binding"
            )
    checkpoint_fields = {
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
    }
    if any(
        evidence["t3"][field] != evidence["t4"][field]
        for field in checkpoint_fields
    ):
        raise TasteGCFSmokeError(
            "Taste T7 T3/T4 do not bind one common frozen GINE"
        )
    if (
        t2_evidence["formal_bundle_root"]
        != evidence["t3"]["checkpoint_dir"]
        or t2_evidence["formal_bundle_model_sha256"]
        != evidence["t3"]["checkpoint_id"]
        or t2_evidence["formal_bundle_sha256s_sha256"]
        != evidence["t3"]["checkpoint_sha256s_sha256"]
    ):
        raise TasteGCFSmokeError(
            "Taste T7 T3/T4 checkpoint differs from held T2 formal bundle"
        )
    checkpoint_value = authority.get("checkpoint")
    if (
        type(checkpoint_value) is not dict
        or set(checkpoint_value) != checkpoint_fields
        or any(
            checkpoint_value[field] != evidence["t3"][field]
            for field in checkpoint_fields
        )
    ):
        raise TasteGCFSmokeError("Taste T7 checkpoint differs from T3/T4")
    checkpoint = stack.enter_context(
        hold_taste_checkpoint_bundle(
            _absolute(
                checkpoint_value["checkpoint_dir"], field="checkpoint_dir"
            ),
            expected_stage_evidence=evidence["t3"],
        )
    )
    payload_names = (
        "model.pt",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "test_evaluation_status.json",
        "temperature_scaling.json",
    )
    payloads = {
        name: checkpoint.read_frozen_gine_payload(name)
        for name in payload_names
    }
    checkpoint.revalidate()
    return held["t3"], held["t4"], checkpoint, payloads, evidence


def hold_train_payload(
    stack: ExitStack,
    authority: Mapping[str, Any],
    checkpoint_payloads: Mapping[str, bytes],
) -> tuple[Any, bytes, dict[str, Any]]:
    split = _json(
        checkpoint_payloads["split_manifest.json"], field="split manifest"
    )
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != DATASET
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGCFSmokeError("Taste T7 split isolation changed")
    files = split.get("files")
    manifest = split.get("train_manifest")
    if (
        type(files) is not dict
        or type(files.get("train")) is not dict
        or type(manifest) is not dict
    ):
        raise TasteGCFSmokeError("Taste T7 train split authority is malformed")
    train_path = _absolute(files["train"].get("path"), field="frozen train CSV")
    train_sha = _sha256(
        files["train"].get("sha256"), field="frozen train CSV SHA"
    )
    records = _native_int(
        manifest.get("num_records"), field="train num_records", minimum=1
    )
    counts = manifest.get("label_counts")
    expected = {
        "path": str(train_path),
        "sha256": train_sha,
        "num_records": records,
        "label_counts": counts,
    }
    if authority.get("train") != expected:
        raise TasteGCFSmokeError(
            "Taste T7 train authority differs from checkpoint"
        )
    from src.utils.retained_readonly_file import hold_readonly_file

    held = stack.enter_context(
        hold_readonly_file(train_path, expected_sha256=train_sha)
    )
    data = held.read_bytes()
    held.revalidate()
    return held, data, expected


def hold_managed_neurosed_predecessors(
    stack: ExitStack,
    release: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, dict[str, Any], float]:
    """Retain and validate one generic managed-v2 NeuroSED final root."""

    from src.utils.managed_execution_v2 import GATE_SCHEMA, VERIFICATION_SCHEMA
    from src.utils.process_identity_v2 import canonical_json_bytes
    from src.utils.retained_readonly_file import hold_readonly_file
    from src.utils.managed_final_consumer_v2 import hold_verified_managed_final

    managed_pass = stack.enter_context(
        hold_readonly_file(
            _absolute(
                release["managed_execution_v2_pass_path"],
                field="managed execution v2 PASS",
            ),
            expected_sha256=release["managed_execution_v2_pass_sha256"],
        )
    )
    if managed_pass.read_bytes() != b"[MANAGED_EXECUTION_V2_PASS]\n":
        raise TasteGCFSmokeError("managed execution v2 PASS marker changed")

    final_root = _absolute(
        release["taste_gcf_neurosed_final_root"],
        field="Taste NeuroSED managed final root",
    )
    neurosed_final = stack.enter_context(
        hold_verified_managed_final(
            final_root,
            required_relative_paths=(
                "artifacts/best.pt",
                "artifacts/feature_schema.json",
                "artifacts/sha256sums.txt",
            ),
        )
    )
    neurosed_pass_path = _absolute(
        release["taste_gcf_neurosed_pass_path"],
        field="Taste NeuroSED generic PASS",
    )
    checkpoint_path = _absolute(
        release["taste_gcf_neurosed_checkpoint_path"],
        field="Taste NeuroSED checkpoint",
    )
    if (
        neurosed_pass_path != final_root / "PASS"
        or checkpoint_path != final_root / "artifacts/best.pt"
    ):
        raise TasteGCFSmokeError("Taste NeuroSED paths escape one managed final root")
    neurosed_pass = stack.enter_context(
        hold_readonly_file(
            neurosed_pass_path,
            expected_sha256=release["taste_gcf_neurosed_pass_sha256"],
        )
    )
    neurosed_gate = stack.enter_context(
        hold_readonly_file(
            final_root / "gate.json",
            expected_sha256=release["taste_gcf_neurosed_gate_sha256"],
        )
    )
    neurosed_verification = stack.enter_context(
        hold_readonly_file(
            final_root / "verification.json",
            expected_sha256=release["taste_gcf_neurosed_verification_sha256"],
        )
    )
    neurosed_checkpoint = stack.enter_context(
        hold_readonly_file(
            checkpoint_path,
            expected_sha256=release["taste_gcf_neurosed_checkpoint_sha256"],
        )
    )
    neurosed_feature_schema = stack.enter_context(
        hold_readonly_file(
            final_root / "artifacts/feature_schema.json",
            expected_sha256=release["taste_gcf_neurosed_feature_schema_sha256"],
        )
    )
    neurosed_sha256s = stack.enter_context(
        hold_readonly_file(
            final_root / "artifacts/sha256sums.txt",
            expected_sha256=release["taste_gcf_neurosed_sha256s_sha256"],
        )
    )
    if neurosed_pass.read_bytes() != b"[MANAGED_EXECUTION_V2_PASS]\n":
        raise TasteGCFSmokeError("Taste NeuroSED generic PASS marker changed")
    gate = _json(neurosed_gate.read_bytes(), field="Taste NeuroSED gate")
    verification = _json(
        neurosed_verification.read_bytes(), field="Taste NeuroSED verification"
    )
    feature_schema_payload = _json(
        neurosed_feature_schema.read_bytes(), field="Taste NeuroSED feature schema"
    )
    domain = verification.get("verification")
    consumer = domain.get("t7_consumer") if type(domain) is dict else None
    published_inventory = verification.get("published_inventory")
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("independent_verifier") is not True
        or gate.get("science_adopted") is not True
        or gate.get("downstream_released") is not True
        or gate.get("verification_sha256") != neurosed_verification.sha256
        or verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != "PASS"
        or verification.get("independent_verifier") is not True
        or verification.get("attempt_id") != neurosed_final.sealed.attempt_id
        or verification.get("generation_token")
        != neurosed_final.sealed.generation_token
        or verification.get("sealed_sha256") != neurosed_final.sealed.seal_sha256
        or verification.get("source_inventory_sha256")
        != neurosed_final.sealed.inventory_sha256
        or verification.get("published_inventory_sha256")
        != gate.get("published_inventory_sha256")
        or hashlib.sha256(canonical_json_bytes(published_inventory)).hexdigest()
        != verification.get("published_inventory_sha256")
        or published_inventory != neurosed_final.inventory.payload()
        or type(consumer) is not dict
        or consumer.get("schema_version")
        != "tastemolnet_gcf_neurosed_t7_consumer_v1"
        or consumer.get("dataset") != "tastemolnet"
        or consumer.get("role") != "GCF_AUXILIARY_DISTANCE_MODEL"
        or consumer.get("classifier") is not False
        or consumer.get("source_label_independent") is not True
        or consumer.get("train_only_fit") is not True
        or consumer.get("validation_only_selection") is not True
        or consumer.get("calibration_loaded") is not False
        or consumer.get("test_loaded") is not False
        or consumer.get("health_gate_status") != "PASS"
        or consumer.get("checkpoint_relative_path") != "artifacts/best.pt"
        or consumer.get("checkpoint_sha256") != neurosed_checkpoint.sha256
        or consumer.get("feature_schema_relative_path")
        != "artifacts/feature_schema.json"
        or consumer.get("feature_schema_sha256") != neurosed_feature_schema.sha256
        or consumer.get("feature_atomic_numbers")
        != feature_schema_payload.get("feature_atomic_numbers")
        or consumer.get("feature_input_dim")
        != feature_schema_payload.get("input_dim")
        or feature_schema_payload.get("feature_atomic_numbers")
        != sorted(set(feature_schema_payload.get("feature_atomic_numbers", [])))
        or consumer.get("sha256s_relative_path") != "artifacts/sha256sums.txt"
        or consumer.get("sha256s_sha256") != neurosed_sha256s.sha256
    ):
        raise TasteGCFSmokeError("Taste NeuroSED managed final evidence changed")
    for field in (
        "checkpoint_sha256",
        "feature_schema_sha256",
        "sha256s_sha256",
        "neurosed_train_graph_ids_hash",
        "neurosed_validation_graph_ids_hash",
    ):
        _sha256(consumer.get(field), field=f"Taste NeuroSED {field}")
    checksum_rows: dict[str, str] = {}
    for line in neurosed_sha256s.read_bytes().decode("utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in checksum_rows:
            raise TasteGCFSmokeError("Taste NeuroSED sha256sums is malformed")
        checksum_rows[relative] = _sha256(digest, field=f"NeuroSED {relative}")
    if (
        checksum_rows.get("best.pt") != neurosed_checkpoint.sha256
        or checksum_rows.get("feature_schema.json") != neurosed_feature_schema.sha256
    ):
        raise TasteGCFSmokeError("Taste NeuroSED artifact hash closure changed")
    evidence = {
        "schema_version": "tastemolnet_gcf_neurosed_managed_final_v1",
        "status": "PASS",
        "marker": "MANAGED_EXECUTION_V2_PASS",
        "final_root": str(final_root),
        "attempt_id": neurosed_final.sealed.attempt_id,
        "generation_token": neurosed_final.sealed.generation_token,
        "pass_path": str(neurosed_pass_path),
        "pass_sha256": neurosed_pass.sha256,
        "gate_path": str(final_root / "gate.json"),
        "gate_sha256": neurosed_gate.sha256,
        "verification_path": str(final_root / "verification.json"),
        "verification_sha256": neurosed_verification.sha256,
        "source_inventory_sha256": neurosed_final.sealed.inventory_sha256,
        "published_inventory_sha256": verification[
            "published_inventory_sha256"
        ],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": neurosed_checkpoint.sha256,
        "feature_schema_path": str(final_root / "artifacts/feature_schema.json"),
        "feature_schema_sha256": neurosed_feature_schema.sha256,
        "sha256s_path": str(final_root / "artifacts/sha256sums.txt"),
        "sha256s_sha256": neurosed_sha256s.sha256,
        "t7_consumer": dict(consumer),
    }
    distance_threshold = float(authority["neurosed"]["distance_threshold"])
    managed_pass.revalidate()
    neurosed_final.revalidate()
    neurosed_pass.revalidate()
    neurosed_gate.revalidate()
    neurosed_verification.revalidate()
    neurosed_checkpoint.revalidate()
    neurosed_feature_schema.revalidate()
    neurosed_sha256s.revalidate()
    return (
        managed_pass,
        neurosed_final,
        neurosed_pass,
        neurosed_gate,
        neurosed_verification,
        neurosed_checkpoint,
        neurosed_feature_schema,
        neurosed_sha256s,
        evidence,
        distance_threshold,
    )


@dataclass(slots=True)
class HeldTasteGCFInputs:
    """All T7 authorities retained through worker close and managed seal."""

    stack: ExitStack
    release: Mapping[str, Any]
    execution: Mapping[str, str]
    release_file: Any
    authority_file: Any
    authority: Mapping[str, Any]
    controller_file: Any
    controller: Mapping[str, Any]
    gpu_file: Any
    gpu: Mapping[str, Any]
    t2: Mapping[str, Any]
    t2_adoption: Any
    t3: Any
    t4: Any
    checkpoint: Any
    checkpoint_payloads: Mapping[str, bytes]
    stage_evidence: Mapping[str, Any]
    train_file: Any
    train_bytes: bytes
    train_contract: Mapping[str, Any]
    managed_v2_pass: Any
    neurosed_final: Any
    neurosed_pass: Any
    neurosed_gate: Any
    neurosed_verification: Any
    neurosed_checkpoint: Any
    neurosed_feature_schema: Any
    neurosed_sha256s: Any
    neurosed_evidence: Mapping[str, Any]
    neurosed_distance_threshold: float
    output_root: Path

    @property
    def managed_stage_root(self) -> Path:
        return _absolute(
            self.release["managed_stage_root"], field="managed stage root"
        )

    @property
    def managed_input_hashes(self) -> Mapping[str, str]:
        return {
            "managed_execution_v2_pass": self.release[
                "managed_execution_v2_pass_sha256"
            ],
            "taste_gcf_neurosed_pass": self.neurosed_evidence[
                "pass_sha256"
            ],
            "taste_gcf_neurosed_gate": self.neurosed_evidence[
                "gate_sha256"
            ],
            "taste_gcf_neurosed_verification": self.neurosed_evidence[
                "verification_sha256"
            ],
            "taste_gcf_neurosed_checkpoint": self.neurosed_evidence[
                "checkpoint_sha256"
            ],
            "taste_gcf_neurosed_feature_schema": self.neurosed_evidence[
                "feature_schema_sha256"
            ],
            "taste_gcf_neurosed_sha256s": self.neurosed_evidence[
                "sha256s_sha256"
            ],
            "taste_gine_t2_gate": self.t2["gate_sha256"],
            "taste_gine_t3_gate": self.stage_evidence["t3"]["gate_sha256"],
            "taste_oracle_t4_gate": self.stage_evidence["t4"]["gate_sha256"],
            "taste_train_csv": self.train_contract["sha256"],
        }

    def revalidate_neurosed(self) -> None:
        self.neurosed_final.revalidate()
        self.neurosed_pass.revalidate()
        self.neurosed_gate.revalidate()
        self.neurosed_verification.revalidate()
        self.neurosed_checkpoint.revalidate()
        self.neurosed_feature_schema.revalidate()
        self.neurosed_sha256s.revalidate()

    def revalidate(self) -> None:
        self.release_file.revalidate()
        self.authority_file.revalidate()
        self.controller_file.revalidate()
        self.gpu_file.revalidate()
        current_t2 = self.t2_adoption.revalidate()
        current_t3 = self.t3.revalidate()
        current_t4 = self.t4.revalidate()
        self.checkpoint.revalidate()
        self.train_file.revalidate()
        self.managed_v2_pass.revalidate()
        self.revalidate_neurosed()
        current_execution = verify_execution_checkout(self.release)
        if current_execution != dict(self.execution):
            raise TasteGCFSmokeError("Taste T7 execution checkout identity changed")
        _verify_critical_blobs(self.authority)
        _verify_controller_process(self.controller)
        _gpu_runtime_evidence(self.authority["gpu_uuid"])
        if current_t2 != dict(self.t2):
            raise TasteGCFSmokeError("Taste T7 T2 adoption changed")
        if (
            current_t3 != dict(self.stage_evidence["t3"])
            or current_t4 != dict(self.stage_evidence["t4"])
        ):
            raise TasteGCFSmokeError("Taste T7 T3/T4 authority changed")

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "HeldTasteGCFInputs":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_tastemolnet_t7_inputs(
    *,
    output_dir: str | Path,
    config_path: str | Path,
) -> HeldTasteGCFInputs:
    """Resolve and retain the exact T2/T3/T4/controller/GPU science inputs."""

    release = assert_execution_released()
    requested_config = Path(config_path).expanduser()
    if requested_config != REPO_ROOT / "configs/hpc.yaml":
        raise TasteGCFSmokeError(
            "Taste T7 requires the exact integrated HPC config"
        )
    stack = ExitStack()
    try:
        from src.utils.retained_readonly_file import hold_readonly_file

        release_file = stack.enter_context(
            hold_readonly_file(
                RELEASE_CONFIG_PATH,
                expected_sha256=_sha256_file(RELEASE_CONFIG_PATH),
            )
        )
        execution = verify_execution_checkout(release)
        authority_file, authority = hold_external_authority(stack, release)
        controller_file, controller, gpu_file, gpu = (
            hold_controller_gpu_receipts(stack, authority)
        )
        t2_adoption, t2 = hold_t2_adoption(stack, authority, release)
        t3, t4, checkpoint, checkpoint_payloads, stage_evidence = (
            hold_stages_and_checkpoint(stack, authority, release, t2)
        )
        train_file, train_bytes, train_contract = hold_train_payload(
            stack, authority, checkpoint_payloads
        )
        (
            managed_v2_pass,
            neurosed_final,
            neurosed_pass,
            neurosed_gate,
            neurosed_verification,
            neurosed_checkpoint,
            neurosed_feature_schema,
            neurosed_sha256s,
            neurosed_evidence,
            neurosed_distance_threshold,
        ) = hold_managed_neurosed_predecessors(stack, release, authority)
        output = _absolute(str(output_dir), field="Taste T7 output root")
        expected_output = _absolute(
            authority["expected_output_root"], field="expected output root"
        )
        output_parent = _absolute(
            authority["output_parent"], field="output parent"
        )
        if output != expected_output or output.parent != output_parent:
            raise TasteGCFSmokeError(
                "Taste T7 output differs from typed controller authority"
            )
        input_paths = (
            RELEASE_CONFIG_PATH,
            _absolute(
                release["external_authority_path"],
                field="external authority path",
            ),
            _absolute(
                authority["controller_receipt_path"],
                field="controller receipt path",
            ),
            _absolute(
                authority["gpu_lease_receipt_path"],
                field="GPU lease receipt path",
            ),
            _absolute(
                authority["checkpoint"]["checkpoint_dir"],
                field="checkpoint path",
            ),
            _absolute(authority["train"]["path"], field="train path"),
            REPO_ROOT,
            _absolute(authority["official_root"], field="official root"),
            _absolute(authority["t3"]["root"], field="T3 path"),
            _absolute(authority["t4"]["root"], field="T4 path"),
            _absolute(authority["t2"]["adoption_root"], field="T2 path"),
            _absolute(
                release["managed_execution_v2_pass_path"],
                field="managed execution v2 PASS path",
            ),
            _absolute(
                release["taste_gcf_neurosed_final_root"],
                field="Taste NeuroSED managed final root",
            ),
            _absolute(
                release["taste_gcf_neurosed_pass_path"],
                field="Taste NeuroSED PASS path",
            ),
            _absolute(
                release["taste_gcf_neurosed_checkpoint_path"],
                field="Taste NeuroSED checkpoint path",
            ),
        )
        if any(
            output == source
            or output in source.parents
            or source in output.parents
            for source in input_paths
        ):
            raise TasteGCFSmokeError(
                "Taste T7 output overlaps an immutable input"
            )
        result = HeldTasteGCFInputs(
            stack=stack,
            release=release,
            execution=execution,
            release_file=release_file,
            authority_file=authority_file,
            authority=authority,
            controller_file=controller_file,
            controller=controller,
            gpu_file=gpu_file,
            gpu=gpu,
            t2=t2,
            t2_adoption=t2_adoption,
            t3=t3,
            t4=t4,
            checkpoint=checkpoint,
            checkpoint_payloads=checkpoint_payloads,
            stage_evidence=stage_evidence,
            train_file=train_file,
            train_bytes=train_bytes,
            train_contract=train_contract,
            managed_v2_pass=managed_v2_pass,
            neurosed_final=neurosed_final,
            neurosed_pass=neurosed_pass,
            neurosed_gate=neurosed_gate,
            neurosed_verification=neurosed_verification,
            neurosed_checkpoint=neurosed_checkpoint,
            neurosed_feature_schema=neurosed_feature_schema,
            neurosed_sha256s=neurosed_sha256s,
            neurosed_evidence=neurosed_evidence,
            neurosed_distance_threshold=neurosed_distance_threshold,
            output_root=output,
        )
        result.revalidate()
        return result
    except Exception:
        stack.close()
        raise


__all__ = [
    "HeldTasteGCFInputs",
    "assert_execution_released",
    "hold_stages_and_checkpoint",
    "hold_managed_neurosed_predecessors",
    "hold_t2_adoption",
    "hold_tastemolnet_t7_inputs",
]
