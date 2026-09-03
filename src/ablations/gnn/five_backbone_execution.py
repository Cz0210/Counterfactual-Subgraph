"""Strict two-lane executor for the BACE five-backbone ablation.

The existing five-backbone status command is deliberately read-only.  This
module consumes one byte-pinned ``science_launch_allowed=true`` decision and a
separate, self-hashed execution spec containing the real science commands.
It adopts the frozen GINE reference and executes only the other four
backbones.  Children are always launched with ``shell=False`` and may yield to
a newly-ready main-table GPU task only through an explicit checkpoint request
and a verified checkpoint receipt.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from uuid import UUID


EXECUTION_SPEC_SCHEMA = "gnn_five_backbone_execution_spec_v1"
EXECUTION_STATE_SCHEMA = "gnn_five_backbone_execution_state_v1"
TASK_TERMINAL_SCHEMA = "gnn_five_backbone_task_terminal_v1"
TASK_CHECKPOINT_SCHEMA = "gnn_five_backbone_task_checkpoint_v1"
STATUS_SCHEMA = "gnn_five_backbone_launch_decision_v1"

GINE_REFERENCE = "gine"
EXECUTED_BACKBONES = ("gin", "gatedgcn_plus", "gcn", "gatv2")
EXECUTION_LANES = {
    "lane0": ("gin", "gatedgcn_plus"),
    "lane1": ("gcn", "gatv2"),
}
STATUS_SCHEDULE = {
    "lane0": ("gine", "gin", "gatedgcn_plus"),
    "lane1": ("gcn", "gatv2"),
}
ALLOWED_SEED_PREFIX = (7, 17, 27)
REQUIRED_GINE_ROLES = {"classifier_checkpoint", "temperature", "run_manifest"}
ALLOWED_TEMPLATE_FIELDS = {
    "backbone",
    "checkpoint_path",
    "checkpoint_request",
    "gpu_id",
    "output_root",
    "seed",
    "task_root",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEMPLATE_FIELD = re.compile(r"\{([a-z_]+)\}")


class FiveBackboneExecutionError(RuntimeError):
    """Execution evidence or a child science result failed closed."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise FiveBackboneExecutionError(f"{field} must be a lowercase SHA256")
    return normalized


def _physical_file(path_like: object, sha256: object, *, role: str) -> Path:
    path = Path(str(path_like or "")).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise FiveBackboneExecutionError(f"{role} must be an absolute physical file")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FiveBackboneExecutionError(f"{role} is absent") from exc
    if not resolved.is_file():
        raise FiveBackboneExecutionError(f"{role} is not a file")
    expected = _require_sha256(sha256, field=f"{role}.sha256")
    if sha256_file(resolved) != expected:
        raise FiveBackboneExecutionError(f"{role} SHA256 changed")
    return resolved


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiveBackboneExecutionError(f"{role} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise FiveBackboneExecutionError(f"{role} must be one JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _uuid4(value: object, *, field: str) -> str:
    raw = str(value or "").strip().lower()
    try:
        parsed = UUID(raw)
    except ValueError as exc:
        raise FiveBackboneExecutionError(f"{field} must be a UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != raw:
        raise FiveBackboneExecutionError(f"{field} must be a canonical UUIDv4")
    return raw


def _safe_relative(value: object, *, field: str) -> str:
    path = Path(str(value or ""))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise FiveBackboneExecutionError(f"{field} must be a safe relative path")
    return str(path)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    role: str
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, index: int) -> "ArtifactIdentity":
        if not isinstance(payload, Mapping) or set(payload) != {"role", "path", "sha256"}:
            raise FiveBackboneExecutionError(
                f"gine_reference_artifacts[{index}] must contain role/path/sha256"
            )
        role = str(payload.get("role") or "")
        if not role:
            raise FiveBackboneExecutionError("GINE reference artifact role is empty")
        path = _physical_file(
            payload.get("path"), payload.get("sha256"), role=f"GINE reference {role}"
        )
        return cls(role=role, path=str(path), sha256=str(payload["sha256"]))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScienceCommandTemplate:
    backbone: str
    lane: str
    argv_template: tuple[str, ...]
    resume_argv_template: tuple[str, ...]
    terminal_relpath: str
    checkpoint_relpath: str
    pause_exit_code: int

    @classmethod
    def from_mapping(
        cls,
        backbone: str,
        payload: Mapping[str, Any],
        *,
        project_root: Path,
    ) -> "ScienceCommandTemplate":
        required = {
            "lane",
            "argv_template",
            "resume_argv_template",
            "terminal_relpath",
            "checkpoint_relpath",
            "pause_exit_code",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise FiveBackboneExecutionError(
                f"science_commands.{backbone} is incomplete; required={sorted(required)}"
            )
        lane = str(payload.get("lane") or "")
        if backbone not in EXECUTION_LANES.get(lane, ()):
            raise FiveBackboneExecutionError(f"{backbone} is assigned to the wrong lane")
        argv = cls._argv(
            payload.get("argv_template"),
            field=f"science_commands.{backbone}.argv_template",
            project_root=project_root,
            resume=False,
        )
        resume_argv = cls._argv(
            payload.get("resume_argv_template"),
            field=f"science_commands.{backbone}.resume_argv_template",
            project_root=project_root,
            resume=True,
        )
        pause_exit_code = payload.get("pause_exit_code")
        if not isinstance(pause_exit_code, int) or not 1 <= pause_exit_code <= 255:
            raise FiveBackboneExecutionError(f"{backbone}.pause_exit_code must be 1..255")
        return cls(
            backbone=backbone,
            lane=lane,
            argv_template=argv,
            resume_argv_template=resume_argv,
            terminal_relpath=_safe_relative(
                payload.get("terminal_relpath"), field=f"{backbone}.terminal_relpath"
            ),
            checkpoint_relpath=_safe_relative(
                payload.get("checkpoint_relpath"), field=f"{backbone}.checkpoint_relpath"
            ),
            pause_exit_code=pause_exit_code,
        )

    @staticmethod
    def _argv(
        raw: object,
        *,
        field: str,
        project_root: Path,
        resume: bool,
    ) -> tuple[str, ...]:
        if not isinstance(raw, list) or not raw or any(
            not isinstance(item, str) or not item for item in raw
        ):
            raise FiveBackboneExecutionError(f"{field} must be a non-empty string list")
        values = tuple(raw)
        fields = {name for token in values for name in _TEMPLATE_FIELD.findall(token)}
        unknown = fields - ALLOWED_TEMPLATE_FIELDS
        if unknown:
            raise FiveBackboneExecutionError(f"{field} has unknown placeholders: {sorted(unknown)}")
        required_fields = {"backbone", "checkpoint_request", "seed", "task_root"}
        if resume:
            required_fields.add("checkpoint_path")
        if not required_fields.issubset(fields):
            raise FiveBackboneExecutionError(
                f"{field} lacks required placeholders: {sorted(required_fields - fields)}"
            )
        if ("--resume" in values) is not resume:
            raise FiveBackboneExecutionError(
                f"{field} must {'include' if resume else 'exclude'} --resume"
            )
        if "--config" not in values or "inference.fallback_to_heuristic=false" not in values:
            raise FiveBackboneExecutionError(
                f"{field} must bind --config and disable heuristic fallback"
            )
        executable = Path(values[0]).expanduser()
        if not executable.is_absolute() or not executable.resolve(strict=True).is_file():
            raise FiveBackboneExecutionError(f"{field}[0] must be a real absolute executable")
        if not os.access(executable.resolve(), os.X_OK):
            raise FiveBackboneExecutionError(f"{field}[0] is not executable")
        executable_name = executable.resolve().name.lower()
        if "python" in executable_name:
            if len(values) < 2 or values[1] in {"-c", "-m"}:
                raise FiveBackboneExecutionError(
                    f"{field} Python invocation must name a physical science script"
                )
            science_script = Path(values[1]).expanduser()
            if not science_script.is_absolute() or science_script.is_symlink():
                raise FiveBackboneExecutionError(f"{field}[1] must be a physical absolute script")
            resolved_script = science_script.resolve(strict=True)
            if not resolved_script.is_file() or resolved_script.suffix != ".py":
                raise FiveBackboneExecutionError(f"{field}[1] is not a Python science script")
            try:
                resolved_script.relative_to(project_root)
            except ValueError as exc:
                raise FiveBackboneExecutionError(
                    f"{field}[1] must live in the immutable project checkout"
                ) from exc
        return values

    def render(
        self,
        *,
        root: Path,
        task_root: Path,
        seed: int,
        gpu_id: str,
        resume: bool,
    ) -> list[str]:
        checkpoint = task_root / self.checkpoint_relpath
        request = task_root / "checkpoint_request.json"
        values = {
            "backbone": self.backbone,
            "checkpoint_path": str(checkpoint),
            "checkpoint_request": str(request),
            "gpu_id": gpu_id,
            "output_root": str(root),
            "seed": str(seed),
            "task_root": str(task_root),
        }
        template = self.resume_argv_template if resume else self.argv_template
        rendered = [token.format_map(values) for token in template]
        if any(_TEMPLATE_FIELD.search(token) for token in rendered):
            raise FiveBackboneExecutionError("science command retained an unresolved placeholder")
        return rendered


@dataclass(frozen=True, slots=True)
class FiveBackboneExecutionSpec:
    run_id: str
    execution_commit: str
    project_root: str
    output_root: str
    max_concurrent_gpus: int
    seeds: tuple[int, ...]
    lane_gpu_ids: Mapping[str, str]
    main_matrix_write_allowed: bool
    gine_reference_artifacts: tuple[ArtifactIdentity, ...]
    science_commands: Mapping[str, ScienceCommandTemplate]
    run_spec_sha256: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_project_root: Path,
    ) -> "FiveBackboneExecutionSpec":
        if payload.get("schema_version") != EXECUTION_SPEC_SCHEMA:
            raise FiveBackboneExecutionError("five-backbone execution spec schema changed")
        run_id = _uuid4(payload.get("run_id"), field="run_id")
        project_root = Path(str(payload.get("project_root") or "")).expanduser()
        if project_root.is_symlink() or project_root.resolve(strict=True) != expected_project_root:
            raise FiveBackboneExecutionError("run spec project_root differs from checkout")
        output = Path(str(payload.get("output_root") or "")).expanduser()
        if not output.is_absolute() or output.is_symlink() or run_id not in output.name:
            raise FiveBackboneExecutionError(
                "output_root must be a fresh absolute path containing the run UUID"
            )
        if output == expected_project_root or expected_project_root in output.parents:
            raise FiveBackboneExecutionError("output_root may not be inside the project checkout")
        commit = str(payload.get("execution_commit") or "").strip().lower()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise FiveBackboneExecutionError("execution_commit must be a full Git SHA")
        if payload.get("max_concurrent_gpus") != 2:
            raise FiveBackboneExecutionError(
                "five-backbone execution requires exactly two GPU lanes"
            )
        raw_seeds = payload.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise FiveBackboneExecutionError("seeds must start with seed 7")
        seeds = tuple(raw_seeds)
        if seeds != ALLOWED_SEED_PREFIX[: len(seeds)]:
            raise FiveBackboneExecutionError("seeds must be the prefix [7], [7,17], or [7,17,27]")
        gpu_ids = payload.get("lane_gpu_ids")
        if not isinstance(gpu_ids, Mapping) or set(gpu_ids) != set(EXECUTION_LANES):
            raise FiveBackboneExecutionError("lane_gpu_ids must bind lane0 and lane1")
        normalized_gpu_ids = {lane: str(gpu_ids[lane]).strip() for lane in EXECUTION_LANES}
        if any(not value or "," in value for value in normalized_gpu_ids.values()) or len(
            set(normalized_gpu_ids.values())
        ) != 2:
            raise FiveBackboneExecutionError("lane GPU identifiers must be distinct single devices")
        if payload.get("main_matrix_write_allowed") is not False:
            raise FiveBackboneExecutionError("main matrix writes must remain disabled")
        raw_artifacts = payload.get("gine_reference_artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise FiveBackboneExecutionError("GINE reference adoption evidence is required")
        artifacts = tuple(
            ArtifactIdentity.from_mapping(item, index=index)
            for index, item in enumerate(raw_artifacts)
        )
        roles = [item.role for item in artifacts]
        if len(set(roles)) != len(roles) or not REQUIRED_GINE_ROLES.issubset(roles):
            raise FiveBackboneExecutionError(
                "GINE adoption requires unique classifier_checkpoint/temperature/run_manifest roles"
            )
        commands = payload.get("science_commands")
        if not isinstance(commands, Mapping) or set(commands) != set(EXECUTED_BACKBONES):
            raise FiveBackboneExecutionError(
                "complete real science commands are required for all four non-GINE backbones"
            )
        parsed_commands = {
            backbone: ScienceCommandTemplate.from_mapping(
                backbone, commands[backbone], project_root=expected_project_root
            )
            for backbone in EXECUTED_BACKBONES
        }
        claimed = _require_sha256(payload.get("run_spec_sha256"), field="run_spec_sha256")
        body = dict(payload)
        body.pop("run_spec_sha256", None)
        if canonical_json_sha256(body) != claimed:
            raise FiveBackboneExecutionError("run spec self-hash changed")
        return cls(
            run_id=run_id,
            execution_commit=commit,
            project_root=str(expected_project_root),
            output_root=str(output),
            max_concurrent_gpus=2,
            seeds=seeds,
            lane_gpu_ids=normalized_gpu_ids,
            main_matrix_write_allowed=False,
            gine_reference_artifacts=artifacts,
            science_commands=parsed_commands,
            run_spec_sha256=claimed,
        )


def load_execution_spec(
    path_like: str | Path,
    expected_sha256: str,
    *,
    project_root: Path,
) -> FiveBackboneExecutionSpec:
    path = _physical_file(path_like, expected_sha256, role="five-backbone run spec")
    return FiveBackboneExecutionSpec.from_mapping(
        _read_json(path, role="five-backbone run spec"),
        expected_project_root=project_root.resolve(strict=True),
    )


@dataclass(frozen=True, slots=True)
class LaunchEvidence:
    status_path: str
    status_sha256: str
    authority_root: str
    matrix_status_sha256: str
    combined_audit_sha256: str


def load_launch_evidence(path_like: str | Path, expected_sha256: str) -> LaunchEvidence:
    path = _physical_file(path_like, expected_sha256, role="five-backbone status")
    payload = _read_json(path, role="five-backbone status")
    if (
        payload.get("schema_version") != STATUS_SCHEMA
        or payload.get("science_launch_allowed") is not True
        or payload.get("state") != "AUTHORIZED_TO_LAUNCH_FIVE_BACKBONE_PHASE1"
        or payload.get("blockers") != []
        or payload.get("max_concurrent_gpus") != 2
        or payload.get("phase1_seed") != 7
        or payload.get("main_gate_pass") is not True
        or payload.get("user_authorized_after_16") is not True
        or payload.get("run_requested") is not True
        or payload.get("no_main_task_waiting_for_gpu") is not True
        or payload.get("proposal_fixed_manifest_pass") is not True
        or payload.get("gatedgcn_plus_runtime_pass") is not True
        or payload.get("graph_mamba_run_enabled") is not False
        or payload.get("main_matrix_modified") is not False
    ):
        raise FiveBackboneExecutionError("status does not authorize five-backbone science")
    if tuple(payload.get("backbones", ())) != (
        "gine",
        "gin",
        "gcn",
        "gatv2",
        "gatedgcn_plus",
    ):
        raise FiveBackboneExecutionError("status backbone set changed")
    schedule = payload.get("schedule")
    if not isinstance(schedule, Mapping) or {
        lane: tuple(schedule.get(lane, ())) for lane in STATUS_SCHEDULE
    } != STATUS_SCHEDULE:
        raise FiveBackboneExecutionError("status two-lane schedule changed")
    main_gate = payload.get("main_gate")
    if not isinstance(main_gate, Mapping) or (
        main_gate.get("science_launch_allowed") is not True
        or main_gate.get("main_matrix_complete_cells") != 16
        or main_gate.get("main_matrix_total_cells") != 16
        or main_gate.get("authority_verified") is not True
        or main_gate.get("final_audit_pass") is not True
        or main_gate.get("figure3_pass") is not True
        or main_gate.get("figure4_pass") is not True
        or main_gate.get("table2_pass") is not True
        or main_gate.get("explicit_run_authorization") is not True
    ):
        raise FiveBackboneExecutionError("status does not bind the complete main-table gate")
    root = Path(str(main_gate.get("authority_root") or "")).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise FiveBackboneExecutionError("status authority root is not a physical directory")
    matrix_sha = _require_sha256(
        main_gate.get("matrix_status_sha256"), field="matrix_status_sha256"
    )
    combined_sha = _require_sha256(
        main_gate.get("combined_audit_sha256"), field="combined_audit_sha256"
    )
    evidence = LaunchEvidence(
        status_path=str(path),
        status_sha256=str(expected_sha256),
        authority_root=str(root.resolve()),
        matrix_status_sha256=matrix_sha,
        combined_audit_sha256=combined_sha,
    )
    assert_main_matrix_unchanged(evidence)
    return evidence


def assert_main_matrix_unchanged(evidence: LaunchEvidence) -> None:
    root = Path(evidence.authority_root)
    matrix = root / "matrix_status.json"
    combined = root / "combined_audit.json"
    if (
        matrix.is_symlink()
        or combined.is_symlink()
        or not matrix.is_file()
        or not combined.is_file()
        or sha256_file(matrix) != evidence.matrix_status_sha256
        or sha256_file(combined) != evidence.combined_audit_sha256
    ):
        raise FiveBackboneExecutionError("main matrix authority changed during ablation")


def _main_waiting(path: Path) -> tuple[bool, str, str | None]:
    """Treat missing/malformed live queue evidence conservatively as waiting."""

    if path.is_symlink() or not path.is_file():
        return True, "MAIN_READY_GPU_EVIDENCE_MISSING", None
    try:
        payload = _read_json(path, role="main READY_GPU queue")
    except FiveBackboneExecutionError:
        return True, "MAIN_READY_GPU_EVIDENCE_INVALID", None
    digest = sha256_file(path)
    tasks = payload.get("ready_waiting_gpu", payload.get("ready_gpu_tasks"))
    if payload.get("status") not in {"PASS", "READY"} or not isinstance(tasks, list):
        return True, "MAIN_READY_GPU_EVIDENCE_INVALID", digest
    if tasks:
        return True, "MAIN_TASK_READY_WAITING_GPU", digest
    return False, "NO_MAIN_TASK_READY_WAITING_GPU", digest


class _StateStore:
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.lock = threading.Lock()
        self.write()

    @classmethod
    def load(cls, path: Path, *, spec: FiveBackboneExecutionSpec) -> "_StateStore":
        payload = _read_json(path, role="execution state")
        claimed = _require_sha256(payload.pop("state_sha256", None), field="state_sha256")
        if canonical_json_sha256(payload) != claimed:
            raise FiveBackboneExecutionError("execution state self-hash changed")
        payload["state_sha256"] = claimed
        if (
            payload.get("schema_version") != EXECUTION_STATE_SCHEMA
            or payload.get("run_id") != spec.run_id
            or payload.get("run_spec_sha256") != spec.run_spec_sha256
            or payload.get("main_matrix_modified") is not False
        ):
            raise FiveBackboneExecutionError("execution state belongs to another run")
        instance = cls.__new__(cls)
        instance.path = path
        instance.payload = payload
        instance.lock = threading.Lock()
        return instance

    def write(self) -> None:
        with self.lock:
            payload = dict(self.payload)
            payload.pop("state_sha256", None)
            payload["state_sha256"] = canonical_json_sha256(payload)
            self.payload = payload
            _atomic_json(self.path, payload)

    def task(self, task_id: str) -> Mapping[str, Any] | None:
        with self.lock:
            value = self.payload.get("tasks", {}).get(task_id)
            return dict(value) if isinstance(value, Mapping) else None

    def update_task(self, task_id: str, receipt: Mapping[str, Any]) -> None:
        with self.lock:
            tasks = dict(self.payload.get("tasks", {}))
            tasks[task_id] = dict(receipt)
            self.payload["tasks"] = tasks
        self.write()

    def set_state(self, state: str, **updates: Any) -> None:
        with self.lock:
            self.payload.update(updates)
            self.payload["state"] = state
        self.write()


def _self_hashed_json(path: Path, *, hash_field: str, role: str) -> dict[str, Any]:
    payload = _read_json(path, role=role)
    claimed = _require_sha256(payload.pop(hash_field, None), field=hash_field)
    if canonical_json_sha256(payload) != claimed:
        raise FiveBackboneExecutionError(f"{role} self-hash changed")
    payload[hash_field] = claimed
    return payload


def _verify_task_terminal(
    path: Path, *, backbone: str, seed: int, task_root: Path
) -> dict[str, Any]:
    payload = _self_hashed_json(path, hash_field="terminal_sha256", role="task terminal")
    expected = {
        "schema_version": TASK_TERMINAL_SCHEMA,
        "status": "PASS",
        "backbone": backbone,
        "seed": seed,
        "output_root": str(task_root),
        "checkpoint_resume_supported": True,
        "selector_frozen_before_test": True,
        "main_matrix_modified": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise FiveBackboneExecutionError(f"{backbone}/seed{seed} terminal contract changed")
    return payload


def _verify_task_checkpoint(
    path: Path, *, backbone: str, seed: int, task_root: Path
) -> dict[str, Any]:
    payload = _self_hashed_json(
        path, hash_field="checkpoint_sha256", role="task checkpoint"
    )
    expected = {
        "schema_version": TASK_CHECKPOINT_SCHEMA,
        "status": "PAUSED_AT_SAFE_CHECKPOINT",
        "backbone": backbone,
        "seed": seed,
        "output_root": str(task_root),
        "checkpoint_resume_supported": True,
        "main_matrix_modified": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise FiveBackboneExecutionError(f"{backbone}/seed{seed} checkpoint is not safe")
    return payload


def _checkpoint_request(
    path: Path,
    *,
    backbone: str,
    seed: int,
    reason: str,
    queue_sha256: str | None,
) -> None:
    if path.exists():
        return
    payload: dict[str, Any] = {
        "schema_version": "gnn_five_backbone_checkpoint_request_v1",
        "action": "CHECKPOINT_THEN_PAUSE",
        "reason": reason,
        "backbone": backbone,
        "seed": seed,
        "main_ready_gpu_queue_sha256": queue_sha256,
        "requested_at_unix": time.time(),
    }
    payload["request_sha256"] = canonical_json_sha256(payload)
    _atomic_json(path, payload)


def _child_environment(
    spec: FiveBackboneExecutionSpec,
    *,
    command: ScienceCommandTemplate,
    seed: int,
    task_root: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if "MATRIX_AUTHORITY" in key or key.startswith("MAIN_MATRIX"):
            environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": spec.lane_gpu_ids[command.lane],
            "GNN_ABLATION_BACKBONE": command.backbone,
            "GNN_ABLATION_CHECKPOINT_REQUEST": str(task_root / "checkpoint_request.json"),
            "GNN_ABLATION_LANE": command.lane,
            "GNN_ABLATION_MAIN_MATRIX_WRITE_ALLOWED": "0",
            "GNN_ABLATION_OUTPUT_ROOT": str(task_root),
            "GNN_ABLATION_RUN_ID": spec.run_id,
            "GNN_ABLATION_SEED": str(seed),
            "PYTHONPATH": spec.project_root,
        }
    )
    return environment


def _run_task(
    spec: FiveBackboneExecutionSpec,
    command: ScienceCommandTemplate,
    *,
    seed: int,
    state: _StateStore,
    live_queue: Path,
    launch_evidence: LaunchEvidence,
    pause_event: threading.Event,
    abort_event: threading.Event,
    poll_seconds: float,
) -> dict[str, Any]:
    task_id = f"{command.backbone}:seed{seed}"
    task_root = Path(spec.output_root) / command.backbone / f"seed{seed}"
    previous = state.task(task_id)
    if previous is not None and previous.get("state") == "PASS":
        terminal_path = task_root / command.terminal_relpath
        terminal = _verify_task_terminal(
            terminal_path, backbone=command.backbone, seed=seed, task_root=task_root
        )
        if (
            previous.get("task_root") != str(task_root)
            or previous.get("terminal_path") != str(terminal_path)
            or previous.get("terminal_sha256") != terminal["terminal_sha256"]
        ):
            raise FiveBackboneExecutionError(f"{task_id} PASS receipt changed")
        return dict(previous)
    resume = previous is not None and previous.get("state") in {
        "PAUSED_AT_SAFE_CHECKPOINT",
        "RUNNING",
    }
    checkpoint_path = task_root / command.checkpoint_relpath
    if resume:
        _verify_task_checkpoint(
            checkpoint_path, backbone=command.backbone, seed=seed, task_root=task_root
        )
    elif task_root.exists():
        raise FiveBackboneExecutionError(
            f"{task_id} has an uncommitted pre-existing task root"
        )
    waiting, reason, _ = _main_waiting(live_queue)
    if waiting or pause_event.is_set() or abort_event.is_set():
        pause_event.set()
        return {"state": "NOT_STARTED_MAIN_PRIORITY", "task_id": task_id, "reason": reason}
    if not resume:
        task_root.mkdir(parents=True)
    request_path = task_root / "checkpoint_request.json"
    if resume and request_path.exists():
        archived_request = task_root / (
            "checkpoint_request.consumed." + sha256_file(request_path) + ".json"
        )
        if archived_request.exists():
            raise FiveBackboneExecutionError(
                f"{task_id} already has the same consumed checkpoint request"
            )
        os.replace(request_path, archived_request)
    argv = command.render(
        root=Path(spec.output_root),
        task_root=task_root,
        seed=seed,
        gpu_id=spec.lane_gpu_ids[command.lane],
        resume=resume,
    )
    authority_text = launch_evidence.authority_root
    if any(authority_text in token for token in argv):
        raise FiveBackboneExecutionError("science command may not receive the main authority root")
    stdout_path = task_root / "stdout.log"
    stderr_path = task_root / "stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=spec.project_root,
            env=_child_environment(spec, command=command, seed=seed, task_root=task_root),
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )
        state.update_task(
            task_id,
            {
                "state": "RUNNING",
                "task_id": task_id,
                "backbone": command.backbone,
                "seed": seed,
                "lane": command.lane,
                "gpu_id": spec.lane_gpu_ids[command.lane],
                "pid": process.pid,
                "resumed": resume,
                "task_root": str(task_root),
            },
        )
        pause_requested = False
        pause_reason = None
        while process.poll() is None:
            waiting, reason, queue_sha = _main_waiting(live_queue)
            if waiting:
                pause_event.set()
            if pause_event.is_set() or abort_event.is_set():
                pause_reason = (
                    "PEER_LANE_FAILED" if abort_event.is_set() else reason
                )
                _checkpoint_request(
                    request_path,
                    backbone=command.backbone,
                    seed=seed,
                    reason=pause_reason,
                    queue_sha256=queue_sha,
                )
                pause_requested = True
            time.sleep(poll_seconds)
        returncode = int(process.returncode)
    assert_main_matrix_unchanged(launch_evidence)
    terminal_path = task_root / command.terminal_relpath
    if returncode == 0:
        terminal = _verify_task_terminal(
            terminal_path, backbone=command.backbone, seed=seed, task_root=task_root
        )
        receipt = {
            "state": "PASS",
            "task_id": task_id,
            "backbone": command.backbone,
            "seed": seed,
            "lane": command.lane,
            "gpu_id": spec.lane_gpu_ids[command.lane],
            "task_root": str(task_root),
            "terminal_path": str(terminal_path),
            "terminal_sha256": terminal["terminal_sha256"],
            "returncode": returncode,
            "resumed": resume,
        }
        state.update_task(task_id, receipt)
        return receipt
    if pause_requested and returncode == command.pause_exit_code:
        checkpoint = _verify_task_checkpoint(
            checkpoint_path, backbone=command.backbone, seed=seed, task_root=task_root
        )
        receipt = {
            "state": "PAUSED_AT_SAFE_CHECKPOINT",
            "task_id": task_id,
            "backbone": command.backbone,
            "seed": seed,
            "lane": command.lane,
            "gpu_id": spec.lane_gpu_ids[command.lane],
            "task_root": str(task_root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "returncode": returncode,
            "pause_reason": pause_reason,
            "resumed": resume,
        }
        state.update_task(task_id, receipt)
        return receipt
    raise FiveBackboneExecutionError(
        f"{task_id} exited {returncode} without a verified PASS terminal or safe checkpoint"
    )


def _run_lane(
    lane: str,
    spec: FiveBackboneExecutionSpec,
    *,
    seed: int,
    state: _StateStore,
    live_queue: Path,
    launch_evidence: LaunchEvidence,
    pause_event: threading.Event,
    abort_event: threading.Event,
    poll_seconds: float,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    try:
        for backbone in EXECUTION_LANES[lane]:
            if pause_event.is_set() or abort_event.is_set():
                break
            receipt = _run_task(
                spec,
                spec.science_commands[backbone],
                seed=seed,
                state=state,
                live_queue=live_queue,
                launch_evidence=launch_evidence,
                pause_event=pause_event,
                abort_event=abort_event,
                poll_seconds=poll_seconds,
            )
            receipts.append(receipt)
            if receipt["state"] != "PASS":
                pause_event.set()
                break
        return receipts
    except BaseException:
        abort_event.set()
        pause_event.set()
        raise


def _new_state(spec: FiveBackboneExecutionSpec, launch: LaunchEvidence) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_STATE_SCHEMA,
        "run_id": spec.run_id,
        "run_spec_sha256": spec.run_spec_sha256,
        "launch_status_sha256": launch.status_sha256,
        "state": "READY",
        "seeds": list(spec.seeds),
        "completed_seeds": [],
        "tasks": {},
        "gine_reference_adopted": False,
        "max_concurrent_gpus": 2,
        "main_matrix_modified": False,
    }


def _adopt_gine(
    spec: FiveBackboneExecutionSpec,
    *,
    state: _StateStore,
    launch: LaunchEvidence,
) -> None:
    root = Path(spec.output_root)
    receipt: dict[str, Any] = {
        "schema_version": "gnn_gine_reference_adoption_v1",
        "status": "PASS_ADOPTED",
        "backbone": GINE_REFERENCE,
        "science_retrained": False,
        "main_matrix_modified": False,
        "artifacts": [item.to_dict() for item in spec.gine_reference_artifacts],
        "authority_root_read_only": launch.authority_root,
    }
    receipt["adoption_sha256"] = canonical_json_sha256(receipt)
    _atomic_json(root / "gine_reference_adoption.json", receipt)
    state.set_state("READY", gine_reference_adopted=True)


def _verify_gine_adoption(spec: FiveBackboneExecutionSpec, launch: LaunchEvidence) -> None:
    path = Path(spec.output_root) / "gine_reference_adoption.json"
    receipt = _self_hashed_json(
        path, hash_field="adoption_sha256", role="GINE adoption receipt"
    )
    expected = {
        "schema_version": "gnn_gine_reference_adoption_v1",
        "status": "PASS_ADOPTED",
        "backbone": GINE_REFERENCE,
        "science_retrained": False,
        "main_matrix_modified": False,
        "artifacts": [item.to_dict() for item in spec.gine_reference_artifacts],
        "authority_root_read_only": launch.authority_root,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise FiveBackboneExecutionError("GINE adoption receipt changed")


def run_five_backbone_execution(
    spec: FiveBackboneExecutionSpec,
    launch: LaunchEvidence,
    *,
    main_ready_gpu_tasks: Path,
    resume: bool,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    """Run at most two GPU children, respecting seed and main-priority barriers."""

    if poll_seconds <= 0:
        raise FiveBackboneExecutionError("poll_seconds must be positive")
    root = Path(spec.output_root)
    authority = Path(launch.authority_root)
    if root == authority or authority in root.parents or root in authority.parents:
        raise FiveBackboneExecutionError("ablation output and main authority must be disjoint")
    state_path = root / "execution_state.json"
    if resume:
        if not root.is_dir() or not state_path.is_file():
            raise FiveBackboneExecutionError("resume requires a committed execution state")
        state = _StateStore.load(state_path, spec=spec)
        if state.payload.get("gine_reference_adopted") is not True:
            raise FiveBackboneExecutionError("resume state does not adopt the GINE reference")
        _verify_gine_adoption(spec, launch)
    else:
        if root.exists():
            raise FiveBackboneExecutionError("fresh execution refuses an existing output root")
        waiting, reason, _ = _main_waiting(main_ready_gpu_tasks)
        if waiting:
            raise FiveBackboneExecutionError(f"main priority blocks fresh launch: {reason}")
        root.mkdir(parents=True)
        state = _StateStore(state_path, _new_state(spec, launch))
        _atomic_json(
            root / "input_manifest.json",
            {
                "schema_version": "gnn_five_backbone_execution_input_v1",
                "run_id": spec.run_id,
                "run_spec_sha256": spec.run_spec_sha256,
                "launch_status_path": launch.status_path,
                "launch_status_sha256": launch.status_sha256,
                "authority_root": launch.authority_root,
                "matrix_status_sha256": launch.matrix_status_sha256,
                "combined_audit_sha256": launch.combined_audit_sha256,
                "main_matrix_write_allowed": False,
                "shell_execution_allowed": False,
            },
        )
        _adopt_gine(spec, state=state, launch=launch)
    lock_path = root / ".writer.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FiveBackboneExecutionError("another orchestrator owns this output root") from exc
        assert_main_matrix_unchanged(launch)
        waiting, reason, _ = _main_waiting(main_ready_gpu_tasks)
        if waiting:
            state.set_state("PAUSED_MAIN_PRIORITY", pause_reason=reason)
            return dict(state.payload)
        completed_seeds = list(state.payload.get("completed_seeds", []))
        for seed in spec.seeds:
            if seed in completed_seeds:
                continue
            pause_event = threading.Event()
            abort_event = threading.Event()
            state.set_state("RUNNING", current_seed=seed, pause_reason=None)
            errors: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gnn-ablation-lane") as pool:
                futures = [
                    pool.submit(
                        _run_lane,
                        lane,
                        spec,
                        seed=seed,
                        state=state,
                        live_queue=main_ready_gpu_tasks,
                        launch_evidence=launch,
                        pause_event=pause_event,
                        abort_event=abort_event,
                        poll_seconds=poll_seconds,
                    )
                    for lane in EXECUTION_LANES
                ]
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append(exc)
            assert_main_matrix_unchanged(launch)
            if errors:
                state.set_state(
                    "FAILED",
                    failure_reasons=[f"{type(error).__name__}:{error}" for error in errors],
                    current_seed=seed,
                )
                raise FiveBackboneExecutionError(
                    "five-backbone lane failed: " + "; ".join(str(error) for error in errors)
                )
            expected = {
                f"{backbone}:seed{seed}" for backbone in EXECUTED_BACKBONES
            }
            passed = {
                task_id
                for task_id, receipt in state.payload.get("tasks", {}).items()
                if isinstance(receipt, Mapping) and receipt.get("state") == "PASS"
            }
            if not expected.issubset(passed):
                state.set_state(
                    "PAUSED_MAIN_PRIORITY",
                    current_seed=seed,
                    pause_reason="MAIN_TASK_READY_WAITING_GPU",
                )
                return dict(state.payload)
            completed_seeds.append(seed)
            state.set_state(
                "RUNNING",
                completed_seeds=completed_seeds,
                current_seed=None,
                max_concurrent_gpus=2,
            )
            waiting, reason, _ = _main_waiting(main_ready_gpu_tasks)
            if waiting:
                state.set_state("PAUSED_MAIN_PRIORITY", pause_reason=reason)
                return dict(state.payload)
        assert_main_matrix_unchanged(launch)
        terminal: dict[str, Any] = {
            "schema_version": "gnn_five_backbone_orchestrator_terminal_v1",
            "status": "PASS",
            "run_id": spec.run_id,
            "run_spec_sha256": spec.run_spec_sha256,
            "launch_status_sha256": launch.status_sha256,
            "gine_reference_adopted": True,
            "gine_science_retrained": False,
            "executed_backbones": list(EXECUTED_BACKBONES),
            "seeds": list(spec.seeds),
            "schedule": {lane: list(values) for lane, values in EXECUTION_LANES.items()},
            "max_concurrent_gpus": 2,
            "main_matrix_modified": False,
            "task_receipts": dict(state.payload.get("tasks", {})),
        }
        terminal["terminal_sha256"] = canonical_json_sha256(terminal)
        _atomic_json(root / "terminal.json", terminal)
        state.set_state(
            "PASS",
            completed_seeds=list(spec.seeds),
            current_seed=None,
            terminal_sha256=terminal["terminal_sha256"],
            main_matrix_modified=False,
        )
        return dict(state.payload)


__all__ = [
    "ALLOWED_SEED_PREFIX",
    "EXECUTED_BACKBONES",
    "EXECUTION_LANES",
    "EXECUTION_SPEC_SCHEMA",
    "FiveBackboneExecutionError",
    "FiveBackboneExecutionSpec",
    "LaunchEvidence",
    "ScienceCommandTemplate",
    "TASK_CHECKPOINT_SCHEMA",
    "TASK_TERMINAL_SCHEMA",
    "assert_main_matrix_unchanged",
    "canonical_json_sha256",
    "load_execution_spec",
    "load_launch_evidence",
    "run_five_backbone_execution",
    "sha256_file",
]
