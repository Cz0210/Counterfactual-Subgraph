#!/usr/bin/env python3
"""Fail-closed four-lane AutoDL process orchestrator.

This controller deliberately models AutoDL work as local operating-system
processes.  It never submits Slurm jobs and it never records an AutoDL PID as a
Slurm job id.  The scientific commands live in a data file so that this module
only owns process isolation, dependency gates, persistent state, and recovery.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


RUN_ID = "autodl_three_lines_20260821_v1"
SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 2
LANE_STATES = {
    "NOT_STARTED",
    "CREATED",
    "STARTING",
    "WAITING_DEPENDENCY",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "STOPPING",
    "STOPPED",
    "ORPHANED_CHILD",
}
TERMINAL_LANE_STATES = {"SUCCEEDED", "FAILED", "STOPPED", "ORPHANED_CHILD"}
SECRET_OPTION = re.compile(
    r"(?i)(password|passwd|secret|token|authorization|api[_-]?key|"
    r"credential|private[_-]?key)"
)
SECRET_VALUE = re.compile(
    r"(?i)(?:BEGIN [A-Z ]*PRIVATE KEY|"
    r"(?:^|[?&;\s])(?:password|passwd|token|secret|authorization|"
    r"api[_-]?key|credential|private[_-]?key)\s*=|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bsk-[A-Za-z0-9_-]{16,}|"
    r"\bAKIA[0-9A-Z]{16}\b)"
)
PLACEHOLDER = re.compile(r"^__CONFIGURE_[A-Z0-9_]+__$")
TOKEN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


class OrchestratorError(RuntimeError):
    """A fail-closed orchestration error with an actionable message."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OrchestratorError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(32 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_or_json(path: Path, *, text: str | None = None) -> dict[str, Any]:
    """Load JSON-compatible YAML without making PyYAML a controller bootstrap.

    JSON is a strict subset of YAML.  The committed production spec uses that
    subset so a broken Python environment cannot prevent ``status``/``stop``.
    Human-authored YAML remains supported when PyYAML is available.
    """

    if text is None:
        text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise OrchestratorError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise OrchestratorError(f"Spec must contain one mapping: {path}")
    return payload


def _replace_tokens(value: str, context: Mapping[str, str]) -> str:
    unknown = sorted(set(TOKEN.findall(value)) - set(context))
    if unknown:
        raise OrchestratorError(f"Unknown template tokens {unknown} in {value!r}")
    for key, replacement in context.items():
        value = value.replace("{" + key + "}", replacement)
    return value


def _absolute_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise OrchestratorError(f"{label} must be absolute: {value}")
    if path == Path("/"):
        raise OrchestratorError(f"{label} cannot be filesystem root")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise OrchestratorError(
                f"{label} contains a symbolic-link component: {current}"
            )
        if not current.exists():
            break
    return path.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _redacted_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for value in command:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("--") and "=" in value:
            option, option_value = value.split("=", 1)
            result.append(
                f"{option}=<redacted>" if SECRET_OPTION.search(option) else value
            )
            continue
        result.append(value)
        if value.startswith("-") and SECRET_OPTION.search(value):
            redact_next = True
    return result


def _assert_no_embedded_secrets(command: Sequence[str]) -> None:
    for index, value in enumerate(command):
        option = value.split("=", 1)[0] if value.startswith("-") else ""
        if option and SECRET_OPTION.search(option):
            raise OrchestratorError(
                "Commands may not accept password/token/secret options; use a "
                f"credential provider instead (argument {index}, option={option})"
            )
        if SECRET_VALUE.search(value):
            raise OrchestratorError(
                "Commands may not embed credential values; use a credential "
                f"provider instead (argument {index})"
            )
        if "=" in value and SECRET_OPTION.search(value.split("=", 1)[0]):
            raise OrchestratorError(
                "Commands may not embed credential values; use a credential "
                f"provider instead (argument {index})"
            )


def _assert_no_secret_environment(environment: Mapping[str, str]) -> None:
    for key, value in environment.items():
        if SECRET_OPTION.search(str(key)):
            raise OrchestratorError(
                f"Stage environment may not contain credential key {key!r}"
            )
        if SECRET_VALUE.search(str(value)):
            raise OrchestratorError(
                f"Stage environment value for {key!r} resembles a credential"
            )


def _sanitized_inherited_environment() -> dict[str, str]:
    """Drop credential-bearing shell variables before spawning any worker."""

    return {
        str(key): str(value)
        for key, value in os.environ.items()
        if not SECRET_OPTION.search(str(key)) and not SECRET_VALUE.search(str(value))
    }


FORMAL_STAGE_NAMES = {
    "mut-freeze",
    "mut-downstream",
    "aids-freeze",
    "aids-downstream",
    "bace-generate",
    "bace-final",
    "bace-globalgce",
    "bace-common4",
}


def _completion_verifier_command(stage: Mapping[str, Any]) -> list[str]:
    """Derive the only accepted formal completion verifier invocation."""

    command = [str(value) for value in stage.get("command") or []]
    if (
        len(command) < 3
        or Path(command[1]).name != "run_three_lines_stage.py"
        or command[2] not in FORMAL_STAGE_NAMES
        or "--verify-only" in command
        or "--resume" in command
    ):
        raise OrchestratorError(
            f"Formal stage {stage.get('id')} lacks the audited stage verifier"
        )
    return [*command, "--verify-only"]


def _git_output(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise OrchestratorError(
            f"git {' '.join(arguments)} failed under {project_root}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _external_worktree_lineage(root: Path) -> dict[str, str]:
    """Accept only the pinned tree plus the known non-code provenance file."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise OrchestratorError("Unable to inspect external COMRECGC worktree")
    records = [value for value in result.stdout.split(b"\0") if value]
    allowed = b"?? vendor_manifest.json"
    if any(value != allowed for value in records):
        raise OrchestratorError(
            "External COMRECGC checkout has tracked/staged or unapproved "
            "untracked changes"
        )
    provenance = root / "vendor_manifest.json"
    if allowed in records:
        if not provenance.is_file() or provenance.is_symlink():
            raise OrchestratorError(
                "External vendor_manifest.json is not a physical root file"
            )
        provenance_digest = sha256_file(provenance)
    else:
        provenance_digest = "ABSENT"
    tree = _git_output(root, "rev-parse", "HEAD^{tree}")
    if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise OrchestratorError("External COMRECGC tree object is invalid")
    return {
        "external_comrecgc_tree": tree,
        "external_provenance_sha256": provenance_digest,
    }


def _git_is_ancestor(project_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise OrchestratorError(
            f"git merge-base failed under {project_root}: {result.stderr.strip()}"
        )
    return result.returncode == 0


def _dependency_order(lanes: Sequence[dict[str, Any]]) -> list[str]:
    stages: dict[str, list[str]] = {}
    declared_order: list[str] = []
    for lane in lanes:
        lane_id = str(lane["id"])
        previous: str | None = None
        for stage in lane["stages"]:
            ref = f"{lane_id}:{stage['id']}"
            declared_order.append(ref)
            dependencies = [str(value) for value in stage.get("dependencies", [])]
            if previous is not None and previous not in dependencies:
                dependencies.append(previous)
            stages[ref] = dependencies
            previous = ref
    unknown = sorted(
        {dependency for values in stages.values() for dependency in values} - set(stages)
    )
    if unknown:
        raise OrchestratorError(f"Unknown stage dependencies: {unknown}")
    incoming = {key: len(value) for key, value in stages.items()}
    children = {key: [] for key in stages}
    for child, dependencies in stages.items():
        for dependency in dependencies:
            children[dependency].append(child)
    ready = [value for value in declared_order if incoming[value] == 0]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in children[current]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(ordered) != len(stages):
        raise OrchestratorError("Stage dependencies contain a cycle")
    return ordered


class LoadedSpec:
    def __init__(
        self,
        path: Path,
        data: dict[str, Any],
        *,
        spec_sha256: str,
    ) -> None:
        self.path = path.resolve()
        self.data = data
        self.spec_sha256 = spec_sha256
        self.run_id = str(data.get("run_id") or "")
        roots = data.get("roots")
        if not isinstance(roots, dict):
            raise OrchestratorError("Spec roots must be a mapping")
        self.project_root = _absolute_path(str(roots.get("project") or ""), "roots.project")
        self.persistent_root = _absolute_path(
            str(roots.get("persistent_run") or ""), "roots.persistent_run"
        )
        self.fast_root = _absolute_path(str(roots.get("fast_run") or ""), "roots.fast_run")
        self.external_root = _absolute_path(
            str(roots.get("external_comrecgc") or ""), "roots.external_comrecgc"
        )
        self.context = {
            "run_id": self.run_id,
            "project_root": str(self.project_root),
            "persistent_root": str(self.persistent_root),
            "fast_root": str(self.fast_root),
            "external_root": str(self.external_root),
        }
        raw_lanes = data.get("lanes")
        if not isinstance(raw_lanes, list):
            raise OrchestratorError("Spec lanes must be a list")
        self.lanes = [self._resolve_lane(value) for value in raw_lanes]
        self.lane_by_id = {str(value["id"]): value for value in self.lanes}
        self.runtime = dict(data.get("runtime") or {})
        self.provenance = dict(data.get("provenance") or {})
        self.policy = dict(data.get("policy") or {})
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "LoadedSpec":
        resolved = path.resolve(strict=True)
        raw = resolved.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OrchestratorError(f"Spec is not valid UTF-8: {resolved}") from exc
        loaded = cls(
            resolved,
            load_yaml_or_json(resolved, text=text),
            spec_sha256=digest,
        )
        if sha256_file(resolved) != digest:
            raise OrchestratorError(f"Spec changed while it was being loaded: {resolved}")
        return loaded

    def _resolve_lane(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise OrchestratorError("Every lane must be a mapping")
        lane_id = str(raw.get("id") or "")
        lane_context = {
            **self.context,
            "lane_id": lane_id,
            "gpu_id": str(raw.get("gpu_id")),
        }
        lane = dict(raw)
        for key in ("input_root", "input_manifest", "output_root", "cache_root", "active_root"):
            lane[key] = _replace_tokens(str(raw.get(key) or ""), lane_context)
        stages: list[dict[str, Any]] = []
        for raw_stage in raw.get("stages") or []:
            if not isinstance(raw_stage, dict):
                raise OrchestratorError(f"Lane {lane_id} has a non-mapping stage")
            stage = dict(raw_stage)
            stage_id = str(stage.get("id") or "")
            stage_context = {**lane_context, "stage_id": stage_id}
            for key in ("command", "resume_command"):
                command = stage.get(key) or []
                if not isinstance(command, list):
                    raise OrchestratorError(f"{lane_id}:{stage_id} {key} must be a list")
                stage[key] = [_replace_tokens(str(value), stage_context) for value in command]
            stage["required_success_sentinel"] = _replace_tokens(
                str(stage.get("required_success_sentinel") or ""), stage_context
            )
            output_manifest = str(stage.get("output_manifest") or "")
            stage["output_manifest"] = (
                _replace_tokens(output_manifest, stage_context) if output_manifest else ""
            )
            output_manifest_root = str(stage.get("output_manifest_root") or "")
            stage["output_manifest_root"] = (
                _replace_tokens(output_manifest_root, stage_context)
                if output_manifest_root
                else ""
            )
            progress_json = str(stage.get("progress_json") or "")
            stage["progress_json"] = (
                _replace_tokens(progress_json, stage_context) if progress_json else ""
            )
            environment = stage.get("environment") or {}
            if not isinstance(environment, dict):
                raise OrchestratorError(
                    f"{lane_id}:{stage_id} environment must be a mapping"
                )
            stage["environment"] = {
                str(key): _replace_tokens(str(value), stage_context)
                for key, value in environment.items()
            }
            stages.append(stage)
        lane["stages"] = stages
        return lane

    @property
    def state_root(self) -> Path:
        return self.persistent_root / "state"

    @property
    def run_state_path(self) -> Path:
        return self.state_root / "run_state.json"

    @property
    def global_lock_path(self) -> Path:
        return self.state_root / "control.lock"

    @property
    def registry_path(self) -> Path:
        return self.persistent_root / "registry" / "autodl_processes.jsonl"

    def lane_state_root(self, lane_id: str) -> Path:
        return self.state_root / "lanes" / lane_id

    def validate(self) -> None:
        if self.data.get("schema_version") != SCHEMA_VERSION:
            raise OrchestratorError(
                "Spec schema_version must be exactly "
                f"{SCHEMA_VERSION}, found {self.data.get('schema_version')!r}"
            )
        if self.run_id != RUN_ID:
            raise OrchestratorError(f"run_id must be exactly {RUN_ID}")
        if self.project_root in {self.persistent_root, self.fast_root}:
            raise OrchestratorError("Project, persistent run, and fast run roots must differ")
        if self.persistent_root == self.fast_root:
            raise OrchestratorError("Persistent and fast roots must differ")
        ids = [str(value.get("id") or "") for value in self.lanes]
        if len(ids) != 4 or len(set(ids)) != 4 or any(not value for value in ids):
            raise OrchestratorError("Exactly four uniquely named lanes are required")
        gpu_ids = [int(value.get("gpu_id")) for value in self.lanes]
        if sorted(gpu_ids) != [0, 1, 2, 3]:
            raise OrchestratorError("The four lanes must map one-to-one to GPU IDs 0,1,2,3")
        for lane in self.lanes:
            lane_id = str(lane["id"])
            if not lane.get("stages"):
                raise OrchestratorError(f"Lane {lane_id} has no stages")
            for path_key in (
                "input_root",
                "input_manifest",
                "output_root",
                "cache_root",
                "active_root",
            ):
                _absolute_path(str(lane[path_key]), f"{lane_id}.{path_key}")
            input_root = Path(str(lane["input_root"])).resolve(strict=False)
            input_manifest = Path(str(lane["input_manifest"])).resolve(strict=False)
            output_root = Path(str(lane["output_root"])).resolve(strict=False)
            if not _is_within(input_manifest, input_root):
                raise OrchestratorError(
                    f"Lane {lane_id} input manifest escapes input root"
                )
            if _is_within(output_root, input_root) or _is_within(input_root, output_root):
                raise OrchestratorError(
                    f"Lane {lane_id} input and output roots must be disjoint"
                )
            if not (
                _is_within(output_root, self.persistent_root)
                or _is_within(output_root, self.fast_root)
            ):
                raise OrchestratorError(f"Lane {lane_id} output escapes managed roots")
            for stage in lane["stages"]:
                ref = f"{lane_id}:{stage.get('id')}"
                if not stage.get("id"):
                    raise OrchestratorError(f"Lane {lane_id} has a stage without id")
                command = stage.get("command") or []
                if not command:
                    raise OrchestratorError(f"{ref} has no command")
                for command_key in ("command", "resume_command"):
                    _assert_no_embedded_secrets(stage.get(command_key) or [])
                _assert_no_secret_environment(stage.get("environment") or {})
                if self.policy.get("require_formal_stage_verifier") is True:
                    _completion_verifier_command(stage)
                sentinel = _absolute_path(
                    str(stage.get("required_success_sentinel") or ""),
                    f"{ref}.required_success_sentinel",
                )
                if not _is_within(sentinel, output_root):
                    raise OrchestratorError(f"{ref} success sentinel escapes output root")
                output_manifest_text = str(stage.get("output_manifest") or "")
                if output_manifest_text:
                    output_manifest = _absolute_path(
                        output_manifest_text, f"{ref}.output_manifest"
                    )
                    if not _is_within(output_manifest, output_root):
                        raise OrchestratorError(
                            f"{ref} output manifest escapes output root"
                        )
                    manifest_root_text = str(
                        stage.get("output_manifest_root") or output_manifest.parent
                    )
                    manifest_root = _absolute_path(
                        manifest_root_text, f"{ref}.output_manifest_root"
                    )
                    if not _is_within(manifest_root, output_root):
                        raise OrchestratorError(
                            f"{ref} output manifest root escapes output root"
                        )
                progress_json = str(stage.get("progress_json") or "")
                if progress_json:
                    progress_path = _absolute_path(
                        progress_json, f"{ref}.progress_json"
                    )
                    active_root = Path(str(lane["active_root"])).resolve(strict=False)
                    if not _is_within(progress_path, active_root):
                        raise OrchestratorError(
                            f"{ref} progress JSON escapes the lane active root"
                        )
            if str(lane.get("dataset")).lower() in {"mutagenicity", "aids"}:
                if lane.get("generation_policy") != "preserved_freeze_only":
                    raise OrchestratorError(
                        f"Lane {lane_id} must use preserved_freeze_only"
                    )
                for stage in lane["stages"]:
                    if str(stage["environment"].get("DISALLOW_GENERATION")) != "1":
                        raise OrchestratorError(
                            f"{lane_id}:{stage['id']} must set DISALLOW_GENERATION=1"
                        )
                    forbidden_entrypoints = {
                        "run_generation.py",
                        "comrecgc_project_generate.sh",
                        "comrecgc_mut_full.sh",
                        "comrecgc_aids_native_full.sh",
                    }
                    command_names = {
                        Path(value).name
                        for key in ("command", "resume_command")
                        for value in stage.get(key) or []
                    }
                    command_text = "\n".join(
                        value
                        for key in ("command", "resume_command")
                        for value in stage.get(key) or []
                    )
                    unsafe = sorted(
                        value
                        for value in forbidden_entrypoints
                        if value in command_names or value in command_text
                    )
                    if unsafe:
                        raise OrchestratorError(
                            f"Preserved lane {lane_id} invokes generation: {unsafe}"
                        )
        write_roots = [
            (str(lane["id"]), key, Path(str(lane[key])).resolve(strict=False))
            for lane in self.lanes
            for key in ("output_root", "cache_root", "active_root")
        ]
        for index, (lane_id, key, path) in enumerate(write_roots):
            for other_lane, other_key, other_path in write_roots[index + 1 :]:
                if _is_within(path, other_path) or _is_within(other_path, path):
                    raise OrchestratorError(
                        "Lane write roots must be disjoint: "
                        f"{lane_id}.{key}={path}, "
                        f"{other_lane}.{other_key}={other_path}"
                    )
        _dependency_order(self.lanes)
        if self.policy.get("strict_three_line_topology", True):
            expected = {
                "mut_recovery": 0,
                "aids_recovery": 1,
                "bace_comrecgc": 2,
                "bace_globalgce_common4": 3,
            }
            actual = {str(value["id"]): int(value["gpu_id"]) for value in self.lanes}
            if actual != expected:
                raise OrchestratorError(f"Unexpected strict lane topology: {actual}")
            common4 = next(
                stage
                for stage in self.lane_by_id["bace_globalgce_common4"]["stages"]
                if stage["id"] == "bace_common4"
            )
            required = {
                "bace_comrecgc:bace_comrecgc_final",
                "bace_globalgce_common4:bace_globalgce_wnode",
            }
            if not required.issubset(set(common4.get("dependencies") or [])):
                raise OrchestratorError(
                    "bace_common4 must depend on both BACE_COMRECGC and "
                    "BACE_GLOBALGCE_WNODE completion"
                )


def _spec_state_binding(spec: LoadedSpec) -> dict[str, Any]:
    """Return the immutable controller/spec/root identity for persisted state."""

    current_digest = sha256_file(spec.path)
    if current_digest != spec.spec_sha256:
        raise OrchestratorError(f"Loaded spec changed on disk: {spec.path}")
    return {
        "schema_version": SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "backend": "autodl",
        "run_id": spec.run_id,
        "spec_schema_version": spec.data.get("schema_version"),
        "spec_path": str(spec.path),
        "spec_sha256": spec.spec_sha256,
        "project_root": str(spec.project_root),
        "external_root": str(spec.external_root),
        "persistent_root": str(spec.persistent_root),
        "fast_root": str(spec.fast_root),
        "roots": {
            "project": str(spec.project_root),
            "external_comrecgc": str(spec.external_root),
            "persistent_run": str(spec.persistent_root),
            "fast_run": str(spec.fast_root),
            "state": str(spec.state_root),
        },
    }


def _validate_run_state_binding(
    spec: LoadedSpec, payload: Mapping[str, Any]
) -> None:
    """Reject state published by another spec, run, schema, or root layout."""

    expected = _spec_state_binding(spec)
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise OrchestratorError(
            f"Persisted run state binding mismatch: {mismatches}"
        )


def _load_bound_run_state(spec: LoadedSpec) -> dict[str, Any]:
    if spec.run_state_path.is_symlink() or not spec.run_state_path.is_file():
        raise OrchestratorError(
            "Persisted run state is missing or non-physical: "
            f"{spec.run_state_path}"
        )
    payload = read_json(spec.run_state_path)
    _validate_run_state_binding(spec, payload)
    return payload


def _selected_lanes(
    spec: LoadedSpec, lane_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Resolve a repeatable public lane selection in stable spec order."""

    if lane_ids is None or len(lane_ids) == 0:
        return list(spec.lanes)
    requested = {str(value) for value in lane_ids}
    unknown = sorted(requested - set(spec.lane_by_id))
    if unknown:
        raise OrchestratorError(
            f"Unknown lane selection {unknown}; valid lanes={sorted(spec.lane_by_id)}"
        )
    return [lane for lane in spec.lanes if str(lane["id"]) in requested]


@contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise OrchestratorError(f"Lock is already held: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lane_paths(spec: LoadedSpec, lane_id: str) -> dict[str, Path]:
    root = spec.lane_state_root(lane_id)
    return {
        "root": root,
        "state": root / "lane_state.json",
        "pid": root / "worker_pid.json",
        "lock": root / "writer.lock",
        "heartbeat": root / "heartbeat.json",
        "stop": root / "STOP_REQUESTED.json",
        "lane_success": root / "LANE_SUCCESS.json",
        "lane_failure": root / "LANE_FAILED.json",
        "sentinels": root / "sentinels",
        "provenance": root / "provenance",
    }


def _stage_ref_sentinel(spec: LoadedSpec, ref: str) -> Path:
    lane_id, stage_id = ref.split(":", 1)
    return _lane_paths(spec, lane_id)["sentinels"] / f"{stage_id}.SUCCESS.json"


def _stage_from_ref(
    spec: LoadedSpec, ref: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    try:
        lane_id, stage_id = ref.split(":", 1)
        lane = spec.lane_by_id[lane_id]
        stage = next(value for value in lane["stages"] if value["id"] == stage_id)
    except (KeyError, ValueError, StopIteration) as exc:
        raise OrchestratorError(f"Unknown stage reference: {ref}") from exc
    return lane, stage


def _dependencies_satisfied(
    spec: LoadedSpec,
    current_lane_id: str,
    stage: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    dependencies = [str(value) for value in stage.get("dependencies") or []]
    missing: list[str] = []
    for ref in dependencies:
        lane, dependency_stage = _stage_from_ref(spec, ref)
        dependency_lane_id = str(lane["id"])
        dependency_stage_id = str(dependency_stage["id"])
        sentinel = _stage_ref_sentinel(spec, ref)
        if sentinel.is_symlink() or not sentinel.is_file():
            missing.append(ref)
            continue
        dependency_state = _load_lane_state(spec, lane)
        persisted_stage = (dependency_state.get("stages") or {}).get(
            dependency_stage_id
        )
        if (
            not isinstance(persisted_stage, Mapping)
            or persisted_stage.get("status") != "SUCCEEDED"
        ):
            missing.append(ref)
            continue
        # A cross-lane release is valid only after the producing lane has
        # durably published its own terminal success.  Same-lane dependencies
        # instead bind the completed stage record while the lane is RUNNING.
        if (
            dependency_lane_id != current_lane_id
            and dependency_state.get("status") != "SUCCEEDED"
        ):
            missing.append(ref)
            continue
        completion = _validate_completed_stage(spec, lane, dependency_stage)
        expected_stage_binding = {
            "stage": dependency_stage_id,
            "status": "SUCCEEDED",
            "sentinel": str(sentinel),
            "output_manifest_digest": completion["output_manifest_digest"],
        }
        stage_mismatches = {
            key: {"expected": value, "actual": persisted_stage.get(key)}
            for key, value in expected_stage_binding.items()
            if persisted_stage.get(key) != value
        }
        if stage_mismatches:
            raise OrchestratorError(
                "Persisted dependency stage binding mismatch for "
                f"{ref}: {stage_mismatches}"
            )
        if dependency_lane_id != current_lane_id:
            _validate_completed_lane(spec, lane, dependency_state)
    if missing:
        return False, missing
    return True, []


def _initial_lane_state(spec: LoadedSpec, lane: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "backend": "autodl",
        "run_id": spec.run_id,
        "spec_sha256": spec.spec_sha256,
        "lane": lane["id"],
        "gpu_id": int(lane["gpu_id"]),
        "dataset": lane["dataset"],
        "method": lane["method"],
        "status": "NOT_STARTED",
        "worker_pid": None,
        "worker_identity": None,
        "child_pid": None,
        "child_identity": None,
        "current_stage": None,
        "current_command": [],
        "retry_count": 0,
        "dependency_status": {},
        "latest_checkpoint": None,
        "input_root": lane["input_root"],
        "output_root": lane["output_root"],
        "cache_root": lane["cache_root"],
        "active_root": lane["active_root"],
        "input_manifest_digest": None,
        "output_manifest_digest": None,
        "git_commit": None,
        "external_commit": None,
        "started_at": None,
        "updated_at": utc_now(),
        "finished_at": None,
        "failure": None,
        "stages": {},
    }


def _validate_lane_state_binding(
    spec: LoadedSpec,
    lane: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    lane_id = str(lane["id"])
    expected = {
        "schema_version": SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "backend": "autodl",
        "run_id": spec.run_id,
        "spec_sha256": spec.spec_sha256,
        "lane": lane_id,
        "gpu_id": int(lane["gpu_id"]),
        "input_root": str(lane["input_root"]),
        "output_root": str(lane["output_root"]),
        "cache_root": str(lane["cache_root"]),
        "active_root": str(lane["active_root"]),
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise OrchestratorError(
            f"Persisted lane state binding mismatch for {lane_id}: {mismatches}"
        )
    if str(state.get("status")) not in LANE_STATES:
        raise OrchestratorError(
            f"Persisted lane state has invalid status for {lane_id}: "
            f"{state.get('status')!r}"
        )


def _load_lane_state(spec: LoadedSpec, lane: Mapping[str, Any]) -> dict[str, Any]:
    path = _lane_paths(spec, str(lane["id"]))["state"]
    if path.is_symlink():
        raise OrchestratorError(f"Lane state must be a physical file: {path}")
    if not path.is_file():
        if spec.run_state_path.is_file():
            raise OrchestratorError(
                f"Persisted run is missing lane state for {lane['id']}: {path}"
            )
        return _initial_lane_state(spec, lane)
    state = read_json(path)
    _validate_lane_state_binding(spec, lane, state)
    return state


def _lane_was_never_started(state: Mapping[str, Any]) -> bool:
    return (
        str(state.get("status")) in NEVER_STARTED_LANE_STATES
        and state.get("started_at") is None
        and state.get("worker_pid") is None
        and state.get("child_pid") is None
        and int(state.get("retry_count") or 0) == 0
        and not (state.get("stages") or {})
    )


def _save_lane_state(spec: LoadedSpec, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    lane_id = str(state.get("lane") or "")
    if lane_id not in spec.lane_by_id:
        raise OrchestratorError(f"Cannot persist state for unknown lane {lane_id!r}")
    _validate_lane_state_binding(spec, spec.lane_by_id[lane_id], state)
    atomic_write_json(_lane_paths(spec, str(state["lane"]))["state"], state)


def _canonical_commands(spec: LoadedSpec) -> dict[str, str]:
    script = spec.project_root / "scripts" / "autodl" / "three_lines.sh"
    return {
        action: f"{shlex.quote(str(script))} {action} --spec {shlex.quote(str(spec.path))}"
        for action in ("status", "resume", "stop")
    }


ACTIVE_LANE_STATES = {
    "STARTING",
    "RUNNING",
    "WAITING_DEPENDENCY",
    "STOPPING",
}
FAILED_LANE_STATES = {"FAILED", "ORPHANED_CHILD", "STALE_COMPLETION"}
NEVER_STARTED_LANE_STATES = {"NOT_STARTED", "CREATED"}


def _aggregate_run_status(statuses: Sequence[str]) -> str:
    """Summarize partial lane activation without treating omission as failure."""

    values = [str(value) for value in statuses]
    if values and all(value == "SUCCEEDED" for value in values):
        return "LANES_COMPLETED"
    if any(value in FAILED_LANE_STATES for value in values):
        return "BLOCKED"
    if any(value in ACTIVE_LANE_STATES for value in values):
        return "RUNNING"
    if values and all(value in NEVER_STARTED_LANE_STATES for value in values):
        return "NOT_STARTED"
    if any(value == "STOPPED" for value in values):
        if all(
            value == "STOPPED" or value in NEVER_STARTED_LANE_STATES
            for value in values
        ):
            return "STOPPED"
        return "PARTIALLY_STOPPED"
    if any(value == "SUCCEEDED" for value in values):
        return "PARTIALLY_COMPLETED"
    return "NOT_STARTED"


def _lane_summary_sets(lanes: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        "active_lanes": sorted(
            lane_id
            for lane_id, state in lanes.items()
            if str(state.get("status")) in ACTIVE_LANE_STATES
        ),
        "not_started_lanes": sorted(
            lane_id
            for lane_id, state in lanes.items()
            if str(state.get("status")) in NEVER_STARTED_LANE_STATES
        ),
        "succeeded_lanes": sorted(
            lane_id
            for lane_id, state in lanes.items()
            if str(state.get("status")) == "SUCCEEDED"
        ),
    }


def _refresh_run_state(spec: LoadedSpec) -> dict[str, Any]:
    with file_lock(spec.global_lock_path):
        previous = _load_bound_run_state(spec)
        lanes = {
            str(lane["id"]): _load_lane_state(spec, lane) for lane in spec.lanes
        }
        statuses = [str(value["status"]) for value in lanes.values()]
        status = _aggregate_run_status(statuses)
        payload = {
            **_spec_state_binding(spec),
            "status": status,
            "created_at": previous.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "spec_path": str(spec.path),
            "project_root": str(spec.project_root),
            "persistent_root": str(spec.persistent_root),
            "fast_root": str(spec.fast_root),
            "external_root": str(spec.external_root),
            "lanes": {
                key: {
                    "status": value["status"],
                    "gpu_id": value["gpu_id"],
                    "worker_pid": value["worker_pid"],
                    "worker_identity": value.get("worker_identity"),
                    "child_pid": value["child_pid"],
                    "child_identity": value.get("child_identity"),
                    "current_stage": value["current_stage"],
                    "retry_count": value["retry_count"],
                    "latest_checkpoint": value["latest_checkpoint"],
                    "heartbeat": str(_lane_paths(spec, key)["heartbeat"]),
                }
                for key, value in lanes.items()
            },
            **_lane_summary_sets(lanes),
            "commands": _canonical_commands(spec),
            "slurm_jobs": [],
            "autodl_pid_is_slurm_job_id": False,
        }
        atomic_write_json(spec.run_state_path, payload)
        return payload


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 1:
            return False
        os.kill(value, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _pid_matches_worker(pid: int, spec: LoadedSpec, lane_id: str) -> bool:
    if lane_id not in spec.lane_by_id or not _pid_alive(pid):
        return False
    paths = _lane_paths(spec, lane_id)
    if paths["pid"].is_symlink() or not paths["pid"].is_file():
        return False
    try:
        record = read_json(paths["pid"])
        state = _load_lane_state(spec, spec.lane_by_id[lane_id])
        recorded_pid = int(record.get("pid", -1))
        state_pid = int(state.get("worker_pid", -1))
    except (OrchestratorError, TypeError, ValueError):
        return False
    expected_record = {
        "schema_version": SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "backend": "autodl",
        "kind": "autodl_worker_pid",
        "run_id": spec.run_id,
        "lane": lane_id,
        "spec_sha256": spec.spec_sha256,
    }
    if any(record.get(key) != value for key, value in expected_record.items()):
        return False
    identity = record.get("worker_identity")
    if recorded_pid != int(pid) or state_pid != int(pid):
        return False
    if state.get("worker_identity") != identity:
        return False
    return _worker_identity_matches(
        identity,
        spec,
        lane_id,
        _worker_command(spec, lane_id),
    )


def _signal_exact_worker(
    pid: int,
    spec: LoadedSpec,
    lane_id: str,
    signum: int,
) -> bool:
    """Signal only the exact worker; Linux uses pidfd to close the reuse race."""

    if not _pid_matches_worker(pid, spec, lane_id):
        return False
    if Path("/proc").is_dir():
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if not callable(pidfd_open) or not callable(pidfd_send_signal):
            raise OrchestratorError(
                "Linux worker signalling requires pidfd_open and "
                "pidfd_send_signal"
            )
        try:
            descriptor = int(pidfd_open(pid, 0))
        except ProcessLookupError:
            return False
        try:
            if not _pid_matches_worker(pid, spec, lane_id):
                return False
            try:
                pidfd_send_signal(descriptor, signum, None, 0)
            except ProcessLookupError:
                return False
            return True
        finally:
            os.close(descriptor)
    if not _pid_matches_worker(pid, spec, lane_id):
        return False
    os.kill(pid, signum)
    return True


def _command_sha256(command: Sequence[str]) -> str:
    """Return an unambiguous digest of an argv vector without persisting argv."""

    encoded = json.dumps(
        [str(value) for value in command],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_linux_proc_stat(payload: bytes) -> tuple[str, int]:
    """Extract Linux ``/proc/<pid>/stat`` starttime and process group.

    The command name is parenthesized and may contain spaces or parentheses, so
    splitting the entire line is unsafe.  Fields after the final ``") "`` begin
    at field 3 (state); pgrp is field 5 and starttime is field 22.
    """

    close = payload.rfind(b") ")
    if close < 0:
        raise OrchestratorError("Malformed /proc process stat record")
    fields = payload[close + 2 :].split()
    if len(fields) <= 19:
        raise OrchestratorError("Truncated /proc process stat record")
    try:
        pgid = int(fields[2])
        starttime = fields[19].decode("ascii")
    except (ValueError, UnicodeDecodeError) as exc:
        raise OrchestratorError("Invalid /proc process identity fields") from exc
    if pgid <= 1 or not starttime.isdigit():
        raise OrchestratorError("Unsafe /proc process identity fields")
    return starttime, pgid


def _current_process_identity(pid: int) -> dict[str, Any] | None:
    """Read a PID-reuse-resistant identity for one live process.

    AutoDL is Linux, where this binds the kernel starttime, raw cmdline digest,
    and process group from procfs.  The ``ps`` fallback exists only so the
    controller's platform-neutral unit tests remain executable; it still binds
    start time, command text, and process group and is never confused with a
    procfs identity.
    """

    if pid <= 1 or not _pid_alive(pid):
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if stat_path.is_file() and cmdline_path.is_file():
        try:
            stat_payload_before = stat_path.read_bytes()
            cmdline = cmdline_path.read_bytes()
            stat_payload_after = stat_path.read_bytes()
            proc_starttime, pgid = _parse_linux_proc_stat(stat_payload_before)
            confirmed_starttime, confirmed_pgid = _parse_linux_proc_stat(
                stat_payload_after
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            return None
        if (proc_starttime, pgid) != (confirmed_starttime, confirmed_pgid):
            return None
        if not cmdline:
            return None
        return {
            "identity_source": "procfs",
            "proc_starttime": proc_starttime,
            "process_start_token": f"procfs:{proc_starttime}",
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
            "pgid": pgid,
        }

    values: dict[str, str] = {}
    for key, column in (
        ("start", "lstart="),
        ("pgid", "pgid="),
        ("command", "command="),
    ):
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", column],
                text=True,
                capture_output=True,
                check=False,
            )
        except (OSError, PermissionError):
            # Non-Linux/sandboxed development hosts may prohibit process-table
            # inspection. Such a child gets an unmatchable capture record; it
            # is never eligible for orphan adoption or signalling by PID.
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        values[key] = result.stdout.strip()
    try:
        pgid = int(values["pgid"])
    except ValueError:
        return None
    if pgid <= 1:
        return None
    return {
        "identity_source": "ps_fallback",
        "proc_starttime": None,
        "process_start_token": f"ps:{values['start']}",
        "cmdline_sha256": hashlib.sha256(
            values["command"].encode("utf-8")
        ).hexdigest(),
        "pgid": pgid,
    }


def _worker_command(spec: LoadedSpec, lane_id: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "_worker",
        "--spec",
        str(spec.path),
        "--lane",
        lane_id,
    ]


def _capture_worker_identity(
    pid: int,
    spec: LoadedSpec,
    lane_id: str,
    command: Sequence[str],
) -> dict[str, Any]:
    """Bind a controller worker to its exact kernel process and argv."""

    process = _current_process_identity(pid)
    base = {
        "schema_version": SCHEMA_VERSION,
        "kind": "autodl_worker",
        "pid": int(pid),
        "run_id": spec.run_id,
        "lane": lane_id,
        "spec_sha256": spec.spec_sha256,
        "command_sha256": _command_sha256(command),
        "captured_at": utc_now(),
    }
    if process is None:
        return {**base, "capture_status": "EXITED_BEFORE_IDENTITY_CAPTURE"}
    if (
        int(process.get("pgid", -1)) != int(pid)
        or (Path("/proc").is_dir() and process.get("identity_source") != "procfs")
    ):
        return {
            **base,
            "capture_status": "UNSAFE_PROCESS_GROUP",
            **process,
        }
    return {**base, "capture_status": "CAPTURED", **process}


def _worker_identity_matches(
    identity: Any,
    spec: LoadedSpec,
    lane_id: str,
    command: Sequence[str],
) -> bool:
    """Return true only for the exact recorded worker, never a reused PID."""

    if not isinstance(identity, Mapping) or identity.get("capture_status") != "CAPTURED":
        return False
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        pid <= 1
        or pgid != pid
        or identity.get("schema_version") != SCHEMA_VERSION
        or identity.get("kind") != "autodl_worker"
        or identity.get("run_id") != spec.run_id
        or identity.get("lane") != lane_id
        or identity.get("spec_sha256") != spec.spec_sha256
        or identity.get("command_sha256") != _command_sha256(command)
    ):
        return False
    current = _current_process_identity(pid)
    if current is None:
        return False
    for key in (
        "identity_source",
        "proc_starttime",
        "process_start_token",
        "cmdline_sha256",
        "pgid",
    ):
        if identity.get(key) != current.get(key):
            return False
    return True


def _capture_child_identity(
    pid: int,
    spec: LoadedSpec,
    lane_id: str,
    stage_id: str,
    command: Sequence[str],
) -> dict[str, Any]:
    """Bind a just-launched scientific child to this exact run/lane/stage."""

    process = _current_process_identity(pid)
    base = {
        "schema_version": SCHEMA_VERSION,
        "pid": int(pid),
        "run_id": spec.run_id,
        "lane": lane_id,
        "stage": stage_id,
        "command_sha256": _command_sha256(command),
        "captured_at": utc_now(),
    }
    if process is None:
        # A very short command can exit between Popen and procfs inspection. It
        # is intentionally never considered a live, signalable child.
        return {**base, "capture_status": "EXITED_BEFORE_IDENTITY_CAPTURE"}
    return {**base, "capture_status": "CAPTURED", **process}


def _child_identity_matches(
    identity: Any,
    spec: LoadedSpec,
    lane_id: str,
    stage_id: str | None,
    command: Sequence[str] | None,
) -> bool:
    """Return true only for the exact recorded child, never merely its PID."""

    if not isinstance(identity, Mapping) or identity.get("capture_status") != "CAPTURED":
        return False
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        pid <= 1
        or pgid <= 1
        or identity.get("run_id") != spec.run_id
        or identity.get("lane") != lane_id
        or not stage_id
        or identity.get("stage") != stage_id
    ):
        return False
    if command is None or identity.get("command_sha256") != _command_sha256(command):
        return False
    current = _current_process_identity(pid)
    if current is None:
        return False
    for key in (
        "identity_source",
        "proc_starttime",
        "process_start_token",
        "cmdline_sha256",
        "pgid",
    ):
        if identity.get(key) != current.get(key):
            return False
    return True


def _state_child_matches(
    spec: LoadedSpec, lane_id: str, state: Mapping[str, Any]
) -> bool:
    identity = state.get("child_identity")
    child_pid = state.get("child_pid")
    if not isinstance(identity, Mapping) or child_pid is None:
        return False
    try:
        if int(identity.get("pid", -1)) != int(child_pid):
            return False
    except (TypeError, ValueError):
        return False
    command = state.get("current_command")
    return _child_identity_matches(
        identity,
        spec,
        lane_id,
        str(state.get("current_stage") or "") or None,
        command if isinstance(command, list) else None,
    )


def _discard_stale_child_reference(
    state: dict[str, Any], *, reason: str
) -> None:
    """Audit and clear a stale/reused child PID without ever signalling it."""

    audit = list(state.get("stale_child_references") or [])
    audit.append(
        {
            "pid": state.get("child_pid"),
            "identity": state.get("child_identity"),
            "reason": reason,
            "discarded_at": utc_now(),
            "signal_sent": False,
        }
    )
    state["stale_child_references"] = audit
    state["child_pid"] = None
    state["child_identity"] = None


def _audit_stale_worker_reference(
    state: dict[str, Any],
    *,
    record: Mapping[str, Any] | None,
    reason: str,
    clear: bool,
) -> None:
    """Retain worker PID-reuse evidence without ever signalling that PID."""

    audit = list(state.get("stale_worker_references") or [])
    audit.append(
        {
            "pid": state.get("worker_pid") or (record or {}).get("pid"),
            "identity": state.get("worker_identity")
            or (record or {}).get("worker_identity"),
            "reason": reason,
            "discarded_at": utc_now(),
            "signal_sent": False,
        }
    )
    state["stale_worker_references"] = audit
    if clear:
        state["worker_pid"] = None
        state["worker_identity"] = None


_CHILD_TERM_GRACE_SECONDS = 5.0


def _signal_child_process_group(
    process: subprocess.Popen[Any], signum: int
) -> None:
    """Signal a child launched into its own session/process group."""

    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 1:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            return
        return
    process.send_signal(signum)


def _terminate_child_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = _CHILD_TERM_GRACE_SECONDS
) -> None:
    """Best-effort TERM/KILL/reap while preserving the caller's exception."""

    try:
        running = process.poll() is None
    except BaseException:
        running = True
    if running:
        try:
            _signal_child_process_group(process, signal.SIGTERM)
        except BaseException:
            pass
    try:
        process.wait(timeout=float(grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    except BaseException:
        pass
    try:
        _signal_child_process_group(process, signal.SIGKILL)
    except BaseException:
        pass
    try:
        process.wait()
    except BaseException:
        pass


def _worker_pid(spec: LoadedSpec, lane_id: str, state: Mapping[str, Any]) -> Any:
    pid = state.get("worker_pid")
    if pid:
        return pid
    pid_path = _lane_paths(spec, lane_id)["pid"]
    if pid_path.is_file():
        return read_json(pid_path).get("pid")
    return None


def _append_registry(spec: LoadedSpec, payload: Mapping[str, Any]) -> None:
    spec.registry_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), sort_keys=True)
    with file_lock(spec.registry_path.with_suffix(".lock")):
        with spec.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _verify_success_sentinel(stage: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    path = Path(str(stage["required_success_sentinel"]))
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise OrchestratorError(f"Required scientific success sentinel missing: {path}")
    payload = read_json(path)
    requirements = stage.get("required_success_fields") or {}
    if not isinstance(requirements, dict):
        raise OrchestratorError("required_success_fields must be a mapping")
    failures = {
        str(key): {"expected": expected, "actual": payload.get(str(key))}
        for key, expected in requirements.items()
        if payload.get(str(key)) != expected
    }
    if failures:
        raise OrchestratorError(f"Scientific success sentinel field mismatch: {failures}")
    return sha256_file(path), payload


def _heartbeat(spec: LoadedSpec, state: Mapping[str, Any], **extra: Any) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "backend": "autodl",
        "run_id": spec.run_id,
        "lane": state["lane"],
        "gpu_id": state["gpu_id"],
        "worker_pid": state.get("worker_pid"),
        "child_pid": state.get("child_pid"),
        "child_identity": state.get("child_identity"),
        "stage": state.get("current_stage"),
        "status": state["status"],
        "heartbeat_at": utc_now(),
        **extra,
    }
    atomic_write_json(_lane_paths(spec, str(state["lane"]))["heartbeat"], payload)


def _read_progress(stage: Mapping[str, Any]) -> dict[str, Any]:
    progress_path = str(stage.get("progress_json") or "")
    if not progress_path:
        return {}
    path = Path(progress_path)
    if not path.is_file():
        return {"progress_path": progress_path, "progress_available": False}
    try:
        payload = read_json(path)
    except (OSError, ValueError, OrchestratorError) as exc:
        return {
            "progress_path": progress_path,
            "progress_available": False,
            "progress_read_error": f"{type(exc).__name__}: {exc}",
        }
    allowed = (
        "current_step",
        "completed_step",
        "next_step",
        "total_steps",
        "steps_per_hour",
        "elapsed_seconds",
        "last_checkpoint_step",
        "latest_checkpoint",
        "heartbeat_at",
    )
    return {
        "progress_path": progress_path,
        "progress_available": True,
        **{key: payload.get(key) for key in allowed},
    }


def _input_manifest_digest(lane: Mapping[str, Any]) -> str:
    path = Path(str(lane["input_manifest"]))
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise OrchestratorError(f"Input manifest missing or empty: {path}")
    root = Path(str(lane["input_root"])).resolve()
    _verify_sha256_manifest(path, root, exact_inventory=True)
    return sha256_file(path)


def _verify_sha256_manifest(
    path: Path, root: Path, *, exact_inventory: bool = False
) -> None:
    """Stream-verify a standard SHA256 manifest without path escapes.

    The manifest digest alone cannot detect input bit rot.  This check runs at
    both stage boundaries, which also proves that preserved MUT/AIDS inputs did
    not change while recovery was reading them.
    """

    if path.is_symlink() or not path.is_file():
        raise OrchestratorError(f"SHA256 manifest is not a physical file: {path}")
    resolved_root = root.resolve()
    entries: set[Path] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        if len(raw_line) < 67 or raw_line[64:66] not in {"  ", " *"}:
            raise OrchestratorError(
                f"Malformed SHA256 manifest row {path}:{line_number}"
            )
        expected = raw_line[:64].lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise OrchestratorError(
                f"Invalid SHA256 digest at {path}:{line_number}"
            )
        relative_text = raw_line[66:]
        if relative_text.startswith("./"):
            relative_text = relative_text[2:]
        relative = Path(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise OrchestratorError(
                f"Unsafe SHA256 manifest path at {path}:{line_number}"
            )
        candidate = (resolved_root / relative).resolve(strict=False)
        if not _is_within(candidate, resolved_root):
            raise OrchestratorError(
                f"SHA256 manifest entry escapes input root: {relative_text}"
            )
        if not candidate.is_file() or candidate.is_symlink():
            raise OrchestratorError(
                f"SHA256 manifest entry is not a physical file: {candidate}"
            )
        actual = sha256_file(candidate)
        if actual != expected:
            raise OrchestratorError(
                f"SHA256 mismatch for manifest entry: {candidate}"
            )
        if relative in entries:
            raise OrchestratorError(
                f"Duplicate SHA256 manifest path at {path}:{line_number}"
            )
        entries.add(relative)
    if not entries:
        raise OrchestratorError(f"SHA256 manifest has no file entries: {path}")
    if exact_inventory:
        actual_files: set[Path] = set()
        for current, directories, files in os.walk(resolved_root):
            current_path = Path(current)
            for name in [*directories, *files]:
                candidate = current_path / name
                if candidate.is_symlink():
                    raise OrchestratorError(
                        f"Input root contains a symbolic link: {candidate}"
                    )
            for name in files:
                candidate = current_path / name
                if candidate.resolve(strict=False) == path.resolve(strict=False):
                    continue
                actual_files.add(candidate.relative_to(resolved_root))
        extras = sorted(actual_files - entries, key=lambda value: value.as_posix())
        missing = sorted(entries - actual_files, key=lambda value: value.as_posix())
        if extras or missing:
            raise OrchestratorError(
                "Input SHA256 manifest inventory mismatch: "
                f"unlisted={extras[:8]}, absent={missing[:8]}"
            )


def _verify_output_manifest(
    lane: Mapping[str, Any], stage: Mapping[str, Any]
) -> str:
    manifest_text = str(stage.get("output_manifest") or "")
    if not manifest_text:
        digest, _payload = _verify_success_sentinel(stage)
        return digest
    path = Path(manifest_text)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise OrchestratorError(f"Output manifest missing or non-physical: {path}")
    root = Path(
        str(stage.get("output_manifest_root") or path.parent)
    ).resolve(strict=False)
    output_root = Path(str(lane["output_root"])).resolve(strict=False)
    if not _is_within(path.resolve(strict=False), output_root):
        raise OrchestratorError(f"Output manifest escapes lane output root: {path}")
    if not _is_within(root, output_root):
        raise OrchestratorError(f"Output manifest root escapes lane output root: {root}")
    if path.suffix == ".sha256":
        _verify_sha256_manifest(path, root)
    return sha256_file(path)


def _run_completion_verifier(
    spec: LoadedSpec,
    lane: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> None:
    if spec.policy.get("require_formal_stage_verifier") is not True:
        return
    command = _completion_verifier_command(stage)
    lane_id = str(lane["id"])
    stage_id = str(stage["id"])
    environment = _sanitized_inherited_environment()
    environment.update(
        {str(key): str(value) for key, value in stage["environment"].items()}
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(lane["gpu_id"]),
            "AUTODL_BACKEND": "1",
            "THREE_LINES_RUN_ID": spec.run_id,
            "THREE_LINES_LANE": lane_id,
            "THREE_LINES_STAGE": stage_id,
            "THREE_LINES_INPUT_ROOT": str(lane["input_root"]),
            "THREE_LINES_OUTPUT_ROOT": str(lane["output_root"]),
            "THREE_LINES_CACHE_ROOT": str(lane["cache_root"]),
            "THREE_LINES_ACTIVE_ROOT": str(lane["active_root"]),
            "THREE_LINES_REQUIRED_SUCCESS_SENTINEL": str(
                stage["required_success_sentinel"]
            ),
        }
    )
    if str(lane["dataset"]).lower() in {"mutagenicity", "aids"}:
        environment["DISALLOW_GENERATION"] = "1"
    try:
        result = subprocess.run(
            command,
            cwd=str(spec.project_root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(
                spec.runtime.get("completion_verify_timeout_seconds", 21_600)
            ),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OrchestratorError(
            f"Formal completion verifier timed out: {lane_id}:{stage_id}"
        ) from exc
    if result.returncode != 0:
        raise OrchestratorError(
            f"Formal completion verifier rejected stage: {lane_id}:{stage_id} "
            f"(exit={result.returncode})"
        )


def _validate_completed_stage(
    spec: LoadedSpec,
    lane: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a persisted stage completion instead of trusting its name."""

    lane_id = str(lane["id"])
    stage_id = str(stage["id"])
    success_path = _stage_success_path(spec, lane_id, stage_id)
    if success_path.is_symlink() or not success_path.is_file():
        raise OrchestratorError(
            f"Orchestration success sentinel is missing or non-physical: {success_path}"
        )
    payload = read_json(success_path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "backend": "autodl",
        "run_id": spec.run_id,
        "lane": lane_id,
        "stage": stage_id,
        "gpu_id": int(lane["gpu_id"]),
        "required_success_sentinel": str(stage["required_success_sentinel"]),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise OrchestratorError(
            f"Stale or forged stage success sentinel {success_path}: {mismatches}"
        )
    input_digest = _input_manifest_digest(lane)
    if payload.get("input_manifest_digest") != input_digest:
        raise OrchestratorError(
            f"Completed stage input digest changed: {lane_id}:{stage_id}"
        )
    _run_completion_verifier(spec, lane, stage)
    scientific_digest, _scientific_payload = _verify_success_sentinel(stage)
    if payload.get("required_success_sentinel_digest") != scientific_digest:
        raise OrchestratorError(
            f"Completed stage scientific sentinel changed: {lane_id}:{stage_id}"
        )
    output_digest = _verify_output_manifest(lane, stage)
    if payload.get("output_manifest_digest") != output_digest:
        raise OrchestratorError(
            f"Completed stage output manifest changed: {lane_id}:{stage_id}"
        )
    return payload


def _validate_completed_lane(
    spec: LoadedSpec,
    lane: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    completions = [
        _validate_completed_stage(spec, lane, stage) for stage in lane["stages"]
    ]
    lane_success = _lane_paths(spec, str(lane["id"]))["lane_success"]
    if lane_success.is_symlink() or not lane_success.is_file():
        raise OrchestratorError(
            f"Succeeded lane lacks a physical lane sentinel: {lane_success}"
        )
    payload = read_json(lane_success)
    final_output_digest = completions[-1]["output_manifest_digest"]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "backend": "autodl",
        "run_id": spec.run_id,
        "lane": str(lane["id"]),
        "gpu_id": int(lane["gpu_id"]),
        "input_manifest_digest": _input_manifest_digest(lane),
        "output_manifest_digest": final_output_digest,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise OrchestratorError(
            f"Succeeded lane sentinel is stale {lane_success}: {mismatches}"
        )
    if state.get("output_manifest_digest") != final_output_digest:
        raise OrchestratorError(
            f"Succeeded lane state output digest is stale: {lane['id']}"
        )
    return payload


def _select_command(stage: Mapping[str, Any], retry_count: int) -> list[str]:
    if retry_count > 0 and stage.get("resume_command"):
        command = [str(value) for value in stage["resume_command"]]
    else:
        command = [str(value) for value in stage["command"]]
    if any(PLACEHOLDER.fullmatch(value) or "__CONFIGURE_" in value for value in command):
        raise OrchestratorError(
            f"Stage {stage['id']} command is not configured: {command}"
        )
    _assert_no_embedded_secrets(command)
    return command


def _stage_success_path(spec: LoadedSpec, lane_id: str, stage_id: str) -> Path:
    return _lane_paths(spec, lane_id)["sentinels"] / f"{stage_id}.SUCCESS.json"


def _stage_failure_path(spec: LoadedSpec, lane_id: str, stage_id: str) -> Path:
    return _lane_paths(spec, lane_id)["sentinels"] / f"{stage_id}.FAILED.json"


def _run_stage(
    spec: LoadedSpec,
    lane: Mapping[str, Any],
    stage: Mapping[str, Any],
    state: dict[str, Any],
    stop_requested: "StopFlag",
) -> None:
    lane_id = str(lane["id"])
    stage_id = str(stage["id"])
    paths = _lane_paths(spec, lane_id)
    command = _select_command(stage, int(state["retry_count"]))
    input_digest_before = _input_manifest_digest(lane)
    log_path = spec.persistent_root / "logs" / lane_id / f"{stage_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(lane["output_root"])).mkdir(parents=True, exist_ok=True)
    Path(str(lane["cache_root"])).mkdir(parents=True, exist_ok=True)
    Path(str(lane["active_root"])).mkdir(parents=True, exist_ok=True)
    environment = _sanitized_inherited_environment()
    environment.update({str(key): str(value) for key, value in stage["environment"].items()})
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(lane["gpu_id"]),
            "AUTODL_BACKEND": "1",
            "THREE_LINES_RUN_ID": spec.run_id,
            "THREE_LINES_LANE": lane_id,
            "THREE_LINES_STAGE": stage_id,
            "THREE_LINES_INPUT_ROOT": str(lane["input_root"]),
            "THREE_LINES_OUTPUT_ROOT": str(lane["output_root"]),
            "THREE_LINES_CACHE_ROOT": str(lane["cache_root"]),
            "THREE_LINES_ACTIVE_ROOT": str(lane["active_root"]),
            "THREE_LINES_REQUIRED_SUCCESS_SENTINEL": str(
                stage["required_success_sentinel"]
            ),
        }
    )
    if str(lane["dataset"]).lower() in {"mutagenicity", "aids"}:
        environment["DISALLOW_GENERATION"] = "1"
    started_at = utc_now()
    git_commit = _git_output(spec.project_root, "rev-parse", "HEAD")
    external_commit = _git_output(spec.external_root, "rev-parse", "HEAD")
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(spec.project_root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            child_identity = _capture_child_identity(
                process.pid, spec, lane_id, stage_id, command
            )
            state.update(
                {
                    "status": "RUNNING",
                    "child_pid": process.pid,
                    "child_identity": child_identity,
                    "current_stage": stage_id,
                    "current_command": _redacted_command(command),
                    "input_manifest_digest": input_digest_before,
                    "started_at": state.get("started_at") or started_at,
                    "failure": None,
                }
            )
            state["stages"][stage_id] = {
                "stage": stage_id,
                "status": "RUNNING",
                "attempt": int(state["retry_count"]) + 1,
                "pid": process.pid,
                "process_identity": child_identity,
                "command": _redacted_command(command),
                "started_at": started_at,
                "log": str(log_path),
            }
            _save_lane_state(spec, state)
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "backend": "autodl",
                "run_id": spec.run_id,
                "lane": lane_id,
                "gpu_id": int(lane["gpu_id"]),
                "pid": process.pid,
                "process_identity": child_identity,
                "command": _redacted_command(command),
                "dataset": lane["dataset"],
                "method": lane["method"],
                "stage": stage_id,
                "input_root": lane["input_root"],
                "output_root": lane["output_root"],
                "input_manifest_digest": input_digest_before,
                "git_commit": git_commit,
                "external_commit": external_commit,
                "started_at": started_at,
                "state": "RUNNING",
                "retry_count": state["retry_count"],
                "slurm_job_id": None,
            }
            provenance_path = paths["provenance"] / (
                f"{stage_id}.attempt-{int(state['retry_count']) + 1}.json"
            )
            atomic_write_json(provenance_path, provenance)
            _append_registry(spec, provenance)
            while process.poll() is None:
                if stop_requested.value or paths["stop"].is_file():
                    _terminate_child_process_group(
                        process,
                        grace_seconds=float(
                            spec.runtime.get("stop_grace_seconds", 30)
                        ),
                    )
                    raise InterruptedError("Stop requested")
                progress = _read_progress(stage)
                if progress.get("latest_checkpoint"):
                    state["latest_checkpoint"] = progress["latest_checkpoint"]
                if progress:
                    state["progress"] = progress
                    _save_lane_state(spec, state)
                _heartbeat(spec, state, log=str(log_path), progress=progress)
                time.sleep(float(spec.runtime.get("heartbeat_seconds", 10)))
            return_code = int(process.returncode or 0)
        except BaseException:
            _terminate_child_process_group(
                process,
                grace_seconds=float(spec.runtime.get("stop_grace_seconds", 30)),
            )
            raise
    if stop_requested.value or paths["stop"].is_file():
        raise InterruptedError("Stop requested")
    input_digest_after = _input_manifest_digest(lane)
    if input_digest_after != input_digest_before:
        raise OrchestratorError(
            f"Lane {lane_id} input manifest changed during {stage_id}"
        )
    if return_code != 0:
        raise OrchestratorError(f"Stage {lane_id}:{stage_id} exited {return_code}")
    _scientific_digest, scientific_payload = _verify_success_sentinel(stage)
    output_digest = _verify_output_manifest(lane, stage)
    completed_at = utc_now()
    orchestration_sentinel = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "backend": "autodl",
        "run_id": spec.run_id,
        "lane": lane_id,
        "stage": stage_id,
        "gpu_id": int(lane["gpu_id"]),
        "input_manifest_digest": input_digest_after,
        "output_manifest_digest": output_digest,
        "required_success_sentinel": str(stage["required_success_sentinel"]),
        "required_success_sentinel_digest": sha256_file(
            Path(str(stage["required_success_sentinel"]))
        ),
        "scientific_status": scientific_payload.get("status"),
        "finished_at": completed_at,
    }
    atomic_write_json(_stage_success_path(spec, lane_id, stage_id), orchestration_sentinel)
    state["stages"][stage_id].update(
        {
            "status": "SUCCEEDED",
            "finished_at": completed_at,
            "return_code": 0,
            "output_manifest_digest": output_digest,
            "sentinel": str(_stage_success_path(spec, lane_id, stage_id)),
        }
    )
    state["child_pid"] = None
    state["child_identity"] = None
    state["output_manifest_digest"] = output_digest
    _save_lane_state(spec, state)
    provenance.update(
        {
            "state": "SUCCEEDED",
            "end_time": completed_at,
            "output_manifest_digest": output_digest,
            "final_gate": "PASS",
        }
    )
    atomic_write_json(provenance_path, provenance)
    _append_registry(spec, provenance)


class StopFlag:
    value = False


def _worker(spec: LoadedSpec, lane_id: str) -> int:
    if lane_id not in spec.lane_by_id:
        raise OrchestratorError(f"Unknown lane: {lane_id}")
    _load_bound_run_state(spec)
    lane = spec.lane_by_id[lane_id]
    paths = _lane_paths(spec, lane_id)
    stop_flag = StopFlag()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_flag.value = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    startup_deadline = time.monotonic() + float(
        spec.runtime.get("worker_start_timeout_seconds", 10)
    )
    while True:
        if paths["pid"].is_file():
            worker_record = read_json(paths["pid"])
            recorded_pid = worker_record.get("pid")
            if recorded_pid == os.getpid() and _pid_matches_worker(
                os.getpid(), spec, lane_id
            ):
                break
        if time.monotonic() >= startup_deadline:
            raise OrchestratorError(
                f"Controller did not publish PID {os.getpid()} for lane {lane_id}"
            )
        time.sleep(0.01)
    with file_lock(paths["lock"], blocking=False):
        state = _load_lane_state(spec, lane)
        worker_identity = worker_record.get("worker_identity")
        if (
            state.get("worker_pid") != os.getpid()
            or state.get("worker_identity") != worker_identity
            or not _worker_identity_matches(
                worker_identity,
                spec,
                lane_id,
                _worker_command(spec, lane_id),
            )
        ):
            raise OrchestratorError(
                f"Controller worker identity publication mismatch for {lane_id}"
            )
        state["worker_pid"] = os.getpid()
        state["worker_identity"] = worker_identity
        state["status"] = "RUNNING"
        state["started_at"] = state.get("started_at") or utc_now()
        state["git_commit"] = _git_output(spec.project_root, "rev-parse", "HEAD")
        state["external_commit"] = _git_output(
            spec.external_root, "rev-parse", "HEAD"
        )
        state["input_manifest_digest"] = _input_manifest_digest(lane)
        _save_lane_state(spec, state)
        worker_record.update(
            {
                "status": "RUNNING",
                "worker_identity": worker_identity,
                "worker_started_at": utc_now(),
            }
        )
        atomic_write_json(paths["pid"], worker_record)
        try:
            for stage in lane["stages"]:
                stage_id = str(stage["id"])
                if _stage_success_path(spec, lane_id, stage_id).is_file():
                    completion = _validate_completed_stage(spec, lane, stage)
                    state["stages"].setdefault(stage_id, {}).update(
                        {
                            "stage": stage_id,
                            "status": "SUCCEEDED",
                            "sentinel": str(
                                _stage_success_path(spec, lane_id, stage_id)
                            ),
                            "output_manifest_digest": completion[
                                "output_manifest_digest"
                            ],
                        }
                    )
                    state["output_manifest_digest"] = completion[
                        "output_manifest_digest"
                    ]
                    _save_lane_state(spec, state)
                    continue
                while True:
                    satisfied, missing = _dependencies_satisfied(
                        spec, lane_id, stage
                    )
                    state["dependency_status"] = {
                        value: value not in missing
                        for value in stage.get("dependencies") or []
                    }
                    if satisfied:
                        break
                    if stop_flag.value or paths["stop"].is_file():
                        raise InterruptedError("Stop requested while waiting")
                    state.update(
                        {
                            "status": "WAITING_DEPENDENCY",
                            "current_stage": stage_id,
                            "child_pid": None,
                            "child_identity": None,
                        }
                    )
                    _save_lane_state(spec, state)
                    _heartbeat(spec, state, missing_dependencies=missing)
                    _refresh_run_state(spec)
                    time.sleep(float(spec.runtime.get("dependency_poll_seconds", 10)))
                state["current_stage"] = stage_id
                _save_lane_state(spec, state)
                _run_stage(spec, lane, stage, state, stop_flag)
            state.update(
                {
                    "status": "SUCCEEDED",
                    "current_stage": None,
                    "current_command": [],
                    "child_pid": None,
                    "child_identity": None,
                    "finished_at": utc_now(),
                }
            )
            # Publish the physical lane sentinel before the top-level
            # SUCCEEDED state. Cross-lane readers treat the old RUNNING state
            # as WAITING; they must never observe SUCCEEDED without its
            # corresponding durable lane sentinel.
            atomic_write_json(
                paths["lane_success"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "PASS",
                    "backend": "autodl",
                    "run_id": spec.run_id,
                    "lane": lane_id,
                    "gpu_id": int(lane["gpu_id"]),
                    "input_manifest_digest": state["input_manifest_digest"],
                    "output_manifest_digest": state["output_manifest_digest"],
                    "finished_at": state["finished_at"],
                },
            )
            if paths["lane_failure"].exists():
                paths["lane_failure"].unlink()
                fsync_directory(paths["lane_failure"].parent)
            _save_lane_state(spec, state)
            _heartbeat(spec, state)
            return 0
        except InterruptedError as exc:
            state.update(
                {
                    "status": "STOPPED",
                    "child_pid": None,
                    "child_identity": None,
                    "finished_at": utc_now(),
                    "failure": str(exc),
                }
            )
            _save_lane_state(spec, state)
            _heartbeat(spec, state)
            return 143
        except Exception as exc:
            stage_id = str(state.get("current_stage") or "unknown")
            failure = {
                "schema_version": SCHEMA_VERSION,
                "status": "FAIL",
                "backend": "autodl",
                "run_id": spec.run_id,
                "lane": lane_id,
                "stage": stage_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": utc_now(),
            }
            atomic_write_json(_stage_failure_path(spec, lane_id, stage_id), failure)
            atomic_write_json(paths["lane_failure"], failure)
            state.update(
                {
                    "status": "FAILED",
                    "child_pid": None,
                    "child_identity": None,
                    "finished_at": utc_now(),
                    "failure": failure,
                }
            )
            if stage_id in state["stages"]:
                state["stages"][stage_id].update(
                    {"status": "FAILED", "finished_at": utc_now(), "failure": failure}
                )
            _save_lane_state(spec, state)
            _heartbeat(spec, state)
            return 1
        finally:
            pid_payload = read_json(paths["pid"])
            pid_payload.update({"status": state["status"], "exited_at": utc_now()})
            atomic_write_json(paths["pid"], pid_payload)
            _refresh_run_state(spec)


def _check_read_only_tree(root: Path) -> None:
    for current_root, directories, files in os.walk(root):
        current = Path(current_root)
        values = [
            current,
            *(current / name for name in directories),
            *(current / name for name in files),
        ]
        for value in values:
            try:
                stat_result = value.lstat()
            except FileNotFoundError as exc:
                raise OrchestratorError(f"Input changed during read-only scan: {value}") from exc
            if value.is_symlink():
                target = value.resolve(strict=True)
                if not _is_within(target, root.resolve()):
                    raise OrchestratorError(
                        f"Input snapshot contains an external symlink: {value} -> {target}"
                    )
                continue
            if stat_result.st_mode & 0o222:
                raise OrchestratorError(f"Input snapshot has writable mode bits: {value}")


def _preflight(
    spec: LoadedSpec, lanes: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    if not spec.project_root.is_dir():
        raise OrchestratorError(f"Project root does not exist: {spec.project_root}")
    code_commit = _git_output(spec.project_root, "rev-parse", "HEAD")
    branch = _git_output(spec.project_root, "branch", "--show-current")
    expected_branch = str(spec.provenance.get("branch") or "")
    if expected_branch and branch != expected_branch:
        raise OrchestratorError(f"Expected branch {expected_branch}, found {branch}")
    requested_commit = str(spec.provenance.get("code_commit") or "HEAD")
    if requested_commit not in {"HEAD", code_commit}:
        resolved = _git_output(spec.project_root, "rev-parse", requested_commit)
        if resolved != code_commit:
            raise OrchestratorError(
                f"Checked-out commit {code_commit} does not match {requested_commit}"
            )
    base_commit = str(spec.provenance.get("base_commit") or "")
    if base_commit and not _git_is_ancestor(spec.project_root, base_commit, code_commit):
        raise OrchestratorError(
            f"Required base commit {base_commit} is not an ancestor of {code_commit}"
        )
    code_status = _git_output(
        spec.project_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if bool(spec.policy.get("require_clean_code", True)) and code_status:
        raise OrchestratorError("Repair worktree must be clean before formal start")
    external_commit = _git_output(spec.external_root, "rev-parse", "HEAD")
    expected_external = str(spec.provenance.get("external_comrecgc_commit") or "")
    if external_commit != expected_external:
        raise OrchestratorError(
            f"External COMRECGC commit mismatch: {external_commit} != {expected_external}"
        )
    external_lineage = _external_worktree_lineage(spec.external_root)
    validated_lanes = list(spec.lanes if lanes is None else lanes)
    for lane in validated_lanes:
        input_root = Path(str(lane["input_root"]))
        if not input_root.is_dir():
            raise OrchestratorError(f"Input root does not exist: {input_root}")
        _input_manifest_digest(lane)
        if bool(spec.policy.get("require_input_read_only", True)):
            _check_read_only_tree(input_root)
        for stage in lane["stages"]:
            _select_command(stage, 0)
    required_gpu_count = int(spec.runtime.get("require_gpu_count", 4))
    if bool(spec.runtime.get("require_nvidia_smi", True)):
        result = subprocess.run(
            ["nvidia-smi", "-L"], text=True, capture_output=True, check=False
        )
        if result.returncode != 0:
            raise OrchestratorError(f"nvidia-smi -L failed: {result.stderr.strip()}")
        visible = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
        if len(visible) != required_gpu_count:
            raise OrchestratorError(
                f"Expected {required_gpu_count} visible GPUs, found {len(visible)}"
            )
    return {
        "status": "PASS",
        "checked_at": utc_now(),
        "code_commit": code_commit,
        "branch": branch,
        "worktree_clean": not bool(code_status),
        "external_comrecgc_commit": external_commit,
        "external_code_clean": True,
        **external_lineage,
        "gpu_count": required_gpu_count,
        "validated_lanes": [str(lane["id"]) for lane in validated_lanes],
    }


def _assert_fresh_managed_roots(
    spec: LoadedSpec, lanes: Sequence[Mapping[str, Any]] | None = None
) -> None:
    """Refuse to adopt untracked output/cache data on a first ``start``."""

    examined: set[Path] = set()
    for lane in spec.lanes if lanes is None else lanes:
        for key in ("output_root", "cache_root", "active_root"):
            path = Path(str(lane[key])).resolve(strict=False)
            if path in examined:
                continue
            examined.add(path)
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise OrchestratorError(
                    f"Managed {key} is not empty but no run state exists: {path}"
                )


def _launch_worker(spec: LoadedSpec, lane: Mapping[str, Any], *, retry: bool) -> int:
    lane_id = str(lane["id"])
    paths = _lane_paths(spec, lane_id)
    state = _load_lane_state(spec, lane)
    if paths["pid"].is_symlink():
        raise OrchestratorError(f"Worker PID record must be physical: {paths['pid']}")
    previous_worker_record = (
        read_json(paths["pid"]) if paths["pid"].is_file() else None
    )
    existing_worker = _worker_pid(spec, lane_id, state)
    if existing_worker and _pid_matches_worker(int(existing_worker), spec, lane_id):
        return int(existing_worker)
    if existing_worker:
        if _pid_alive(existing_worker):
            _audit_stale_worker_reference(
                state,
                record=previous_worker_record,
                reason=(
                    "recorded worker PID is live but its exact starttime, "
                    "cmdline, PGID, run, lane, or command identity mismatches"
                ),
                clear=False,
            )
            _save_lane_state(spec, state)
            raise OrchestratorError(
                f"Lane {lane_id} has a live but non-matching worker PID "
                f"{existing_worker}; refusing a second writer"
            )
        _audit_stale_worker_reference(
            state,
            record=previous_worker_record,
            reason="recorded worker process is no longer alive",
            clear=True,
        )
    child_pid = state.get("child_pid")
    if child_pid:
        if _state_child_matches(spec, lane_id, state):
            state["status"] = "ORPHANED_CHILD"
            state["failure"] = {
                "error": (
                    "Worker exited while its exactly identified child is still "
                    "alive; refusing a second writer"
                ),
                "child_pid": child_pid,
                "child_identity": state.get("child_identity"),
            }
            _save_lane_state(spec, state)
            raise OrchestratorError(
                f"Lane {lane_id} has exact orphan child PID {child_pid}; "
                "run stop before resume"
            )
        if _pid_alive(child_pid):
            _discard_stale_child_reference(
                state,
                reason=(
                    "recorded PID is live but its starttime/cmdline/pgid or "
                    "run/lane/stage identity no longer matches"
                ),
            )
        else:
            _discard_stale_child_reference(
                state, reason="recorded child process is no longer alive"
            )
    if retry:
        state["retry_count"] = int(state.get("retry_count") or 0) + 1
    if paths["pid"].is_file():
        previous_pid = read_json(paths["pid"])
        atomic_write_json(
            paths["provenance"]
            / f"worker_pid.before-attempt-{int(state.get('retry_count') or 0) + 1}.json",
            previous_pid,
        )
        paths["pid"].unlink()
        fsync_directory(paths["pid"].parent)
    state.update(
        {
            "status": "STARTING",
            "worker_pid": None,
            "worker_identity": None,
            "child_pid": None,
            "child_identity": None,
            "failure": None,
            "finished_at": None,
        }
    )
    if paths["stop"].exists():
        paths["stop"].unlink()
    _save_lane_state(spec, state)
    supervisor_log = spec.persistent_root / "logs" / lane_id / "supervisor.log"
    supervisor_log.parent.mkdir(parents=True, exist_ok=True)
    argv = _worker_command(spec, lane_id)
    with supervisor_log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            argv,
            cwd=str(spec.project_root),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_sanitized_inherited_environment(),
        )
    worker_identity = _capture_worker_identity(
        process.pid, spec, lane_id, argv
    )
    if worker_identity.get("capture_status") != "CAPTURED":
        try:
            process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.wait(timeout=float(spec.runtime.get("stop_grace_seconds", 30)))
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            process.wait()
        state.update(
            {
                "status": "FAILED",
                "worker_pid": None,
                "worker_identity": None,
                "finished_at": utc_now(),
                "failure": {
                    "error": "Unable to capture a safe worker process identity",
                    "worker_identity": worker_identity,
                },
            }
        )
        _save_lane_state(spec, state)
        raise OrchestratorError(
            f"Unable to capture a safe worker identity for lane {lane_id}"
        )
    state["worker_pid"] = process.pid
    state["worker_identity"] = worker_identity
    _save_lane_state(spec, state)
    atomic_write_json(
        paths["pid"],
        {
            "schema_version": SCHEMA_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "backend": "autodl",
            "kind": "autodl_worker_pid",
            "run_id": spec.run_id,
            "lane": lane_id,
            "spec_sha256": spec.spec_sha256,
            "pid": process.pid,
            "worker_identity": worker_identity,
            "started_at": utc_now(),
            "status": "STARTING",
            "slurm_job_id": None,
        },
    )
    return process.pid


def start(
    spec: LoadedSpec, lane_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    selected_lanes = _selected_lanes(spec, lane_ids)
    selected_lane_ids = [str(lane["id"]) for lane in selected_lanes]
    if spec.run_state_path.exists():
        existing = read_json(spec.run_state_path)
        raise OrchestratorError(
            f"Run state already exists with status={existing.get('status')}; use resume"
        )
    preflight = _preflight(spec, selected_lanes)
    spec.state_root.mkdir(parents=True, exist_ok=True)
    with file_lock(spec.global_lock_path):
        if spec.run_state_path.exists():
            existing = read_json(spec.run_state_path)
            raise OrchestratorError(
                f"Run state already exists with status={existing.get('status')}; use resume"
            )
        stale_state_entries = [
            value
            for value in spec.state_root.iterdir()
            if value.name != spec.global_lock_path.name
        ]
        if stale_state_entries:
            raise OrchestratorError(
                "State root contains artifacts without run_state.json; refusing "
                f"implicit adoption: {stale_state_entries}"
            )
        _assert_fresh_managed_roots(spec, selected_lanes)
        for lane in spec.lanes:
            paths = _lane_paths(spec, str(lane["id"]))
            paths["sentinels"].mkdir(parents=True, exist_ok=True)
            paths["provenance"].mkdir(parents=True, exist_ok=True)
            _save_lane_state(spec, _initial_lane_state(spec, lane))
        atomic_write_json(spec.state_root / "preflight.json", preflight)
        atomic_write_json(
            spec.run_state_path,
            {
                **_spec_state_binding(spec),
                "status": "NOT_STARTED",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "project_root": str(spec.project_root),
                "persistent_root": str(spec.persistent_root),
                "fast_root": str(spec.fast_root),
                "external_root": str(spec.external_root),
                "commands": _canonical_commands(spec),
                "lanes": {},
                "slurm_jobs": [],
                "autodl_pid_is_slurm_job_id": False,
            },
        )
        pids = {
            str(lane["id"]): _launch_worker(spec, lane, retry=False)
            for lane in selected_lanes
        }
    payload = _refresh_run_state(spec)
    payload["launched_pids"] = pids
    payload["requested_lanes"] = selected_lane_ids
    payload["preflight_validated_lanes"] = preflight["validated_lanes"]
    return payload


def resume(
    spec: LoadedSpec, lane_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    if not spec.run_state_path.is_file():
        raise OrchestratorError("No persisted run exists; use start")
    _load_bound_run_state(spec)
    selected_lanes = _selected_lanes(spec, lane_ids)
    selected_lane_ids = [str(lane["id"]) for lane in selected_lanes]
    preflight = _preflight(spec, selected_lanes)
    launched: dict[str, int] = {}
    skipped: dict[str, str] = {}
    with file_lock(spec.global_lock_path):
        _load_bound_run_state(spec)
        states = {
            str(lane["id"]): _load_lane_state(spec, lane)
            for lane in selected_lanes
        }
        never_started = [
            lane
            for lane in selected_lanes
            if _lane_was_never_started(states[str(lane["id"])])
        ]
        _assert_fresh_managed_roots(spec, never_started)
        for lane in selected_lanes:
            lane_id = str(lane["id"])
            state = states[lane_id]
            state = _load_lane_state(spec, lane)
            if state["status"] == "SUCCEEDED":
                _validate_completed_lane(spec, lane, state)
                skipped[lane_id] = "ALREADY_SUCCEEDED"
                continue
            pid = _worker_pid(spec, lane_id, state)
            if pid and _pid_matches_worker(int(pid), spec, lane_id):
                skipped[lane_id] = "ALREADY_ACTIVE"
                continue
            launched[lane_id] = _launch_worker(
                spec,
                lane,
                retry=not _lane_was_never_started(state),
            )
    payload = _refresh_run_state(spec)
    payload["resumed_pids"] = launched
    payload["skipped_lanes"] = skipped
    payload["requested_lanes"] = selected_lane_ids
    payload["preflight_validated_lanes"] = preflight["validated_lanes"]
    return payload


def stop(spec: LoadedSpec) -> dict[str, Any]:
    if not spec.run_state_path.is_file():
        raise OrchestratorError("No persisted run exists")
    _load_bound_run_state(spec)
    stopped: dict[str, Any] = {}
    with file_lock(spec.global_lock_path):
        _load_bound_run_state(spec)
        for lane in spec.lanes:
            lane_id = str(lane["id"])
            state = _load_lane_state(spec, lane)
            paths = _lane_paths(spec, lane_id)
            atomic_write_json(
                paths["stop"], {"requested_at": utc_now(), "lane": lane_id}
            )
            pid = _worker_pid(spec, lane_id, state)
            if pid and _pid_matches_worker(int(pid), spec, lane_id):
                # Re-read the full kernel/starttime/cmdline/PGID identity
                # immediately before signalling. Linux additionally signals a
                # pidfd, so PID reuse after validation cannot retarget kill.
                if not _signal_exact_worker(
                    int(pid), spec, lane_id, int(signal.SIGTERM)
                ):
                    record = (
                        read_json(paths["pid"])
                        if paths["pid"].is_file() and not paths["pid"].is_symlink()
                        else None
                    )
                    _audit_stale_worker_reference(
                        state,
                        record=record,
                        reason="worker identity changed during stop revalidation",
                        clear=False,
                    )
                    _save_lane_state(spec, state)
                    stopped[lane_id] = {
                        "stale_worker_pid": pid,
                        "signal": None,
                        "target": "none_identity_mismatch",
                    }
                    continue
                # Signal only the validated worker. Its in-memory Popen handle
                # owns the exact child and performs the child termination after
                # observing the stop flag. This avoids a stale child PID or PGID
                # widening the signal target.
                stopped[lane_id] = {
                    "worker_pid": pid,
                    "signal": "SIGTERM",
                    "target": "validated_worker_only",
                }
                state["status"] = "STOPPED"
                state["finished_at"] = utc_now()
                state["failure"] = "Operator stop requested"
                _save_lane_state(spec, state)
            else:
                if pid and _pid_alive(pid):
                    record = (
                        read_json(paths["pid"])
                        if paths["pid"].is_file() and not paths["pid"].is_symlink()
                        else None
                    )
                    _audit_stale_worker_reference(
                        state,
                        record=record,
                        reason=(
                            "recorded worker PID is live but exact identity "
                            "does not match"
                        ),
                        clear=False,
                    )
                    _save_lane_state(spec, state)
                    stopped[lane_id] = {
                        "stale_worker_pid": pid,
                        "signal": None,
                        "target": "none_identity_mismatch",
                    }
                    continue
                child = state.get("child_pid")
                if child and _state_child_matches(spec, lane_id, state):
                    child_identity = dict(state["child_identity"])
                    child_pgid = int(child_identity["pgid"])
                    # Revalidate once immediately before signalling to narrow
                    # the PID/PGID reuse window. Never derive the target from a
                    # merely live integer PID.
                    if not _state_child_matches(spec, lane_id, state):
                        _discard_stale_child_reference(
                            state,
                            reason="child identity changed during stop revalidation",
                        )
                        stopped[lane_id] = {
                            "stale_child_pid": child,
                            "signal": None,
                            "target": "none_identity_mismatch",
                        }
                        _save_lane_state(spec, state)
                        continue
                    os.killpg(child_pgid, signal.SIGTERM)
                    stopped[lane_id] = {
                        "orphan_child_pid": child,
                        "signal": "SIGTERM",
                        "target": "exact_child_process_group",
                        "pgid": child_pgid,
                    }
                    state["status"] = "STOPPED"
                    state["finished_at"] = utc_now()
                    state["failure"] = (
                        "Operator stop requested for exactly identified orphan child"
                    )
                    _save_lane_state(spec, state)
                elif child:
                    live = _pid_alive(child)
                    _discard_stale_child_reference(
                        state,
                        reason=(
                            "recorded PID is live but exact child identity mismatches"
                            if live
                            else "recorded child process is no longer alive"
                        ),
                    )
                    stopped[lane_id] = {
                        "stale_child_pid": child,
                        "signal": None,
                        "target": "none_identity_mismatch",
                    }
                    _save_lane_state(spec, state)
                else:
                    stopped[lane_id] = {"worker_pid": pid, "signal": None}
    payload = _refresh_run_state(spec)
    payload["stop_requests"] = stopped
    return payload


def status(spec: LoadedSpec) -> dict[str, Any]:
    if not spec.run_state_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": "autodl",
            "run_id": spec.run_id,
            "status": "NOT_STARTED",
            "commands": _canonical_commands(spec),
            "lanes": {},
            "active_lanes": [],
            "not_started_lanes": sorted(spec.lane_by_id),
            "succeeded_lanes": [],
        }
    payload = _load_bound_run_state(spec)
    lane_details: dict[str, Any] = {}
    for lane in spec.lanes:
        lane_id = str(lane["id"])
        state = _load_lane_state(spec, lane)
        completion_validation: dict[str, Any] = {"verified": False}
        if state["status"] == "SUCCEEDED":
            try:
                _validate_completed_lane(spec, lane, state)
                completion_validation = {"verified": True}
            except Exception as exc:
                state = dict(state)
                state["status"] = "STALE_COMPLETION"
                completion_validation = {
                    "verified": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        worker_pid = _worker_pid(spec, lane_id, state)
        child_pid = state.get("child_pid")
        lane_details[lane_id] = {
            **state,
            "worker_alive": bool(
                worker_pid and _pid_matches_worker(int(worker_pid), spec, lane_id)
            ),
            "child_alive": bool(child_pid and _state_child_matches(spec, lane_id, state)),
            "child_pid_live_but_identity_mismatch": bool(
                child_pid
                and _pid_alive(child_pid)
                and not _state_child_matches(spec, lane_id, state)
            ),
            "heartbeat_path": str(_lane_paths(spec, lane_id)["heartbeat"]),
            "success_sentinels": {
                str(stage["id"]): _stage_success_path(
                    spec, lane_id, str(stage["id"])
                ).is_file()
                for stage in lane["stages"]
            },
            "completion_validation": completion_validation,
        }
    payload["lanes"] = lane_details
    statuses = [str(value["status"]) for value in lane_details.values()]
    payload["status"] = _aggregate_run_status(statuses)
    payload.update(_lane_summary_sets(lane_details))
    payload["commands"] = _canonical_commands(spec)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "resume", "stop", "_worker"))
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("ops/specs/autodl_three_lines_20260821.yaml"),
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        metavar="LANE_ID",
        help=(
            "start/resume only; repeat to activate selected lanes incrementally "
            "(omitting launches all lanes)"
        ),
    )
    return parser


def _validate_paired_wrapper_arguments(args: argparse.Namespace) -> None:
    """Accept only the repository-mandated, behavior-neutral wrapper parity.

    ``scripts/slurm/run_three_lines.sh`` is a read-only status wrapper. AGENTS.md
    requires its paired Python invocation to carry the HPC config (and permits
    the standard inference fallback override), but neither option may mutate
    this AutoDL controller's scientific configuration.
    """

    if args.config is not None and Path(str(args.config)) != Path("configs/hpc.yaml"):
        raise OrchestratorError(
            "Controller --config is wrapper parity only and must be configs/hpc.yaml"
        )
    allowed_overrides = {"inference.fallback_to_heuristic=false"}
    rejected = [str(value) for value in args.set if str(value) not in allowed_overrides]
    if rejected:
        raise OrchestratorError(
            "Controller --set is wrapper parity only; unsupported override(s): "
            + ", ".join(rejected)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_paired_wrapper_arguments(args)
        spec = LoadedSpec.load(args.spec)
        if args.action == "_worker":
            if len(args.lane) != 1:
                raise OrchestratorError("Internal worker requires exactly one --lane")
            return _worker(spec, args.lane[0])
        if args.lane and args.action not in {"start", "resume"}:
            raise OrchestratorError("Public --lane is supported only by start/resume")
        if args.action == "start":
            payload = start(spec, args.lane)
        elif args.action == "resume":
            payload = resume(spec, args.lane)
        elif args.action == "status":
            payload = status(spec)
        else:
            payload = stop(spec)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "status": "BLOCKED",
            "action": args.action,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "at": utc_now(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
