"""One-shot Mut post-A/B successor contract and execution primitives.

The existing bounded A/B owner remains authoritative for trace-on, trace-off,
the stepwise comparator, and reload parity.  This module begins at that owner's
sealed ``next_action.json`` and can select only one of two lanes:

* exact equivalence -> historical adoption -> standardized evaluation -> publish;
* confirmed scientific divergence -> fresh trace-off Route B.

Operational or verifier failures never become Route-B evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .autodl_mut_first_divergence_v1 import file_sha256, stable_sha256
from .autodl_mut_post_ab_continuation_v1 import (
    DECISION_SCHEMA,
    MutPostABError,
    validate_ab_owner_terminal,
    validate_same_contract_gate,
)


SPEC_SCHEMA = "mut_next_stage_executor_task_spec_v1"
TERMINAL_SCHEMA = "mut_next_stage_executor_terminal_v1"
CONSUMPTION_SCHEMA = "mut_next_action_consumption_v1"
ADOPTION_STAGES = (
    "HISTORICAL_50K_ADOPTION",
    "STANDARDIZED_EVALUATION",
    "FIGURE_TABLE_EXPORT",
    "MATRIX_PUBLISH",
)
ROUTE_B_STAGES = ("ROUTE_B",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class MutNextStageError(RuntimeError):
    """The successor plan or one-shot consumption is unsafe."""


def _absolute(value: Any, *, field: str, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise MutNextStageError(f"{field} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise MutNextStageError(f"{field} must be an absolute non-symlink path")
    try:
        return path.resolve(strict=must_exist)
    except OSError as exc:
        raise MutNextStageError(f"{field} is absent: {path}") from exc


def _json(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise MutNextStageError(f"{field} is absent or indirect: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutNextStageError(f"{field} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MutNextStageError(f"{field} must be one JSON object")
    return value


def _executable(value: Any, *, field: str, must_exist: bool) -> Path:
    """Normalize an absolute interpreter symlink to its physical target."""

    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise MutNextStageError(f"{field} must be an absolute executable path")
    try:
        target = Path(value).resolve(strict=must_exist)
    except OSError as exc:
        raise MutNextStageError(f"{field} is absent: {value}") from exc
    if must_exist and not target.is_file():
        raise MutNextStageError(f"{field} is not a file: {target}")
    return target


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MutNextStageError(f"output may not be a symlink: {path}")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_stage(raw: Mapping[str, Any], *, check_files: bool) -> dict[str, Any]:
    required = {
        "stage",
        "argv",
        "cwd",
        "environment",
        "expected_terminal",
        "expected_terminal_status",
        "output_root",
        "pair_store_recomputed",
        "dbscan_recomputed",
        "calibration_used_for_selection",
        "test_used_for_selection",
    }
    if set(raw) not in (required, required | {"argv_sha256"}):
        raise MutNextStageError(f"stage keys changed: {raw.get('stage')}")
    row = dict(raw)
    stage = str(row.get("stage") or "")
    if stage not in set(ADOPTION_STAGES) | set(ROUTE_B_STAGES):
        raise MutNextStageError(f"unsupported successor stage: {stage}")
    argv = row.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise MutNextStageError(f"{stage}.argv is invalid")
    executable = _executable(
        argv[0], field=f"{stage}.argv[0]", must_exist=check_files
    )
    row["argv"] = [str(executable), *argv[1:]]
    row["cwd"] = str(_absolute(row["cwd"], field=f"{stage}.cwd", must_exist=check_files))
    row["expected_terminal"] = str(
        _absolute(row["expected_terminal"], field=f"{stage}.expected_terminal")
    )
    row["output_root"] = str(_absolute(row["output_root"], field=f"{stage}.output_root"))
    statuses = row.get("expected_terminal_status")
    if (
        not isinstance(statuses, list)
        or not statuses
        or any(not isinstance(item, str) or not item for item in statuses)
    ):
        raise MutNextStageError(f"{stage}.expected_terminal_status is invalid")
    environment = row.get("environment")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise MutNextStageError(f"{stage}.environment is invalid")
    if any(key.startswith("MUT_NEXT_ACTION_") for key in environment):
        raise MutNextStageError(
            f"{stage}.environment may not spoof executor next-action bindings"
        )
    row["environment"] = dict(sorted(environment.items()))
    for flag in (
        "pair_store_recomputed",
        "dbscan_recomputed",
        "calibration_used_for_selection",
        "test_used_for_selection",
    ):
        if row.get(flag) not in (True, False):
            raise MutNextStageError(f"{stage}.{flag} must be boolean")
    if stage in ADOPTION_STAGES and (
        row["pair_store_recomputed"]
        or row["dbscan_recomputed"]
        or row["calibration_used_for_selection"]
        or row["test_used_for_selection"]
    ):
        raise MutNextStageError(f"{stage} violates the adoption/no-leakage contract")
    argv_sha = stable_sha256(row["argv"])
    if raw.get("argv_sha256") not in (None, argv_sha):
        raise MutNextStageError(f"{stage}.argv binding changed")
    row["argv_sha256"] = argv_sha
    return row


def build_successor_spec(
    template: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    required = {
        "task_id",
        "execution_commit",
        "predecessor_task_id",
        "predecessor_task_spec",
        "predecessor_terminal",
        "next_action_path",
        "runtime_root",
        "lease_path",
        "publisher_id",
        "publisher_locator",
        "matrix_authority_root",
        "adoption_pipeline",
        "route_b_pipeline",
    }
    if set(template) != required:
        raise MutNextStageError(
            f"successor template keys changed: missing={required - set(template)}, "
            f"extra={set(template) - required}"
        )
    value = dict(template)
    for field in ("task_id", "predecessor_task_id", "publisher_id"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise MutNextStageError(f"{field} is empty")
    commit = value.get("execution_commit")
    if not isinstance(commit, str) or _GIT_SHA.fullmatch(commit) is None:
        raise MutNextStageError("execution_commit must be a full Git SHA")
    paths = {
        field: _absolute(
            value[field],
            field=field,
            must_exist=check_files and field in {"predecessor_task_spec", "matrix_authority_root"},
        )
        for field in (
            "predecessor_task_spec",
            "predecessor_terminal",
            "next_action_path",
            "runtime_root",
            "lease_path",
            "publisher_locator",
            "matrix_authority_root",
        )
    }
    if check_files:
        from .autodl_mut_same_contract_ab_v1 import validate_same_contract_ab_spec

        validate_same_contract_ab_spec(
            _json(paths["predecessor_task_spec"], field="predecessor task spec"),
            check_files=True,
        )
    if paths["predecessor_terminal"].parent == paths["next_action_path"].parent:
        # They normally live in separate owner/continuation roots.  Sharing a
        # directory makes accidental rename of the terminal too easy.
        raise MutNextStageError("predecessor terminal and next action must be isolated")
    adoption = [
        _normalize_stage(row, check_files=check_files)
        for row in value["adoption_pipeline"]
    ]
    route_b = [
        _normalize_stage(row, check_files=check_files)
        for row in value["route_b_pipeline"]
    ]
    if tuple(row["stage"] for row in adoption) != ADOPTION_STAGES:
        raise MutNextStageError("adoption pipeline order changed")
    if tuple(row["stage"] for row in route_b) != ROUTE_B_STAGES:
        raise MutNextStageError("Route-B pipeline order changed")
    roots = [row["output_root"] for row in adoption + route_b]
    if len(roots) != len(set(roots)):
        raise MutNextStageError("successor stages share an output root")
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA,
        **{key: value[key] for key in required if key not in {"adoption_pipeline", "route_b_pipeline"}},
        **{field: str(path) for field, path in paths.items()},
        "predecessor_task_spec_sha256": file_sha256(paths["predecessor_task_spec"]),
        "adoption_pipeline": adoption,
        "route_b_pipeline": route_b,
        "same_contract_ab_owned_by_predecessor": True,
        "trace_on_off_sequential": True,
        "route_b_only_for_scientific_divergence": True,
        "route_b_M_MAX": 50_000,
        "route_b_M_MIN": 20_000,
        "route_b_candidate_capacity": 100_000,
        "route_b_checkpoint_interval": 2_500,
        "route_b_convergence_patience": 2,
        "route_b_test_early_stop": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    spec["spec_sha256"] = stable_sha256(spec)
    return spec


def validate_successor_spec(
    raw: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    value = dict(raw)
    if value.get("schema_version") != SPEC_SCHEMA:
        raise MutNextStageError("successor spec schema changed")
    observed = value.get("spec_sha256")
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise MutNextStageError("successor spec hash is malformed")
    if observed != stable_sha256(
        {key: item for key, item in value.items() if key != "spec_sha256"}
    ):
        raise MutNextStageError("successor spec self hash changed")
    frozen = {
        "same_contract_ab_owned_by_predecessor": True,
        "trace_on_off_sequential": True,
        "route_b_only_for_scientific_divergence": True,
        "route_b_M_MAX": 50_000,
        "route_b_M_MIN": 20_000,
        "route_b_candidate_capacity": 100_000,
        "route_b_checkpoint_interval": 2_500,
        "route_b_convergence_patience": 2,
        "route_b_test_early_stop": False,
    }
    if any(value.get(key) != expected for key, expected in frozen.items()):
        raise MutNextStageError("successor scientific contract changed")
    template_keys = {
        "task_id",
        "execution_commit",
        "predecessor_task_id",
        "predecessor_task_spec",
        "predecessor_terminal",
        "next_action_path",
        "runtime_root",
        "lease_path",
        "publisher_id",
        "publisher_locator",
        "matrix_authority_root",
        "adoption_pipeline",
        "route_b_pipeline",
    }
    rebuilt = build_successor_spec(
        {key: value[key] for key in template_keys}, check_files=check_files
    )
    if rebuilt["predecessor_task_spec_sha256"] != value.get(
        "predecessor_task_spec_sha256"
    ):
        raise MutNextStageError("predecessor task spec bytes changed")
    # Creation time belongs to the signed input and is not regenerated.
    rebuilt["created_at"] = value.get("created_at")
    rebuilt["spec_sha256"] = value["spec_sha256"]
    if rebuilt != value:
        raise MutNextStageError("successor spec canonical content changed")
    return value


def validate_next_action(
    action: Mapping[str, Any],
    *,
    predecessor_terminal: Path,
    expected_task_id: str | None = None,
    expected_task_spec: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    value = dict(action)
    if value.get("schema_version") != DECISION_SCHEMA:
        raise MutNextStageError("next action schema changed")
    observed = value.get("decision_sha256")
    if observed != stable_sha256(
        {key: item for key, item in value.items() if key != "decision_sha256"}
    ):
        raise MutNextStageError("next action self hash changed")
    gate = _absolute(value.get("same_contract_gate"), field="same_contract_gate", must_exist=True)
    gate_value = validate_same_contract_gate(_json(gate, field="same-contract gate"), gate_path=gate)
    terminal = _json(predecessor_terminal, field="A/B owner terminal")
    terminal_task_id = str(terminal.get("task_id") or "")
    if expected_task_id is not None and terminal_task_id != expected_task_id:
        raise MutNextStageError("A/B owner terminal binds another predecessor task")
    try:
        validate_ab_owner_terminal(
            terminal,
            task_id=terminal_task_id,
            gate_path=gate,
        )
    except MutPostABError as exc:
        raise MutNextStageError(str(exc)) from exc
    if value.get("same_contract_gate_sha256") != file_sha256(gate):
        raise MutNextStageError("next action gate bytes changed")
    if value.get("same_contract_gate_summary_sha256") != gate_value.get("summary_sha256"):
        raise MutNextStageError("next action gate summary changed")
    if expected_task_spec is not None:
        expected_spec = _absolute(
            str(expected_task_spec), field="expected predecessor task spec", must_exist=True
        )
        action_spec = _absolute(
            value.get("same_contract_ab_spec"), field="same_contract_ab_spec", must_exist=True
        )
        if action_spec != expected_spec or value.get(
            "same_contract_ab_spec_sha256"
        ) != file_sha256(expected_spec):
            raise MutNextStageError("next action binds another predecessor task spec")
    branch = value.get("branch")
    classification = value.get("classification")
    if (
        branch == "HISTORICAL_ADOPTION_GATES_REQUIRED"
        and classification == "TRACE_ALIAS_ONLY"
        and value.get("historical_adoption_gate_eligible") is True
        and value.get("route_b_evidence_eligible") is False
    ):
        return "ADOPTION", value
    if (
        branch == "ROUTE_B_AUTHORIZATION_REQUIRED"
        and classification == "SCIENTIFIC_STATE_DIVERGENCE"
        and value.get("route_b_evidence_eligible") is True
        and value.get("historical_adoption_gate_eligible") is False
    ):
        return "ROUTE_B", value
    if branch == "ENGINEERING_REPAIR_REQUIRED":
        return "ENGINEERING_REPAIR", value
    raise MutNextStageError("next action has a contradictory branch/classification")


def consume_next_action_once(
    *,
    action_path: Path,
    predecessor_terminal: Path,
    task_spec_sha256: str,
    expected_task_id: str | None = None,
    expected_task_spec: Path | None = None,
) -> tuple[str, dict[str, Any], Path, dict[str, Any]]:
    """Validate then atomically rename a next action on its own filesystem."""

    source = _absolute(str(action_path), field="next_action", must_exist=True)
    action_sha = file_sha256(source)
    lane, action = validate_next_action(
        action=_json(source, field="next action"),
        predecessor_terminal=predecessor_terminal,
        expected_task_id=expected_task_id,
        expected_task_spec=expected_task_spec,
    )
    consumed = source.with_name(f"next_action.consumed-{action_sha}.json")
    if consumed.exists() or consumed.is_symlink():
        raise MutNextStageError("next action was already consumed")
    os.replace(source, consumed)
    directory = os.open(consumed.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    receipt: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "status": "CONSUMED_ONCE",
        "lane": lane,
        "consumed_path": str(consumed),
        "next_action_sha256": action_sha,
        "decision_sha256": action["decision_sha256"],
        "predecessor_terminal": str(predecessor_terminal),
        "predecessor_terminal_sha256": file_sha256(predecessor_terminal),
        "task_spec_sha256": task_spec_sha256,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = stable_sha256(receipt)
    return lane, action, consumed, receipt


def acquire_lease(path: Path):
    target = _absolute(str(path), field="executor lease")
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = target.open("a+b")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def run_stage(
    stage: Mapping[str, Any],
    *,
    log_path: Path,
    progress: Callable[[str, int | None], None] | None = None,
) -> dict[str, Any]:
    row = dict(stage)
    if progress is not None:
        progress(str(row["stage"]), None)
    environment = {**os.environ, **dict(row["environment"])}
    environment["RUN_GNN_ABLATION"] = "0"
    environment["RUN_LLM_ABLATION"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        child = subprocess.Popen(
            list(row["argv"]),
            cwd=str(row["cwd"]),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if progress is not None:
            progress(str(row["stage"]), child.pid)
        returncode = child.wait()
    if returncode != 0:
        raise MutNextStageError(f"{row['stage']} exited {returncode}")
    terminal_path = Path(str(row["expected_terminal"]))
    terminal = _json(terminal_path, field=f"{row['stage']} terminal")
    status = terminal.get("status", terminal.get("state"))
    if status not in row["expected_terminal_status"]:
        raise MutNextStageError(
            f"{row['stage']} terminal status is not accepted: {status}"
        )
    return {
        "stage": row["stage"],
        "returncode": returncode,
        "terminal": str(terminal_path),
        "terminal_sha256": file_sha256(terminal_path),
        "terminal_status": status,
        "route_b_started": terminal.get("route_b_started"),
        "fresh_50k_started": terminal.get("fresh_50k_started"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "ADOPTION_STAGES",
    "CONSUMPTION_SCHEMA",
    "MutNextStageError",
    "ROUTE_B_STAGES",
    "SPEC_SCHEMA",
    "TERMINAL_SCHEMA",
    "acquire_lease",
    "atomic_json",
    "build_successor_spec",
    "consume_next_action_once",
    "run_stage",
    "validate_next_action",
    "validate_successor_spec",
]
