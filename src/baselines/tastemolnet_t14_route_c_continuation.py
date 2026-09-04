"""Durable Route C generation-to-paper continuation for TasteMolNet T14.

The Route C generator is deliberately train-only.  This module keeps its
``GENERATION_PASS`` distinct from a method-cell PASS, waits for the generation
owner and every writer to leave the generation root, and only then exposes the
deferred calibration/test paths to the existing T14 postprocess/verifier.

Matrix publication remains owned by the already-running fast16 publisher
queue.  The continuation writes the queue's pre-bound cell locator and waits
for the unique hash-closed authority to adopt it; it never starts a publisher.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

from scripts.autodl.append_bace_gcf_matrix_authority import _verify_authority
from src.baselines.tastemolnet_comrecgc_full import (
    GENERATION_PASS_MARKER,
    validate_t14_full_output,
)
from src.baselines.tastemolnet_comrecgc_postprocess import PASS_MARKER
from src.baselines.tastemolnet_t14_route_c_fresh import (
    file_sha256,
    load_spec as load_route_c_spec,
)
from src.eval.am_legacy_standardization import scan_live_writers
from src.eval.fast16_matrix_authority_pointer import (
    DEFAULT_LOCK_PATH,
    DEFAULT_STATE_PATH,
    read_authority_pointer,
)


SPEC_SCHEMA = "tastemolnet_t14_route_c_continuation_spec_v1"
HANDOFF_SCHEMA = "tastemolnet_t14_route_c_generation_handoff_v1"
OWNER_SCHEMA = "tastemolnet_t14_route_c_continuation_owner_v1"
HEARTBEAT_SCHEMA = "tastemolnet_t14_route_c_continuation_heartbeat_v1"
TERMINAL_SCHEMA = "tastemolnet_t14_route_c_continuation_terminal_v1"
ROUTE_OWNER_SCHEMA = "tastemolnet_t14_route_c_owner_v1"
QUEUE_SCHEMA = "fast16_matrix_publisher_queue_v1"
QUEUE_HEARTBEAT_SCHEMA = "fast16_matrix_publisher_heartbeat_v1"
LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
CELL_ID = "TasteMolNet/ComRecGC"
DATASET = "TasteMolNet"
METHOD = "ComRecGC"
GPU_INDEX = 2
WAITING_EXIT = 75
GENERATION_PASS_BYTES = f"{GENERATION_PASS_MARKER}\n".encode("utf-8")
FINAL_PASS_BYTES = f"{PASS_MARKER}\n".encode("utf-8")

DEFERRED_INPUT_KEYS = (
    "TASTEMOLNET_CALIBRATION_CSV",
    "TASTEMOLNET_TEST_CSV",
)
POSTPROCESS_PATH_KEYS = (
    *DEFERRED_INPUT_KEYS,
    "TASTEMOLNET_T3_OUTPUT_ROOT",
    "MOLCLR_ROOT",
    "MOLCLR_CHECKPOINT",
    "TASTEMOLNET_WNODE_THRESHOLD_JSON",
    "WNODE_CACHE_DB",
    "NODE_EMBEDDING_CACHE_DIR",
    "AUTODL_DATA_ROOT",
    "AUTODL_RUNTIME_ROOT",
    "AUTODL_CONTROL_ROOT",
    "AUTODL_PYTHON",
)
POSTPROCESS_FIXED_ENVIRONMENT = {
    "RUN_TASTEMOLNET": "1",
    "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
    "TASTE_PAPER_RESULTS_ALLOWED": "1",
    "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
    "RUN_GNN_ABLATION": "0",
    "RUN_LLM_ABLATION": "0",
}
EXACT_PUBLISHER_ENTRYPOINT = "run_fast16_matrix_publisher_queue.py"


class T14RouteCContinuationError(RuntimeError):
    """The Route C generation-to-paper handoff is incomplete or changed."""


def require_unpublished_matrix_cell(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    """Fail before a new Route C attempt when T14 is already authoritative.

    An absent pointer is intentionally only observed, never initialized here.
    When it exists, the ordinary locked reader reopens the hash-closed
    authority before deciding whether the cell remains available.
    """

    state = Path(state_path)
    lock = Path(lock_path)
    if state != DEFAULT_STATE_PATH or lock != DEFAULT_LOCK_PATH:
        raise T14RouteCContinuationError("Route C must use the unique fast16 authority")
    if state.is_symlink():
        raise T14RouteCContinuationError("fast16 authority pointer is indirect")
    if not state.exists():
        return {
            "status": "MATRIX_CELL_UNAPPLIED",
            "cell_id": CELL_ID,
            "authority_state_path": str(state),
        }
    if not state.is_file():
        raise T14RouteCContinuationError("fast16 authority pointer is not a file")
    pointer = read_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=None,
    )
    if CELL_ID in pointer.get("applied_cells", []):
        raise T14RouteCContinuationError(
            "fast16 authority already contains TasteMolNet/ComRecGC; "
            "refusing a new Route C attempt"
        )
    return {
        "status": "MATRIX_CELL_UNAPPLIED",
        "cell_id": CELL_ID,
        "authority_state_path": str(state),
        "latest_authority_root": pointer["latest_authority_root"],
        "latest_count": pointer["latest_count"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute(value: Any, *, field: str) -> Path:
    """Validate an absolute normalized spelling without touching the path."""

    if not isinstance(value, str) or not value:
        raise T14RouteCContinuationError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise T14RouteCContinuationError(f"{field} must be normalized and absolute")
    return path


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _physical_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise T14RouteCContinuationError(
            f"{label} must be one physical nonempty JSON file: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14RouteCContinuationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise T14RouteCContinuationError(f"{label} must be one JSON object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise T14RouteCContinuationError(f"refusing an indirect JSON path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _self_hash(value: Mapping[str, Any], *, field: str) -> str:
    expected = stable_sha256({key: item for key, item in value.items() if key != field})
    observed = value.get(field)
    if observed != expected:
        raise T14RouteCContinuationError(f"{field} changed")
    return str(observed)


def _queue_binding(
    manifest_path: Path,
    *,
    locator_path: Path,
    t3_root: Path,
) -> dict[str, Any]:
    manifest = _physical_json(manifest_path, label="fast16 publisher queue manifest")
    if manifest.get("schema_version") != QUEUE_SCHEMA:
        raise T14RouteCContinuationError("fast16 publisher queue schema changed")
    if manifest.get("authority_state_path") != str(DEFAULT_STATE_PATH):
        raise T14RouteCContinuationError("queue does not use the unique fast16 state")
    if manifest.get("authority_lock_path") != str(DEFAULT_LOCK_PATH):
        raise T14RouteCContinuationError("queue does not use the unique fast16 lock")
    initial = _absolute(
        manifest.get("initial_authority_root"), field="initial_authority_root"
    )
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise T14RouteCContinuationError("fast16 publisher queue has no cell list")
    matching = [
        row
        for row in cells
        if isinstance(row, Mapping)
        and row.get("dataset") == DATASET
        and row.get("method") == METHOD
    ]
    if len(matching) != 1:
        raise T14RouteCContinuationError(
            "queue must contain exactly one TasteMolNet/ComRecGC cell"
        )
    cell = dict(matching[0])
    if cell.get("terminal_root") is not None:
        raise T14RouteCContinuationError(
            "Route C queue cell must use its pre-bound terminal locator"
        )
    if cell.get("terminal_root_locator") != str(locator_path):
        raise T14RouteCContinuationError("queue binds a different T14 locator")
    output_root = _absolute(cell.get("output_root"), field="matrix output root")
    taste = manifest.get("taste")
    required_taste = {
        "t3_root",
        "policy_path",
        "policy_receipt",
        "prepared_root",
        "graph_cache_root",
    }
    if not isinstance(taste, Mapping) or not required_taste.issubset(taste):
        raise T14RouteCContinuationError("queue lacks the shared Taste binding")
    if taste.get("t3_root") != str(t3_root):
        raise T14RouteCContinuationError("postprocess and matrix queue use different T3 roots")
    for field in required_taste:
        _absolute(taste.get(field), field=f"queue taste.{field}")
    return {
        "initial_authority_root": str(initial),
        "matrix_output_root": str(output_root),
        "taste_binding_sha256": stable_sha256(dict(taste)),
    }


def build_continuation_spec(
    *,
    descriptor_path: Path,
    route_c_spec_path: Path,
    config_path: Path,
    continuation_entrypoint: Path,
    postprocess_wrapper: Path,
    postprocess_science_root: Path,
    postprocess_final_root: Path,
    locator_path: Path,
    publisher_queue_manifest: Path,
    publisher_heartbeat: Path,
    publisher_pid_file: Path,
    postprocess_environment: Mapping[str, str],
    poll_seconds: int = 60,
) -> dict[str, Any]:
    """Build a pre-generation descriptor without opening calibration or test."""

    # This read-only gate must run before any continuation root is created.
    require_unpublished_matrix_cell()
    route = load_route_c_spec(Path(route_c_spec_path), check_files=True)
    descriptor = Path(descriptor_path)
    config = Path(config_path)
    entrypoint = Path(continuation_entrypoint)
    wrapper = Path(postprocess_wrapper)
    queue = Path(publisher_queue_manifest)
    locator = Path(locator_path)
    science = Path(postprocess_science_root)
    final = Path(postprocess_final_root)
    queue_heartbeat = Path(publisher_heartbeat)
    queue_pid = Path(publisher_pid_file)
    owner_root = Path(route["owner_root"])
    generation_root = Path(route["output_root"])
    forbidden = Path(route["forbidden_legacy_root"])
    expected_descriptor = owner_root / "T14_ROUTE_C_CONTINUATION_SPEC.json"
    if descriptor != expected_descriptor:
        raise T14RouteCContinuationError(
            "Route C continuation descriptor must live in its owner root"
        )
    if not 5 <= int(poll_seconds) <= 3600:
        raise T14RouteCContinuationError("continuation poll_seconds must be in [5, 3600]")
    required_environment = set(POSTPROCESS_FIXED_ENVIRONMENT) | set(
        POSTPROCESS_PATH_KEYS
    )
    environment = dict(postprocess_environment)
    if set(environment) != required_environment:
        raise T14RouteCContinuationError(
            "Route C postprocess environment fields changed"
        )
    for key, expected in POSTPROCESS_FIXED_ENVIRONMENT.items():
        if environment.get(key) != expected:
            raise T14RouteCContinuationError(f"postprocess environment changed: {key}")
    paths = {
        key: _absolute(environment.get(key), field=f"postprocess_environment.{key}")
        for key in POSTPROCESS_PATH_KEYS
    }
    if paths["AUTODL_PYTHON"] != Path(route["python"]):
        raise T14RouteCContinuationError("generation and postprocess Python differ")
    if paths["TASTEMOLNET_T3_OUTPUT_ROOT"] != Path(
        route["science_environment"]["TASTEMOLNET_T3_OUTPUT_ROOT"]
    ):
        raise T14RouteCContinuationError("generation and postprocess T3 roots differ")
    if paths["AUTODL_CONTROL_ROOT"] / "fast16_matrix_authority" / "state.json" != DEFAULT_STATE_PATH:
        raise T14RouteCContinuationError("postprocess control root differs from fast16 authority")
    if paths["AUTODL_CONTROL_ROOT"] / "fast16_matrix_authority" / "publish.lock" != DEFAULT_LOCK_PATH:
        raise T14RouteCContinuationError("postprocess control root differs from fast16 lock")
    for path, label in (
        (config, "config"),
        (entrypoint, "continuation entrypoint"),
        (wrapper, "postprocess wrapper"),
        (queue, "publisher queue manifest"),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise T14RouteCContinuationError(f"{label} must be one physical file")
    for path, label in (
        (science, "postprocess science root"),
        (final, "postprocess final root"),
        (locator, "matrix locator"),
        (queue_heartbeat, "publisher heartbeat"),
        (queue_pid, "publisher PID file"),
    ):
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise T14RouteCContinuationError(f"{label} must be normalized and absolute")
    queue_binding = _queue_binding(
        queue,
        locator_path=locator,
        t3_root=paths["TASTEMOLNET_T3_OUTPUT_ROOT"],
    )
    protected = (generation_root, owner_root, forbidden)
    new_roots = (science, final, Path(queue_binding["matrix_output_root"]))
    for candidate in new_roots:
        if any(
            _is_within(candidate, root) or _is_within(root, candidate)
            for root in protected
        ):
            raise T14RouteCContinuationError(
                "postprocess/publication roots overlap generation, owner, or legacy roots"
            )
    if len(set(new_roots)) != len(new_roots):
        raise T14RouteCContinuationError("postprocess and matrix output roots must differ")
    for candidate in (*new_roots, locator):
        if candidate.exists() or candidate.is_symlink():
            raise T14RouteCContinuationError(
                f"fresh Route C continuation path already exists: {candidate}"
            )
    value: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA,
        "descriptor_path": str(descriptor),
        "route_c_spec": str(route_c_spec_path),
        "route_c_spec_sha256": file_sha256(route_c_spec_path),
        "route_c_execution_commit": route["execution_commit"],
        "route_c_attempt_uuid": route["attempt_uuid"],
        "generation_root": str(generation_root),
        "generation_owner_root": str(owner_root),
        "generation_owner_terminal": str(owner_root / "terminal.json"),
        "generation_handoff": str(owner_root / "generation_to_postprocess_handoff.json"),
        "continuation_root": str(owner_root / "continuation"),
        "config": str(config),
        "config_sha256": file_sha256(config),
        "python": route["python"],
        "continuation_entrypoint": str(entrypoint),
        "continuation_entrypoint_sha256": file_sha256(entrypoint),
        "postprocess_wrapper": str(wrapper),
        "postprocess_wrapper_sha256": file_sha256(wrapper),
        "postprocess_science_root": str(science),
        "postprocess_final_root": str(final),
        "postprocess_run_id": f"taste-t14-route-c-postprocess-{route['attempt_uuid']}",
        "postprocess_gpu_index": GPU_INDEX,
        "postprocess_environment": environment,
        "deferred_inputs": list(DEFERRED_INPUT_KEYS),
        "deferred_inputs_opened_before_generation_freeze": False,
        "generation_pass_is_method_cell_pass": False,
        "publisher_started_by_continuation": False,
        "matrix": {
            "cell_id": CELL_ID,
            "queue_manifest": str(queue),
            "queue_manifest_sha256": file_sha256(queue),
            "queue_heartbeat": str(queue_heartbeat),
            "queue_pid_file": str(queue_pid),
            "authority_state_path": str(DEFAULT_STATE_PATH),
            "authority_lock_path": str(DEFAULT_LOCK_PATH),
            "locator_path": str(locator),
            **queue_binding,
        },
        "poll_seconds": int(poll_seconds),
        "created_at": _utc_now(),
    }
    value["spec_sha256"] = stable_sha256(value)
    return validate_continuation_spec(value, check_files=True)


def validate_continuation_spec(
    raw: Mapping[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    """Validate descriptor bytes while keeping deferred split paths unopened."""

    if not isinstance(raw, Mapping):
        raise T14RouteCContinuationError("continuation spec must be one object")
    value = dict(raw)
    required = {
        "schema_version",
        "descriptor_path",
        "route_c_spec",
        "route_c_spec_sha256",
        "route_c_execution_commit",
        "route_c_attempt_uuid",
        "generation_root",
        "generation_owner_root",
        "generation_owner_terminal",
        "generation_handoff",
        "continuation_root",
        "config",
        "config_sha256",
        "python",
        "continuation_entrypoint",
        "continuation_entrypoint_sha256",
        "postprocess_wrapper",
        "postprocess_wrapper_sha256",
        "postprocess_science_root",
        "postprocess_final_root",
        "postprocess_run_id",
        "postprocess_gpu_index",
        "postprocess_environment",
        "deferred_inputs",
        "deferred_inputs_opened_before_generation_freeze",
        "generation_pass_is_method_cell_pass",
        "publisher_started_by_continuation",
        "matrix",
        "poll_seconds",
        "created_at",
        "spec_sha256",
    }
    if set(value) != required or value.get("schema_version") != SPEC_SCHEMA:
        raise T14RouteCContinuationError("continuation spec fields/schema changed")
    _self_hash(value, field="spec_sha256")
    if (
        value.get("postprocess_gpu_index") != GPU_INDEX
        or value.get("deferred_inputs") != list(DEFERRED_INPUT_KEYS)
        or value.get("deferred_inputs_opened_before_generation_freeze") is not False
        or value.get("generation_pass_is_method_cell_pass") is not False
        or value.get("publisher_started_by_continuation") is not False
        or not 5 <= int(value.get("poll_seconds", 0)) <= 3600
    ):
        raise T14RouteCContinuationError("continuation safety contract changed")
    route_spec_path = _absolute(value.get("route_c_spec"), field="route_c_spec")
    descriptor = _absolute(value.get("descriptor_path"), field="descriptor_path")
    owner_root = _absolute(value.get("generation_owner_root"), field="generation_owner_root")
    generation_root = _absolute(value.get("generation_root"), field="generation_root")
    expected_paths = {
        "descriptor_path": owner_root / "T14_ROUTE_C_CONTINUATION_SPEC.json",
        "generation_owner_terminal": owner_root / "terminal.json",
        "generation_handoff": owner_root / "generation_to_postprocess_handoff.json",
        "continuation_root": owner_root / "continuation",
    }
    for field, expected in expected_paths.items():
        if _absolute(value.get(field), field=field) != expected:
            raise T14RouteCContinuationError(f"continuation path changed: {field}")
    environment = value.get("postprocess_environment")
    if not isinstance(environment, Mapping):
        raise T14RouteCContinuationError("postprocess environment is malformed")
    required_environment = set(POSTPROCESS_FIXED_ENVIRONMENT) | set(
        POSTPROCESS_PATH_KEYS
    )
    if set(environment) != required_environment:
        raise T14RouteCContinuationError("postprocess environment fields changed")
    for key, expected in POSTPROCESS_FIXED_ENVIRONMENT.items():
        if environment.get(key) != expected:
            raise T14RouteCContinuationError(f"postprocess environment changed: {key}")
    for key in POSTPROCESS_PATH_KEYS:
        _absolute(environment.get(key), field=f"postprocess_environment.{key}")
    matrix = value.get("matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != {
        "cell_id",
        "queue_manifest",
        "queue_manifest_sha256",
        "queue_heartbeat",
        "queue_pid_file",
        "authority_state_path",
        "authority_lock_path",
        "locator_path",
        "initial_authority_root",
        "matrix_output_root",
        "taste_binding_sha256",
    }:
        raise T14RouteCContinuationError("matrix queue binding changed")
    if (
        matrix.get("cell_id") != CELL_ID
        or matrix.get("authority_state_path") != str(DEFAULT_STATE_PATH)
        or matrix.get("authority_lock_path") != str(DEFAULT_LOCK_PATH)
    ):
        raise T14RouteCContinuationError("matrix authority binding changed")
    for field in (
        "queue_manifest",
        "queue_heartbeat",
        "queue_pid_file",
        "authority_state_path",
        "authority_lock_path",
        "locator_path",
        "initial_authority_root",
        "matrix_output_root",
    ):
        _absolute(matrix.get(field), field=f"matrix.{field}")
    if check_files:
        route = load_route_c_spec(route_spec_path, check_files=True)
        if (
            file_sha256(route_spec_path) != value.get("route_c_spec_sha256")
            or route.get("execution_commit") != value.get("route_c_execution_commit")
            or route.get("attempt_uuid") != value.get("route_c_attempt_uuid")
            or route.get("output_root") != str(generation_root)
            or route.get("owner_root") != str(owner_root)
            or route.get("legacy_checkpoint_loaded") is not False
        ):
            raise T14RouteCContinuationError("Route C generation descriptor changed")
        physical_files = {
            "config": "config_sha256",
            "continuation_entrypoint": "continuation_entrypoint_sha256",
            "postprocess_wrapper": "postprocess_wrapper_sha256",
        }
        for field, sha_field in physical_files.items():
            path = _absolute(value.get(field), field=field)
            if path.is_symlink() or not path.is_file():
                raise T14RouteCContinuationError(f"continuation file changed: {field}")
            if file_sha256(path) != value.get(sha_field):
                raise T14RouteCContinuationError(f"continuation bytes changed: {field}")
        queue_path = _absolute(matrix.get("queue_manifest"), field="matrix.queue_manifest")
        if file_sha256(queue_path) != matrix.get("queue_manifest_sha256"):
            raise T14RouteCContinuationError("publisher queue manifest bytes changed")
        binding = _queue_binding(
            queue_path,
            locator_path=Path(str(matrix["locator_path"])),
            t3_root=Path(str(environment["TASTEMOLNET_T3_OUTPUT_ROOT"])),
        )
        for field in ("initial_authority_root", "matrix_output_root", "taste_binding_sha256"):
            if matrix.get(field) != binding[field]:
                raise T14RouteCContinuationError(f"publisher queue binding changed: {field}")
    # No filesystem operation on calibration or test is permitted above.
    return value


def write_continuation_spec(path: Path, value: Mapping[str, Any]) -> None:
    checked = validate_continuation_spec(value, check_files=True)
    if path != Path(str(checked["descriptor_path"])):
        raise T14RouteCContinuationError("continuation spec destination changed")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"continuation spec must be fresh: {path}")
    _atomic_json(path, checked)


def load_continuation_spec(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise T14RouteCContinuationError(
            "continuation spec must be one absolute physical file"
        )
    value = _physical_json(path, label="Route C continuation spec")
    if value.get("descriptor_path") != str(path):
        raise T14RouteCContinuationError("continuation spec path binding changed")
    return validate_continuation_spec(value, check_files=True)


def _validate_generation_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    checked = dict(value)
    required = {
        "status": "PASS",
        "marker": GENERATION_PASS_MARKER,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_result_eligible": False,
        "method_cell_pass": False,
    }
    changed = [key for key, expected in required.items() if checked.get(key) != expected]
    if changed:
        raise T14RouteCContinuationError(
            "generation-only verification was relabelled as a method PASS: "
            + ",".join(changed)
        )
    return checked


def _pending_owner_terminal(
    spec: Mapping[str, Any], handoff: Mapping[str, Any]
) -> dict[str, Any]:
    terminal: dict[str, Any] = {
        "schema_version": ROUTE_OWNER_SCHEMA,
        "status": "GENERATION_PASS_PENDING_POSTPROCESS",
        "owner_pid": handoff["generation_owner_pid"],
        "owner_start_ticks": handoff["generation_owner_start_ticks"],
        "output_root": spec["generation_root"],
        "generation_handoff": spec["generation_handoff"],
        "generation_handoff_sha256": handoff["handoff_sha256"],
        "continuation_spec": spec["descriptor_path"],
        "continuation_spec_sha256": spec["spec_sha256"],
        "generation_pass": True,
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "continuation_required": True,
        "continuation_launch_status": "PENDING_LAUNCH",
        "publisher_started": False,
        # Bind repair of the handoff->terminal crash window to an already
        # sealed timestamp instead of minting different terminal bytes.
        "completed_at": handoff["created_at"],
    }
    terminal["terminal_sha256"] = stable_sha256(terminal)
    return terminal


def publish_generation_handoff(
    *,
    continuation_spec_path: Path,
    generation_verification: Mapping[str, Any],
    owner_pid: int,
    owner_start_ticks: int,
) -> dict[str, Any]:
    """Publish the immutable pending handoff without opening held-out inputs.

    The Route C owner calls this only after its generation CLI has independently
    validated ``GENERATION_PASS``.  This function writes both the handoff and
    the owner terminal.  Neither is a method-cell PASS.
    """

    spec = load_continuation_spec(Path(continuation_spec_path))
    verification = _validate_generation_projection(generation_verification)
    generation = Path(spec["generation_root"])
    marker = generation / "GENERATION_PASS"
    if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != GENERATION_PASS_BYTES:
        raise T14RouteCContinuationError("Route C generation marker is absent or changed")
    if verification.get("output_root") != str(generation):
        raise T14RouteCContinuationError("generation verification binds another root")
    manifest_path = generation / "generation_manifest.json"
    manifest = _physical_json(manifest_path, label="Route C generation manifest")
    manifest_required = {
        "status": "PASS",
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_result_eligible": False,
        "method_cell_pass": False,
    }
    if any(manifest.get(key) != expected for key, expected in manifest_required.items()):
        raise T14RouteCContinuationError("Route C generation manifest is not train-only")
    if type(owner_pid) is not int or owner_pid <= 0:
        raise T14RouteCContinuationError("generation owner PID is invalid")
    if type(owner_start_ticks) is not int or owner_start_ticks <= 0:
        raise T14RouteCContinuationError("generation owner start ticks are invalid")
    handoff_path = Path(spec["generation_handoff"])
    terminal_path = Path(spec["generation_owner_terminal"])
    if not (handoff_path.exists() or handoff_path.is_symlink()) and (
        terminal_path.exists() or terminal_path.is_symlink()
    ):
        raise T14RouteCContinuationError(
            "Route C owner terminal exists without its generation handoff"
        )
    if handoff_path.exists() or handoff_path.is_symlink():
        handoff = validate_generation_handoff(handoff_path, spec=spec)
        if handoff.get("generation_inventory_sha256") != verification.get(
            "inventory_sha256"
        ):
            raise T14RouteCContinuationError(
                "existing generation handoff binds another inventory"
            )
    else:
        handoff = {
            "schema_version": HANDOFF_SCHEMA,
            "status": "GENERATION_PASS_PENDING_POSTPROCESS",
            "continuation_spec": str(continuation_spec_path),
            "continuation_spec_sha256": spec["spec_sha256"],
            "route_c_spec": spec["route_c_spec"],
            "route_c_spec_sha256": spec["route_c_spec_sha256"],
            "route_c_attempt_uuid": spec["route_c_attempt_uuid"],
            "generation_root": str(generation),
            "generation_pass_sha256": file_sha256(marker),
            "generation_manifest_sha256": file_sha256(manifest_path),
            "generation_inventory_sha256": verification.get("inventory_sha256"),
            "generation_owner_pid": owner_pid,
            "generation_owner_start_ticks": owner_start_ticks,
            "generation_owner_terminal": spec["generation_owner_terminal"],
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "paper_result_eligible": False,
            "method_cell_pass": False,
            "publisher_started": False,
            "created_at": _utc_now(),
        }
        handoff["handoff_sha256"] = stable_sha256(handoff)
        _atomic_json(handoff_path, handoff)

    terminal = _pending_owner_terminal(spec, handoff)
    if terminal_path.exists() or terminal_path.is_symlink():
        observed = _physical_json(terminal_path, label="Route C owner terminal")
        _self_hash(observed, field="terminal_sha256")
        if observed != terminal:
            raise T14RouteCContinuationError(
                "existing Route C owner terminal differs from sealed handoff"
            )
    else:
        _atomic_json(terminal_path, terminal)
    return handoff


def validate_generation_handoff(
    path: Path, *, spec: Mapping[str, Any]
) -> dict[str, Any]:
    value = _physical_json(path, label="Route C generation handoff")
    required = {
        "schema_version",
        "status",
        "continuation_spec",
        "continuation_spec_sha256",
        "route_c_spec",
        "route_c_spec_sha256",
        "route_c_attempt_uuid",
        "generation_root",
        "generation_pass_sha256",
        "generation_manifest_sha256",
        "generation_inventory_sha256",
        "generation_owner_pid",
        "generation_owner_start_ticks",
        "generation_owner_terminal",
        "validation_loaded",
        "calibration_loaded",
        "test_loaded",
        "paper_result_eligible",
        "method_cell_pass",
        "publisher_started",
        "created_at",
        "handoff_sha256",
    }
    if set(value) != required or value.get("schema_version") != HANDOFF_SCHEMA:
        raise T14RouteCContinuationError("generation handoff schema changed")
    _self_hash(value, field="handoff_sha256")
    fixed = {
        "status": "GENERATION_PASS_PENDING_POSTPROCESS",
        "continuation_spec": spec["descriptor_path"],
        "continuation_spec_sha256": spec["spec_sha256"],
        "route_c_spec": spec["route_c_spec"],
        "route_c_spec_sha256": spec["route_c_spec_sha256"],
        "route_c_attempt_uuid": spec["route_c_attempt_uuid"],
        "generation_root": spec["generation_root"],
        "generation_owner_terminal": spec["generation_owner_terminal"],
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "publisher_started": False,
    }
    changed = [key for key, expected in fixed.items() if value.get(key) != expected]
    if changed:
        raise T14RouteCContinuationError(f"generation handoff changed: {changed}")
    for field in ("generation_owner_pid", "generation_owner_start_ticks"):
        if type(value.get(field)) is not int or int(value[field]) <= 0:
            raise T14RouteCContinuationError(f"generation handoff {field} is invalid")
    marker = Path(spec["generation_root"]) / "GENERATION_PASS"
    manifest = Path(spec["generation_root"]) / "generation_manifest.json"
    if (
        file_sha256(marker) != value.get("generation_pass_sha256")
        or file_sha256(manifest) != value.get("generation_manifest_sha256")
    ):
        raise T14RouteCContinuationError("generation handoff source bytes changed")
    return value


def continuation_command(continuation_spec_path: Path) -> list[str]:
    spec = load_continuation_spec(Path(continuation_spec_path))
    return [
        spec["python"],
        "-I",
        "-B",
        spec["continuation_entrypoint"],
        "--config",
        spec["config"],
        "--set",
        "inference.fallback_to_heuristic=false",
        "--continuation-spec",
        spec["descriptor_path"],
    ]


def _proc_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    path = proc_root / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def launch_continuation_owner(
    continuation_spec_path: Path,
    *,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Adopt a live owner or relaunch the same descriptor/root after exit.

    The launch lock prevents concurrent binders.  A dead continuation may be
    relaunched only against the same immutable descriptor, generation handoff,
    science root, final root, and locator.  A structured ``BLOCKED`` terminal
    is not retried automatically.
    """

    spec = load_continuation_spec(Path(continuation_spec_path))
    handoff = validate_generation_handoff(Path(spec["generation_handoff"]), spec=spec)
    terminal = _physical_json(
        Path(spec["generation_owner_terminal"]), label="Route C owner terminal"
    )
    _self_hash(terminal, field="terminal_sha256")
    if terminal != _pending_owner_terminal(spec, handoff):
        raise T14RouteCContinuationError(
            "Route C owner terminal is not the exact pending handoff"
        )
    continuation_root = Path(spec["continuation_root"])
    if not continuation_root.exists() and not continuation_root.is_symlink():
        # A retry of this binder may reuse only an already-sealed continuation
        # root.  A genuinely new owner cannot start after another route has
        # already published the T14 method cell.
        require_unpublished_matrix_cell()
    continuation_root.mkdir(parents=True, exist_ok=True)
    if continuation_root.is_symlink():
        raise T14RouteCContinuationError("continuation root cannot be indirect")
    launch_lock_path = continuation_root / "launch.lock"
    lock_descriptor = os.open(
        launch_lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise T14RouteCContinuationError(
                "another Route C continuation binder is active"
            ) from exc
        command = continuation_command(Path(continuation_spec_path))
        latest_path = continuation_root / "launch.json"
        prior: dict[str, Any] | None = None
        if latest_path.exists() or latest_path.is_symlink():
            prior = _physical_json(latest_path, label="continuation launch receipt")
            launch_fields = {
                "schema_version",
                "status",
                "launch_ordinal",
                "relaunch_same_descriptor",
                "pid",
                "start_ticks",
                "command",
                "command_sha256",
                "continuation_spec",
                "continuation_spec_sha256",
                "generation_handoff_sha256",
                "postprocess_science_root",
                "postprocess_final_root",
                "matrix_locator",
                "publisher_started",
                "started_at",
                "launch_sha256",
            }
            if set(prior) != launch_fields or prior.get("schema_version") != (
                "tastemolnet_t14_route_c_continuation_launch_v1"
            ) or prior.get("status") != "STARTED":
                raise T14RouteCContinuationError("continuation launch schema changed")
            _self_hash(prior, field="launch_sha256")
            ordinal = prior.get("launch_ordinal")
            if (
                type(ordinal) is not int
                or ordinal <= 0
                or prior.get("relaunch_same_descriptor") is not (ordinal > 1)
                or prior.get("command") != command
                or prior.get("command_sha256") != stable_sha256(command)
                or prior.get("continuation_spec") != spec["descriptor_path"]
                or prior.get("continuation_spec_sha256") != spec["spec_sha256"]
                or prior.get("generation_handoff_sha256") != handoff["handoff_sha256"]
                or prior.get("postprocess_science_root")
                != spec["postprocess_science_root"]
                or prior.get("postprocess_final_root") != spec["postprocess_final_root"]
                or prior.get("matrix_locator") != spec["matrix"]["locator_path"]
                or prior.get("publisher_started") is not False
            ):
                raise T14RouteCContinuationError("continuation launch binding changed")
            sealed_attempt = (
                continuation_root / "launches" / f"launch-{ordinal:04d}.json"
            )
            if _physical_json(
                sealed_attempt, label="sealed continuation launch attempt"
            ) != prior:
                raise T14RouteCContinuationError(
                    "continuation launch pointer differs from sealed attempt"
                )
            pid = prior.get("pid")
            ticks = prior.get("start_ticks")
            if type(pid) is not int or type(ticks) is not int:
                raise T14RouteCContinuationError("continuation launch PID is invalid")
            if _process_alive(pid, ticks, proc_root=proc_root):
                tokens = _process_tokens(proc_root / str(pid) / "cmdline")
                if (
                    spec["continuation_entrypoint"] not in tokens
                    or spec["descriptor_path"] not in tokens
                ):
                    raise T14RouteCContinuationError(
                        "live continuation PID command changed"
                    )
                return {
                    **prior,
                    "launch_status": "ADOPTED_LIVE_OWNER",
                }
        continuation_terminal = continuation_root / "terminal.json"
        if continuation_terminal.is_file() and not continuation_terminal.is_symlink():
            completed = _physical_json(
                continuation_terminal, label="continuation terminal"
            )
            if completed.get("schema_version") != TERMINAL_SCHEMA:
                raise T14RouteCContinuationError("continuation terminal schema changed")
            if completed.get("status") == "PASS":
                _self_hash(completed, field="terminal_sha256")
                return {
                    "schema_version": "tastemolnet_t14_route_c_continuation_launch_v1",
                    "status": "PASS",
                    "launch_status": "ALREADY_COMPLETE",
                    "continuation_spec": spec["descriptor_path"],
                    "continuation_spec_sha256": spec["spec_sha256"],
                    "publisher_started": False,
                }
            raise T14RouteCContinuationError(
                "structured continuation failure requires review before relaunch"
            )
        elif continuation_terminal.exists() or continuation_terminal.is_symlink():
            raise T14RouteCContinuationError("continuation terminal is indirect")

        ordinal = 1 if prior is None else int(prior.get("launch_ordinal", 0)) + 1
        stdout_path = continuation_root / "owner.out"
        stderr_path = continuation_root / "owner.err"
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("TASTEMOLNET_")
            and not key.startswith("T14_")
            and key not in POSTPROCESS_FIXED_ENVIRONMENT
        }
        environment.update({"PYTHONNOUSERSITE": "1"})
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
            "ab", buffering=0
        ) as stderr:
            child = popen(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        ticks = _proc_start_ticks(int(child.pid), proc_root=proc_root)
        if ticks is None:
            child.terminate()
            raise T14RouteCContinuationError("continuation child identity is unreadable")
        receipt = {
            "schema_version": "tastemolnet_t14_route_c_continuation_launch_v1",
            "status": "STARTED",
            "launch_ordinal": ordinal,
            "relaunch_same_descriptor": prior is not None,
            "pid": int(child.pid),
            "start_ticks": ticks,
            "command": command,
            "command_sha256": stable_sha256(command),
            "continuation_spec": spec["descriptor_path"],
            "continuation_spec_sha256": spec["spec_sha256"],
            "generation_handoff_sha256": handoff["handoff_sha256"],
            "postprocess_science_root": spec["postprocess_science_root"],
            "postprocess_final_root": spec["postprocess_final_root"],
            "matrix_locator": spec["matrix"]["locator_path"],
            "publisher_started": False,
            "started_at": _utc_now(),
        }
        receipt["launch_sha256"] = stable_sha256(receipt)
        attempts = continuation_root / "launches"
        attempts.mkdir(parents=True, exist_ok=True)
        attempt_path = attempts / f"launch-{ordinal:04d}.json"
        if attempt_path.exists() or attempt_path.is_symlink():
            child.terminate()
            raise T14RouteCContinuationError("continuation launch ordinal exists")
        _atomic_json(attempt_path, receipt)
        _atomic_json(latest_path, receipt)
        return {**receipt, "launch_status": "STARTED"}
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _process_alive(
    pid: int, start_ticks: int, *, proc_root: Path = Path("/proc")
) -> bool:
    return _proc_start_ticks(pid, proc_root=proc_root) == start_ticks


def _process_tokens(path: Path) -> list[str]:
    try:
        return [
            token.decode("utf-8", errors="replace")
            for token in path.read_bytes().split(b"\0")
            if token
        ]
    except OSError:
        return []


def find_fast16_publishers(*, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    publishers: list[dict[str, Any]] = []
    try:
        directories = list(proc_root.iterdir())
    except OSError as exc:
        raise T14RouteCContinuationError("procfs is unavailable for publisher audit") from exc
    for directory in directories:
        if not directory.name.isdigit():
            continue
        tokens = _process_tokens(directory / "cmdline")
        if not any(Path(token).name == EXACT_PUBLISHER_ENTRYPOINT for token in tokens):
            continue
        ticks = _proc_start_ticks(int(directory.name), proc_root=proc_root)
        if ticks is None:
            continue
        publishers.append(
            {
                "pid": int(directory.name),
                "start_ticks": ticks,
                "tokens": tokens,
                "command_sha256": stable_sha256(tokens),
            }
        )
    return sorted(publishers, key=lambda row: int(row["pid"]))


def _command_option(tokens: list[str], option: str) -> str:
    """Return one unambiguous long-option value from a procfs command line."""

    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise T14RouteCContinuationError(
                    f"fast16 publisher command omits {option} value"
                )
            values.append(tokens[index + 1])
        elif token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise T14RouteCContinuationError(
            f"fast16 publisher command has ambiguous {option} binding"
        )
    return values[0]


def _publisher_queue_claim(publisher: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen one live publisher's queue and identify its exact cell claims."""

    tokens = list(publisher.get("tokens") or [])
    queue_spelling = _command_option(tokens, "--queue-manifest")
    heartbeat_spelling = _command_option(tokens, "--heartbeat-path")
    queue_path = _absolute(queue_spelling, field="live publisher queue manifest")
    heartbeat_path = _absolute(
        heartbeat_spelling, field="live publisher heartbeat path"
    )
    queue = _physical_json(queue_path, label="live fast16 publisher queue")
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise T14RouteCContinuationError("live publisher queue schema changed")
    if (
        queue.get("authority_state_path") != str(DEFAULT_STATE_PATH)
        or queue.get("authority_lock_path") != str(DEFAULT_LOCK_PATH)
    ):
        raise T14RouteCContinuationError(
            "live publisher escapes the unique fast16 authority"
        )
    cells = queue.get("cells")
    if not isinstance(cells, list) or not cells:
        raise T14RouteCContinuationError("live publisher queue has no cells")
    claims: list[str] = []
    for row in cells:
        if not isinstance(row, Mapping):
            raise T14RouteCContinuationError("live publisher queue cell is malformed")
        dataset = row.get("dataset")
        method = row.get("method")
        if not isinstance(dataset, str) or not isinstance(method, str):
            raise T14RouteCContinuationError("live publisher queue cell identity changed")
        claims.append(f"{dataset}/{method}")
    if len(claims) != len(set(claims)):
        raise T14RouteCContinuationError("live publisher queue duplicates a cell")
    return {
        **dict(publisher),
        "queue_manifest": str(queue_path),
        "queue_manifest_sha256": file_sha256(queue_path),
        "heartbeat_path": str(heartbeat_path),
        "claimed_cells": claims,
        "claims_t14": CELL_ID in claims,
    }


def _validate_owner_terminal(spec: Mapping[str, Any], handoff: Mapping[str, Any]) -> None:
    terminal = _physical_json(
        Path(spec["generation_owner_terminal"]), label="Route C owner terminal"
    )
    _self_hash(terminal, field="terminal_sha256")
    if terminal != _pending_owner_terminal(spec, handoff):
        raise T14RouteCContinuationError("Route C owner terminal was relabelled")


def _heartbeat(root: Path, spec: Mapping[str, Any], *, state: str, **extra: Any) -> None:
    _atomic_json(
        root / "heartbeat.json",
        {
            "schema_version": HEARTBEAT_SCHEMA,
            "pid": os.getpid(),
            "state": state,
            "generation_root": spec["generation_root"],
            "postprocess_science_root": spec["postprocess_science_root"],
            "postprocess_final_root": spec["postprocess_final_root"],
            "generation_pass_is_method_cell_pass": False,
            "publisher_started": False,
            "updated_at": _utc_now(),
            **extra,
        },
    )


def _generation_is_frozen(
    spec: Mapping[str, Any], handoff: Mapping[str, Any], *, proc_root: Path
) -> dict[str, Any]:
    generation = Path(spec["generation_root"])
    writer_audit = scan_live_writers(generation, proc_root=str(proc_root))
    if (
        writer_audit.get("procfs_verified") is not True
        or writer_audit.get("writable_fd_count") != 0
    ):
        raise T14RouteCContinuationError("generation root still has a live writer")
    verification = _validate_generation_projection(validate_t14_full_output(generation))
    if (
        verification.get("output_root") != str(generation)
        or verification.get("inventory_sha256") != handoff.get("generation_inventory_sha256")
        or file_sha256(generation / "GENERATION_PASS")
        != handoff.get("generation_pass_sha256")
        or file_sha256(generation / "generation_manifest.json")
        != handoff.get("generation_manifest_sha256")
    ):
        raise T14RouteCContinuationError("generation freeze differs from owner handoff")
    receipt: dict[str, Any] = {
        "schema_version": "tastemolnet_t14_route_c_generation_freeze_v1",
        "status": "GENERATION_FROZEN_POSTPROCESS_INPUTS_UNOPENED",
        "generation_root": str(generation),
        "generation_inventory_sha256": verification["inventory_sha256"],
        "generation_writer_audit": writer_audit,
        "generation_pass_is_method_cell_pass": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "publisher_started": False,
        "frozen_at": _utc_now(),
    }
    receipt["freeze_sha256"] = stable_sha256(receipt)
    return receipt


def _open_deferred_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """First permitted calibration/test filesystem access in this owner."""

    environment = spec["postprocess_environment"]
    identities: dict[str, Any] = {}
    for key in DEFERRED_INPUT_KEYS:
        path = Path(environment[key])
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise T14RouteCContinuationError(f"deferred input is absent or indirect: {key}")
        identities[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "schema_version": "tastemolnet_t14_route_c_deferred_inputs_v1",
        "status": "OPENED_AFTER_GENERATION_FREEZE",
        "inputs": identities,
        "generation_pass_is_method_cell_pass": False,
        "opened_at": _utc_now(),
    }


def _postprocess_environment(spec: Mapping[str, Any], *, resume: bool) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TASTEMOLNET_")
        and not key.startswith("T14_")
        and key not in POSTPROCESS_FIXED_ENVIRONMENT
    }
    environment.update(spec["postprocess_environment"])
    environment.update(
        {
            "TASTEMOLNET_T14_GENERATION_ROOT": spec["generation_root"],
            "TASTEMOLNET_T14_POSTPROCESS_ROOT": spec["postprocess_science_root"],
            "TASTEMOLNET_T14_FINAL_ROOT": spec["postprocess_final_root"],
            "TASTEMOLNET_T14_POSTPROCESS_RUN_ID": spec["postprocess_run_id"],
            "TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX": str(GPU_INDEX),
            "TASTEMOLNET_T14_POSTPROCESS_RESUME": "1" if resume else "0",
            "RUN_GNN_ABLATION": "0",
            "RUN_LLM_ABLATION": "0",
        }
    )
    return environment


def _run_postprocess(
    spec: Mapping[str, Any],
    *,
    root: Path,
    sleep: Callable[[float], None],
) -> None:
    science = Path(spec["postprocess_science_root"])
    final = Path(spec["postprocess_final_root"])
    writer_lock_path = root / "postprocess_writer.lock"
    while True:
        writer_descriptor = os.open(
            writer_lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            try:
                fcntl.flock(
                    writer_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                os.close(writer_descriptor)
                writer_descriptor = -1
                _heartbeat(
                    root,
                    spec,
                    state="WAITING_FOR_PRIOR_POSTPROCESS_WRITER",
                    child_pid=None,
                )
                sleep(float(spec["poll_seconds"]))
                continue
            if final.is_dir() and not final.is_symlink():
                return
            if final.exists() or final.is_symlink():
                raise T14RouteCContinuationError(
                    "T14 final root is not a physical directory"
                )
            if science.exists() or science.is_symlink():
                if (
                    science.is_symlink()
                    or not science.is_dir()
                    or not (science / "postprocess_checkpoint.json").is_file()
                ):
                    raise T14RouteCContinuationError(
                        "partial postprocess root is not resumable"
                    )
                resume = True
            else:
                resume = False
            environment = _postprocess_environment(spec, resume=resume)
            with (root / "postprocess.out").open("ab", buffering=0) as stdout, (
                root / "postprocess.err"
            ).open("ab", buffering=0) as stderr:
                child = subprocess.Popen(
                    ["bash", spec["postprocess_wrapper"]],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    start_new_session=False,
                    # If this continuation is hard-killed, the exact wrapper
                    # and its verifier retain the lock until they exit.  A
                    # relaunched owner therefore cannot become a second
                    # writer while the first child still owns this root.
                    pass_fds=(writer_descriptor,),
                )
            # The child now owns the shared open-file description.  Closing
            # only this parent's copy intentionally leaves its flock live.
            os.close(writer_descriptor)
            writer_descriptor = -1
        finally:
            if writer_descriptor >= 0:
                os.close(writer_descriptor)
        while child.poll() is None:
            _heartbeat(
                root,
                spec,
                state="POSTPROCESS_AND_FINAL_VERIFY_RUNNING",
                child_pid=child.pid,
                calibration_loaded=True,
                test_access_owned_by_existing_postprocessor=True,
            )
            sleep(float(spec["poll_seconds"]))
        return_code = int(child.wait())
        if return_code == WAITING_EXIT and not science.exists() and not final.exists():
            _heartbeat(root, spec, state="WAITING_FOR_IDLE_GPU2", child_pid=None)
            sleep(float(spec["poll_seconds"]))
            continue
        if return_code != 0:
            raise T14RouteCContinuationError(
                f"existing T14 postprocess/verifier failed with exit {return_code}"
            )
        return


def _validate_final(spec: Mapping[str, Any]) -> dict[str, Any]:
    final = Path(spec["postprocess_final_root"])
    marker = final / "PASS"
    if (
        final.is_symlink()
        or not final.is_dir()
        or marker.is_symlink()
        or not marker.is_file()
        or marker.read_bytes() != FINAL_PASS_BYTES
    ):
        raise T14RouteCContinuationError("T14 independent final PASS is absent")
    manifest = _physical_json(final / "run_manifest.json", label="T14 final manifest")
    audit = _physical_json(
        final / "final_artifact_audit.json", label="T14 final artifact audit"
    )
    if (
        manifest.get("status") != "PASS"
        or manifest.get("dataset") != DATASET
        or manifest.get("method") != METHOD
        or manifest.get("source_generation_root") != spec["generation_root"]
        or manifest.get("selection_frozen_before_test") is not True
        or manifest.get("test_used_for_selection") is not False
        or audit.get("status") != "PASS"
        or audit.get("independent_verifier") is not True
        or audit.get("checks", {}).get("selection_frozen_before_test") is not True
    ):
        raise T14RouteCContinuationError("T14 final verifier closure changed")
    return {
        "terminal_root": str(final),
        "pass_sha256": file_sha256(marker),
        "run_manifest_sha256": file_sha256(final / "run_manifest.json"),
        "final_artifact_audit_sha256": file_sha256(
            final / "final_artifact_audit.json"
        ),
    }


def _publish_locator(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["matrix"]["locator_path"])
    payload = {
        "schema_version": LOCATOR_SCHEMA,
        "status": "READY",
        "dataset": DATASET,
        "method": METHOD,
        "terminal_root": spec["postprocess_final_root"],
    }
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _physical_json(path, label="T14 matrix locator") != payload:
            raise T14RouteCContinuationError("T14 matrix locator already differs")
        return payload
    _atomic_json(path, payload)
    return payload


def _matrix_result(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    matrix = spec["matrix"]
    state_path = Path(matrix["authority_state_path"])
    lock_path = Path(matrix["authority_lock_path"])
    if not state_path.is_file() or state_path.is_symlink():
        return None
    pointer = read_authority_pointer(
        state_path=state_path,
        lock_path=lock_path,
        initial_authority_root=None,
    )
    if CELL_ID not in pointer["applied_cells"]:
        return None
    authority = _verify_authority(pointer["latest_authority_root"])
    row = authority["rows"][(DATASET, METHOD)]
    final = Path(spec["postprocess_final_root"]).resolve(strict=True)
    standardized = Path(str(row.get("standardized_output_root") or ""))
    try:
        selected = standardized.resolve(strict=True)
    except OSError as exc:
        raise T14RouteCContinuationError(
            "matrix authority TasteMolNet/ComRecGC root is absent"
        ) from exc
    if selected != final:
        raise T14RouteCContinuationError(
            "matrix authority already binds a different TasteMolNet/ComRecGC terminal"
        )
    return {
        "authority_root": str(authority["root"]),
        "matrix_complete_cells": int(authority["complete"]),
        "matrix_status_sha256": authority["matrix_sha256"],
        "combined_audit_sha256": authority["combined_sha256"],
        "standardized_output_root": str(selected),
    }


def _validate_unique_publisher(
    spec: Mapping[str, Any], *, proc_root: Path
) -> dict[str, Any] | None:
    # Multiple publishers may serialize unrelated T12/T13 cells through the
    # same state/lock.  Uniqueness here is deliberately scoped to the live
    # queue claiming TasteMolNet/ComRecGC.
    publishers = [
        _publisher_queue_claim(row)
        for row in find_fast16_publishers(proc_root=proc_root)
    ]
    matching = [row for row in publishers if row["claims_t14"]]
    if len(matching) > 1:
        raise T14RouteCContinuationError(
            "multiple live fast16 publishers claim TasteMolNet/ComRecGC"
        )
    if not matching:
        return None
    publisher = matching[0]
    matrix = spec["matrix"]
    pid_path = Path(matrix["queue_pid_file"])
    if pid_path.is_symlink() or not pid_path.is_file():
        raise T14RouteCContinuationError("fast16 publisher PID file is absent")
    try:
        recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise T14RouteCContinuationError("fast16 publisher PID file is invalid") from exc
    if (
        recorded_pid != publisher["pid"]
        or publisher["queue_manifest"] != matrix["queue_manifest"]
        or publisher["queue_manifest_sha256"] != matrix["queue_manifest_sha256"]
        or publisher["heartbeat_path"] != matrix["queue_heartbeat"]
    ):
        raise T14RouteCContinuationError("fast16 publisher process identity changed")
    heartbeat = _physical_json(
        Path(matrix["queue_heartbeat"]), label="fast16 publisher heartbeat"
    )
    if (
        heartbeat.get("schema_version") != QUEUE_HEARTBEAT_SCHEMA
        or heartbeat.get("pid") != recorded_pid
        or heartbeat.get("queue_manifest_path") != matrix["queue_manifest"]
        or heartbeat.get("queue_manifest_sha256") != matrix["queue_manifest_sha256"]
        or heartbeat.get("authority_state_path") != matrix["authority_state_path"]
        or heartbeat.get("authority_lock_path") != matrix["authority_lock_path"]
    ):
        raise T14RouteCContinuationError("fast16 publisher heartbeat binding changed")
    cell = heartbeat.get("cells", {}).get(CELL_ID, {})
    if isinstance(cell, Mapping) and cell.get("state") == "BLOCKED_TERMINAL_VALIDATION":
        raise T14RouteCContinuationError(
            "fast16 publisher rejected the immutable T14 final: "
            + str(cell.get("error") or "unknown error")
        )
    return publisher


def run_continuation(
    continuation_spec_path: Path,
    *,
    once: bool = False,
    proc_root: Path = Path("/proc"),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the persistent handoff; return only after matrix adoption or ``once``."""

    spec = load_continuation_spec(Path(continuation_spec_path))
    root = Path(spec["continuation_root"])
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise T14RouteCContinuationError("continuation root cannot be a symlink")
    lock_path = root / "owner.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    previous = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise T14RouteCContinuationError(
                "Route C continuation already has one owner"
            ) from exc
        handoff = validate_generation_handoff(
            Path(spec["generation_handoff"]), spec=spec
        )
        _validate_owner_terminal(spec, handoff)
        _atomic_json(
            root / "owner.json",
            {
                "schema_version": OWNER_SCHEMA,
                "status": "OWNER_CONFIRMED",
                "pid": os.getpid(),
                "continuation_spec": spec["descriptor_path"],
                "continuation_spec_sha256": spec["spec_sha256"],
                "generation_handoff_sha256": handoff["handoff_sha256"],
                "publisher_started": False,
                "started_at": _utc_now(),
            },
        )
        while _process_alive(
            handoff["generation_owner_pid"],
            handoff["generation_owner_start_ticks"],
            proc_root=proc_root,
        ):
            _heartbeat(root, spec, state="WAITING_FOR_GENERATION_OWNER_EXIT")
            if once or stopped:
                return {"status": "WAITING_FOR_GENERATION_OWNER_EXIT"}
            sleep(float(spec["poll_seconds"]))
        _validate_owner_terminal(spec, handoff)
        freeze_path = root / "generation_freeze.json"
        if freeze_path.is_file() and not freeze_path.is_symlink():
            freeze = _physical_json(freeze_path, label="generation freeze receipt")
            _self_hash(freeze, field="freeze_sha256")
        else:
            freeze = _generation_is_frozen(spec, handoff, proc_root=proc_root)
            _atomic_json(freeze_path, freeze)
        if freeze.get("status") != "GENERATION_FROZEN_POSTPROCESS_INPUTS_UNOPENED":
            raise T14RouteCContinuationError("generation freeze receipt changed")
        if (
            freeze.get("generation_root") != spec["generation_root"]
            or freeze.get("generation_inventory_sha256")
            != handoff.get("generation_inventory_sha256")
            or freeze.get("generation_pass_is_method_cell_pass") is not False
            or freeze.get("calibration_loaded") is not False
            or freeze.get("test_loaded") is not False
            or freeze.get("publisher_started") is not False
        ):
            raise T14RouteCContinuationError("generation freeze binding changed")
        _heartbeat(
            root,
            spec,
            state="GENERATION_FROZEN_POSTPROCESS_INPUTS_UNOPENED",
            calibration_loaded=False,
            test_loaded=False,
        )

        # This is intentionally the first access to either held-out path.
        deferred = _open_deferred_inputs(spec)
        _atomic_json(root / "deferred_inputs.json", deferred)
        _run_postprocess(spec, root=root, sleep=sleep)
        final = _validate_final(spec)
        _atomic_json(root / "final_verifier.json", final)
        locator = _publish_locator(spec)
        _atomic_json(root / "locator_receipt.json", locator)

        while not stopped:
            applied = _matrix_result(spec)
            if applied is not None:
                terminal: dict[str, Any] = {
                    "schema_version": TERMINAL_SCHEMA,
                    "status": "PASS",
                    "generation_root": spec["generation_root"],
                    "generation_pass_was_method_cell_pass": False,
                    "postprocess_final_root": spec["postprocess_final_root"],
                    "matrix_locator": spec["matrix"]["locator_path"],
                    "matrix": applied,
                    "paper_result_eligible": True,
                    "method_cell_pass": True,
                    "publisher_started": False,
                    "completed_at": _utc_now(),
                }
                terminal["terminal_sha256"] = stable_sha256(terminal)
                _atomic_json(root / "terminal.json", terminal)
                _heartbeat(
                    root,
                    spec,
                    state="MATRIX_CELL_PASS",
                    matrix_complete_cells=applied["matrix_complete_cells"],
                    method_cell_pass=True,
                )
                return terminal
            publisher = _validate_unique_publisher(spec, proc_root=proc_root)
            _heartbeat(
                root,
                spec,
                state=(
                    "WAITING_FOR_FAST16_MATRIX_ADOPTION"
                    if publisher is not None
                    else "WAITING_FOR_UNIQUE_FAST16_PUBLISHER"
                ),
                publisher_pid=(None if publisher is None else publisher["pid"]),
                method_cell_pass=False,
            )
            if once:
                return {
                    "status": (
                        "WAITING_FOR_FAST16_MATRIX_ADOPTION"
                        if publisher is not None
                        else "WAITING_FOR_UNIQUE_FAST16_PUBLISHER"
                    ),
                    "method_cell_pass": False,
                    "publisher_started": False,
                }
            sleep(float(spec["poll_seconds"]))
        return {
            "status": "STOPPED_PENDING_MATRIX_ADOPTION",
            "method_cell_pass": False,
            "publisher_started": False,
        }
    except BaseException as exc:
        _atomic_json(
            root / "terminal.json",
            {
                "schema_version": TERMINAL_SCHEMA,
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
                "generation_pass_was_method_cell_pass": False,
                "paper_result_eligible": False,
                "method_cell_pass": False,
                "publisher_started": False,
                "failed_at": _utc_now(),
            },
        )
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "CELL_ID",
    "DEFERRED_INPUT_KEYS",
    "HANDOFF_SCHEMA",
    "SPEC_SCHEMA",
    "T14RouteCContinuationError",
    "build_continuation_spec",
    "continuation_command",
    "find_fast16_publishers",
    "launch_continuation_owner",
    "load_continuation_spec",
    "publish_generation_handoff",
    "require_unpublished_matrix_cell",
    "run_continuation",
    "stable_sha256",
    "validate_continuation_spec",
    "validate_generation_handoff",
    "write_continuation_spec",
]
