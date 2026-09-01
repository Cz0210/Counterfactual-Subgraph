"""Exact BACE ComRecGC 20k/25k handover and checkpoint finalization.

This is intentionally a dataset-specific executor.  It consumes the existing
read-only resource-cap observer request, reopens one fully committed generation
checkpoint, proves the exact live process generation, sends SIGTERM to that PID
only, and materializes the checkpoint into a fresh generation root.  It never
uses calibration or test data to decide whether generation may stop.

The original 50k scientific command is retained as provenance.  The resource
cap is an explicitly authorized terminal policy layered on top of its exact
completed-step checkpoint; no random-walk step is replayed or recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time
from typing import Any, Mapping, Sequence

from src.baselines.bace_gnn_baseline_tasks import (
    build_bace_baseline_controller_fragment,
)
from src.baselines.comrecgc.contracts import (
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.baselines.comrecgc.frozen_payload import payload_file_audit
from src.baselines.comrecgc.generation_checkpoint import load_generation_checkpoint
from src.baselines.comrecgc.graph_trace import ActionTraceRecorder
from src.baselines.comrecgc.project_dataset import load_bace_generation_bundle
from src.utils.autodl_runtime import atomic_write_json


EXECUTOR_SCHEMA = "bace_comrecgc_resource_cap_executor_v1"
RECEIPT_SCHEMA = "bace_comrecgc_resource_cap_receipt_v1"
MATERIALIZATION_SCHEMA = "bace_comrecgc_resource_cap_materialization_v1"
POSTPROCESS_SCHEMA = "bace_comrecgc_resource_cap_postprocess_fragment_v1"
LIVENESS_SCHEMA = "bace_comrecgc_liveness_diagnostic_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ELIGIBLE_STATES = frozenset(
    {
        "HANDOVER_ELIGIBLE",
        "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP",
    }
)


class BaceComRecGCResourceCapExecutorError(RuntimeError):
    """The exact process/checkpoint handover cannot proceed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _absolute(
    value: str | Path,
    *,
    label: str,
    existing: bool = False,
    directory: bool | None = None,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BaceComRecGCResourceCapExecutorError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=existing)
    except FileNotFoundError as exc:
        raise BaceComRecGCResourceCapExecutorError(
            f"{label} does not exist: {path}"
        ) from exc
    if existing and directory is True and not resolved.is_dir():
        raise BaceComRecGCResourceCapExecutorError(
            f"{label} must be a physical directory: {resolved}"
        )
    if existing and directory is False and not resolved.is_file():
        raise BaceComRecGCResourceCapExecutorError(
            f"{label} must be a physical file: {resolved}"
        )
    if existing and path.is_symlink():
        raise BaceComRecGCResourceCapExecutorError(
            f"{label} must not be a symbolic link: {path}"
        )
    return resolved


def _object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaceComRecGCResourceCapExecutorError(
            f"invalid {label} JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BaceComRecGCResourceCapExecutorError(
            f"{label} must be one JSON object: {path}"
        )
    return dict(value)


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


@dataclass(frozen=True, slots=True)
class ProcessContract:
    pid: int
    start_ticks: int
    cmdline_sha256: str
    cwd: str
    output_root: str
    controller_id: str
    controller_receipt: str
    controller_receipt_sha256: str
    expected_ppid: int | None = None

    def validate(self) -> None:
        if isinstance(self.pid, bool) or int(self.pid) <= 1:
            raise BaceComRecGCResourceCapExecutorError("science PID is invalid")
        if isinstance(self.start_ticks, bool) or int(self.start_ticks) <= 0:
            raise BaceComRecGCResourceCapExecutorError(
                "science PID start ticks are invalid"
            )
        for label, value in (
            ("cmdline_sha256", self.cmdline_sha256),
            ("controller_receipt_sha256", self.controller_receipt_sha256),
        ):
            if SHA256_RE.fullmatch(str(value)) is None:
                raise BaceComRecGCResourceCapExecutorError(f"{label} is invalid")
        if not str(self.controller_id).strip():
            raise BaceComRecGCResourceCapExecutorError("controller_id is empty")
        if self.expected_ppid is not None and (
            isinstance(self.expected_ppid, bool) or int(self.expected_ppid) <= 0
        ):
            raise BaceComRecGCResourceCapExecutorError("expected_ppid is invalid")
        _absolute(self.cwd, label="science cwd", existing=True, directory=True)
        _absolute(
            self.output_root,
            label="science output root",
            existing=True,
            directory=True,
        )
        _absolute(
            self.controller_receipt,
            label="controller receipt",
            existing=True,
            directory=False,
        )


def _parse_proc_stat(path: Path) -> dict[str, int | str]:
    raw = path.read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing <= 0:
        raise BaceComRecGCResourceCapExecutorError("proc stat is malformed")
    fields = raw[closing + 2 :].split()
    if len(fields) < 22:
        raise BaceComRecGCResourceCapExecutorError("proc stat is incomplete")
    return {
        "state": fields[0],
        "ppid": int(fields[1]),
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "start_ticks": int(fields[19]),
    }


def _recursive_contains(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_recursive_contains(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_recursive_contains(item, expected) for item in value)
    return False


def _recursive_process_binding(value: Any, *, pid: int, start_ticks: int) -> bool:
    if isinstance(value, Mapping):
        values = tuple(value.values())
        if int(pid) in values and int(start_ticks) in values:
            return True
        return any(
            _recursive_process_binding(item, pid=pid, start_ticks=start_ticks)
            for item in values
        )
    if isinstance(value, (list, tuple)):
        return any(
            _recursive_process_binding(item, pid=pid, start_ticks=start_ticks)
            for item in value
        )
    return False


def verify_exact_process(
    contract: ProcessContract,
    *,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Reopen every process identity immediately before a possible SIGTERM."""

    contract.validate()
    proc = _absolute(proc_root, label="proc root", existing=True, directory=True)
    process = proc / str(int(contract.pid))
    stat = _parse_proc_stat(process / "stat")
    if stat["state"] in {"Z", "X", "x"}:
        raise BaceComRecGCResourceCapExecutorError(
            f"science process is not live: state={stat['state']}"
        )
    if int(stat["start_ticks"]) != int(contract.start_ticks):
        raise BaceComRecGCResourceCapExecutorError(
            "science PID generation start ticks changed"
        )
    if contract.expected_ppid is not None and int(stat["ppid"]) != int(
        contract.expected_ppid
    ):
        raise BaceComRecGCResourceCapExecutorError("science parent PID changed")
    cmdline = (process / "cmdline").read_bytes()
    cmdline_sha = hashlib.sha256(cmdline).hexdigest()
    if cmdline_sha != contract.cmdline_sha256:
        raise BaceComRecGCResourceCapExecutorError("science command line changed")
    argv = [part.decode("utf-8", errors="strict") for part in cmdline.split(b"\0") if part]
    try:
        output_index = argv.index("--output-dir")
        argv_output = _absolute(
            argv[output_index + 1],
            label="process --output-dir",
            existing=True,
            directory=True,
        )
    except (IndexError, ValueError) as exc:
        raise BaceComRecGCResourceCapExecutorError(
            "science command does not contain a complete --output-dir"
        ) from exc
    expected_output = _absolute(
        contract.output_root,
        label="science output root",
        existing=True,
        directory=True,
    )
    if argv_output != expected_output:
        raise BaceComRecGCResourceCapExecutorError(
            "science command output root changed"
        )
    observed_cwd = Path(os.readlink(process / "cwd")).resolve(strict=True)
    expected_cwd = _absolute(
        contract.cwd, label="science cwd", existing=True, directory=True
    )
    if observed_cwd != expected_cwd:
        raise BaceComRecGCResourceCapExecutorError("science cwd changed")
    controller_path = _absolute(
        contract.controller_receipt,
        label="controller receipt",
        existing=True,
        directory=False,
    )
    if sha256_file(controller_path) != contract.controller_receipt_sha256:
        raise BaceComRecGCResourceCapExecutorError("controller receipt changed")
    controller = _object(controller_path, label="controller receipt")
    if not _recursive_contains(controller, contract.controller_id):
        raise BaceComRecGCResourceCapExecutorError(
            "controller receipt does not bind controller_id"
        )
    if not _recursive_process_binding(
        controller,
        pid=int(contract.pid),
        start_ticks=int(contract.start_ticks),
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "controller receipt does not bind the science PID generation"
        )
    return {
        "pid": int(contract.pid),
        "start_ticks": int(contract.start_ticks),
        "state": str(stat["state"]),
        "ppid": int(stat["ppid"]),
        "utime_ticks": int(stat["utime_ticks"]),
        "stime_ticks": int(stat["stime_ticks"]),
        "cmdline_sha256": cmdline_sha,
        "argv": argv,
        "cwd": str(observed_cwd),
        "output_root": str(argv_output),
        "controller_id": contract.controller_id,
        "controller_receipt": str(controller_path),
        "controller_receipt_sha256": contract.controller_receipt_sha256,
        "verified_at": utc_now(),
    }


def classify_liveness(
    observations: Sequence[Mapping[str, Any]],
    *,
    stalled_after_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Classify BACE generation without mistaking a stale progress file for stall.

    Positive CPU time, output growth, a progress/checkpoint advance, or an
    explicit in-flight atomic checkpoint makes the route ``RUNNING_SLOW``.
    ``STALLED`` requires a full one-hour observation span with all four science
    indicators unchanged and no checkpoint fsync/rename evidence.
    """

    if len(observations) < 2:
        return {
            "schema_version": LIVENESS_SCHEMA,
            "state": "INSUFFICIENT_OBSERVATIONS",
            "signal_allowed": False,
            "observation_count": len(observations),
        }
    rows = [dict(row) for row in observations]
    required = {
        "observed_monotonic",
        "pid",
        "start_ticks",
        "cpu_ticks",
        "progress_step",
        "checkpoint_step",
        "output_bytes",
        "checkpoint_write_in_progress",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise BaceComRecGCResourceCapExecutorError(
                f"liveness observation is incomplete: {missing}"
            )
    if any(
        int(row["pid"]) != int(rows[0]["pid"])
        or int(row["start_ticks"]) != int(rows[0]["start_ticks"])
        for row in rows[1:]
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "liveness observations cross process generations"
        )
    times = [float(row["observed_monotonic"]) for row in rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise BaceComRecGCResourceCapExecutorError(
            "liveness observation time is not strictly increasing"
        )
    fields = ("cpu_ticks", "progress_step", "checkpoint_step", "output_bytes")
    for field in fields:
        values = [int(row[field]) for row in rows]
        if any(right < left for left, right in zip(values, values[1:])):
            raise BaceComRecGCResourceCapExecutorError(
                f"liveness counter regressed: {field}"
            )
    deltas = {field: int(rows[-1][field]) - int(rows[0][field]) for field in fields}
    span = times[-1] - times[0]
    checkpoint_busy = any(bool(row["checkpoint_write_in_progress"]) for row in rows)
    positive_science = any(value > 0 for value in deltas.values())
    if positive_science or checkpoint_busy:
        state = "RUNNING_SLOW"
    elif span >= float(stalled_after_seconds):
        state = "STALLED"
    else:
        state = "STALL_NOT_PROVEN"
    return {
        "schema_version": LIVENESS_SCHEMA,
        "state": state,
        "signal_allowed": state == "STALLED",
        "observation_count": len(rows),
        "observation_span_seconds": span,
        "deltas": deltas,
        "checkpoint_write_observed": checkpoint_busy,
        "pid": int(rows[0]["pid"]),
        "start_ticks": int(rows[0]["start_ticks"]),
    }


def _copy_checkpoint_trace_chunks(
    *,
    trace_state: Mapping[str, Any],
    source_trace_root: Path,
    destination_trace_root: Path,
) -> None:
    if destination_trace_root.exists() or destination_trace_root.is_symlink():
        raise FileExistsError(
            f"fresh trace root already exists: {destination_trace_root}"
        )
    chunks = trace_state.get("chunks")
    if not isinstance(chunks, list):
        raise BaceComRecGCResourceCapExecutorError(
            "checkpoint trace chunk inventory is missing"
        )
    chunk_root = destination_trace_root / "selected_action_trace_chunks"
    chunk_root.mkdir(parents=True, exist_ok=False)
    for expected_index, row in enumerate(chunks):
        if not isinstance(row, Mapping) or int(row.get("index", -1)) != expected_index:
            raise BaceComRecGCResourceCapExecutorError(
                "checkpoint trace chunk order is invalid"
            )
        relative = Path(str(row.get("path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parent != Path("selected_action_trace_chunks")
        ):
            raise BaceComRecGCResourceCapExecutorError(
                "checkpoint trace chunk path is unsafe"
            )
        source = source_trace_root / relative
        destination = destination_trace_root / relative
        if not source.is_file() or source.is_symlink():
            raise BaceComRecGCResourceCapExecutorError(
                f"checkpoint trace chunk is missing: {source}"
            )
        if source.stat().st_size != int(row.get("bytes", -1)) or sha256_file(
            source
        ) != str(row.get("sha256")):
            raise BaceComRecGCResourceCapExecutorError(
                f"checkpoint trace chunk identity changed: {source}"
            )
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256_file(destination) != str(row["sha256"]):
            raise BaceComRecGCResourceCapExecutorError(
                f"copied trace chunk failed SHA256 verification: {destination}"
            )


def _torch_save_atomic(payload: Mapping[str, Any], path: Path) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - production dependency
        raise BaceComRecGCResourceCapExecutorError(
            "checkpoint materialization requires PyTorch"
        ) from exc
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_request(
    request: Mapping[str, Any],
    *,
    checkpoint_digest: str,
    completed_step: int,
) -> dict[str, Any]:
    status = str(request.get("status") or "")
    if status not in ELIGIBLE_STATES:
        raise BaceComRecGCResourceCapExecutorError(
            f"resource-cap request is not eligible: {status or 'MISSING'}"
        )
    if int(request.get("m_effective", -1)) != int(completed_step):
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap request/checkpoint step differs"
        )
    if request.get("checkpoint_digest") != checkpoint_digest:
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap request/checkpoint digest differs"
        )
    unique = request.get("valid_unique_count")
    lineage = request.get("lineage_error_count")
    if isinstance(unique, bool) or not isinstance(unique, int) or unique < 0:
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap valid_unique_count is invalid"
        )
    if isinstance(lineage, bool) or not isinstance(lineage, int) or lineage < 0:
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap lineage_error_count is invalid"
        )
    reason = str(request.get("reason") or "")
    if completed_step % 2_500 != 0:
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap request is not on a registered checkpoint boundary"
        )
    if status == "HANDOVER_ELIGIBLE":
        if lineage != 0 or unique < 10:
            raise BaceComRecGCResourceCapExecutorError(
                "eligible resource cap lacks ten clean unique rules"
            )
        if completed_step < 10_000 or completed_step > 25_000:
            raise BaceComRecGCResourceCapExecutorError(
                "eligible resource cap step is outside authorized bounds"
            )
        expected_reason = (
            "PREREGISTERED_CONVERGENCE_PASS"
            if completed_step < 20_000
            else (
                "RESOURCE_CAP_20000"
                if completed_step == 20_000
                else "RESOURCE_CAP_25000_FALLBACK"
            )
        )
        if reason != expected_reason:
            raise BaceComRecGCResourceCapExecutorError(
                "eligible resource-cap reason differs from the registered policy"
            )
    else:
        if completed_step != 25_000:
            raise BaceComRecGCResourceCapExecutorError(
                "scientific failure stop is allowed only at the 25k absolute cap"
            )
        if reason != "ABSOLUTE_CAP_INSUFFICIENT_RULES_OR_LINEAGE_ERRORS":
            raise BaceComRecGCResourceCapExecutorError(
                "scientific-failure reason differs from the registered policy"
            )
        if lineage == 0 and unique >= 10:
            raise BaceComRecGCResourceCapExecutorError(
                "scientific failure contradicts the clean ten-rule handover gate"
            )
    return {
        "status": status,
        "reason": reason,
        "m_effective": int(completed_step),
        "valid_unique_count": int(unique),
        "lineage_error_count": int(lineage),
    }


def materialize_resource_cap_checkpoint(
    *,
    checkpoint_dir: str | Path,
    expected_checkpoint_digest: str,
    source_trace_root: str | Path,
    source_resolved_config: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    resource_cap_receipt: Mapping[str, Any],
    loaded_checkpoint: Any | None = None,
) -> dict[str, Any]:
    """Finalize one committed checkpoint without running another walk step."""

    checkpoint = _absolute(
        checkpoint_dir,
        label="checkpoint_dir",
        existing=True,
        directory=True,
    )
    trace_source = _absolute(
        source_trace_root,
        label="source_trace_root",
        existing=True,
        directory=True,
    )
    config_path = _absolute(
        source_resolved_config,
        label="source_resolved_config",
        existing=True,
        directory=False,
    )
    dataset = _absolute(dataset_dir, label="dataset_dir", existing=True, directory=True)
    output = _absolute(output_dir, label="output_dir")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"resource-cap output must be fresh: {output}")
    if SHA256_RE.fullmatch(str(expected_checkpoint_digest)) is None:
        raise BaceComRecGCResourceCapExecutorError(
            "expected checkpoint digest is invalid"
        )
    loaded = (
        loaded_checkpoint
        if loaded_checkpoint is not None
        else load_generation_checkpoint(
            checkpoint,
            expected_completed_step=int(resource_cap_receipt["m_effective"]),
        )
    )
    validation = loaded.validation
    if validation.checkpoint_dir.resolve(strict=True) != checkpoint:
        raise BaceComRecGCResourceCapExecutorError(
            "preloaded checkpoint directory differs from the requested checkpoint"
        )
    if loaded.completed_step != int(resource_cap_receipt["m_effective"]):
        raise BaceComRecGCResourceCapExecutorError(
            "preloaded checkpoint step differs from the resource-cap receipt"
        )
    if validation.checkpoint_digest != expected_checkpoint_digest:
        raise BaceComRecGCResourceCapExecutorError("checkpoint digest changed")
    if validation.total_steps != 50_000:
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap source must be the original BACE 50k trajectory"
        )
    request = _validate_request(
        resource_cap_receipt,
        checkpoint_digest=validation.checkpoint_digest,
        completed_step=validation.completed_step,
    )
    if request["status"] != "HANDOVER_ELIGIBLE":
        raise BaceComRecGCResourceCapExecutorError(
            "scientific-failure checkpoint cannot be materialized as a result"
        )
    config = _object(config_path, label="source resolved config")
    config_sha = stable_json_sha256(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    if config.get("config_sha256") != config_sha:
        raise BaceComRecGCResourceCapExecutorError(
            "source resolved config content hash differs"
        )
    if (
        config.get("dataset") != "bace"
        or config.get("mode") != "full"
        or int(config.get("parent_limit", -1)) != 360
        or int(config.get("total_steps", -1)) != 50_000
        or config.get("checkpoint_provenance")
        != validation.provenance_fingerprints
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "source resolved config differs from BACE full checkpoint provenance"
        )
    output.mkdir(parents=True, exist_ok=False)
    trace_output = output / "trace"
    _copy_checkpoint_trace_chunks(
        trace_state=loaded.trace_state,
        source_trace_root=trace_source,
        destination_trace_root=trace_output,
    )
    official = loaded.algorithm_state.get("official_state")
    if not isinstance(official, Mapping):
        raise BaceComRecGCResourceCapExecutorError(
            "checkpoint official runtime state is missing"
        )
    required_official = {
        "graph_map",
        "graph_index_map",
        "counterfactual_candidates",
        "MAX_COUNTERFACTUAL_SIZE",
        "traversed_hashes",
        "input_graphs_covered",
    }
    missing = sorted(required_official - set(official))
    if missing:
        raise BaceComRecGCResourceCapExecutorError(
            f"checkpoint official state is incomplete: {missing}"
        )
    payload: dict[str, Any] = {
        key: official[key] for key in sorted(required_official)
    }
    bundle = load_bace_generation_bundle(dataset_dir=dataset, parent_limit=360)
    if stable_json_sha256(bundle.parent_ids) != validation.provenance_fingerprints.get(
        "generation_parent_ids_sha256"
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "BACE parent order differs from checkpoint provenance"
        )
    trace_state = loaded.trace_state
    recorder = ActionTraceRecorder(
        output_dir=trace_output,
        chunk_size=int(trace_state.get("chunk_size", 512)),
        compact_enumeration=bool(trace_state.get("compact_enumeration")),
    )
    recorder.restore_checkpoint_state(trace_state)
    closure_audit = output / "frozen_payload_closure_audit.json"
    trace_summary = recorder.write(
        trace_output,
        payload,
        source_graphs_by_parent_id=dict(
            zip(bundle.parent_ids, bundle.graphs, strict=True)
        ),
        compact_candidate_lineage=True,
        frozen_payload_backing_store=loaded.sqlite_snapshot_path,
        frozen_payload_audit_path=closure_audit,
    )
    result_path = output / "counterfactuals.pt"
    _torch_save_atomic(payload, result_path)
    result_audit = payload_file_audit(result_path)
    candidate_count = len(payload.get("counterfactual_candidates") or ())
    visited_count = len(payload.get("graph_map") or {})
    traversed_count = len(payload.get("traversed_hashes") or ())
    if traversed_count != validation.completed_step:
        raise BaceComRecGCResourceCapExecutorError(
            "materialized traversal length differs from checkpoint step"
        )
    stop_reason = str(request["reason"])
    early_stop_used = stop_reason == "PREREGISTERED_CONVERGENCE_PASS"
    cap_used = stop_reason.startswith("RESOURCE_CAP_")
    manifest = {
        **config,
        "schema_version": MATERIALIZATION_SCHEMA,
        "counterfactuals_path": str(result_path),
        "counterfactuals_sha256": sha256_file(result_path),
        "counterfactuals_bytes": result_path.stat().st_size,
        "counterfactual_candidate_count": candidate_count,
        "visited_graph_count": visited_count,
        "traversed_step_count": traversed_count,
        "trace_enabled": True,
        "trace_summary": trace_summary,
        "frozen_payload_closure_audit_path": str(closure_audit),
        "frozen_payload_closure_audit_sha256": sha256_file(closure_audit),
        "frozen_payload_closure": trace_summary.get("frozen_payload_closure"),
        "counterfactual_payload_audit": result_audit,
        "candidate_order_source": "official_frequency_reinforced_order",
        "algorithm_rerun": False,
        "materialized_from_checkpoint": str(validation.checkpoint_dir),
        "checkpoint_digest": validation.checkpoint_digest,
        "checkpoint_reload_pass": True,
        "original_generation_configured_steps": 50_000,
        "M_configured_max": 20_000,
        "M_fallback_max": 25_000,
        "M_effective": validation.completed_step,
        "resource_cap_used": cap_used,
        "early_stop_used": early_stop_used,
        "stop_reason": stop_reason,
        "valid_unique_rule_count": request["valid_unique_count"],
        "lineage_error_count": request["lineage_error_count"],
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
        "completed_at": utc_now(),
    }
    write_json(output / "run_manifest.json", manifest)
    write_json(
        output / "progress.json",
        {
            "schema_version": MATERIALIZATION_SCHEMA,
            "current_step": validation.completed_step,
            "completed_step": validation.completed_step,
            "next_step": validation.completed_step + 1,
            "max_steps": 20_000,
            "original_max_steps": 50_000,
            "run_complete": True,
            "resource_cap_used": cap_used,
            "early_stop_used": early_stop_used,
            "stop_reason": stop_reason,
            "checkpoint_digest": validation.checkpoint_digest,
            "updated_at": utc_now(),
        },
    )
    write_json(
        output / "resource_cap_receipt.json", dict(resource_cap_receipt)
    )
    write_json(
        output / "_RUN_COMPLETE.json",
        {
            "schema_version": MATERIALIZATION_SCHEMA,
            "run_complete": True,
            "counterfactuals_sha256": manifest["counterfactuals_sha256"],
            "M_configured_max": 20_000,
            "M_effective": validation.completed_step,
            "resource_cap_used": cap_used,
            "early_stop_used": early_stop_used,
            "stop_reason": stop_reason,
        },
    )
    return manifest


def _write_post20k_exclusion_receipts(
    *,
    state_root: Path,
    source_generation_root: str | Path,
    checkpoint: Any,
    materialized_manifest: Mapping[str, Any],
    resource_cap_receipt: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Bind the adopted 20k universe while preserving later uncommitted files."""

    configured_max = resource_cap_receipt.get("M_configured_max")
    effective_step = resource_cap_receipt.get("M_effective")
    valid_unique = resource_cap_receipt.get("valid_unique_count")
    lineage_errors = resource_cap_receipt.get("lineage_error_count")
    if (
        type(checkpoint.completed_step) is not int
        or checkpoint.completed_step != 20_000
        or type(configured_max) is not int
        or configured_max != 20_000
        or type(effective_step) is not int
        or effective_step != 20_000
        or resource_cap_receipt.get("stop_reason") != "RESOURCE_CAP_20000"
        or type(valid_unique) is not int
        or valid_unique < 10
        or type(lineage_errors) is not int
        or lineage_errors != 0
        or materialized_manifest.get("M_configured_max") != 20_000
        or materialized_manifest.get("M_effective") != 20_000
        or materialized_manifest.get("stop_reason") != "RESOURCE_CAP_20000"
        or materialized_manifest.get("resource_cap_used") is not True
        or materialized_manifest.get("early_stop_used") is not False
        or materialized_manifest.get("algorithm_rerun") is not False
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "20k exclusion receipt requires the exact clean committed-20k handover"
        )
    source = _absolute(
        source_generation_root,
        label="science output root",
        existing=True,
        directory=True,
    )
    handover_receipt_path = _absolute(
        state_root / "resource_cap_receipt.json",
        label="20k handover resource-cap receipt",
        existing=True,
        directory=False,
    )
    handover_receipt = _object(
        handover_receipt_path,
        label="20k handover resource-cap receipt",
    )
    if handover_receipt != dict(resource_cap_receipt):
        raise BaceComRecGCResourceCapExecutorError(
            "20k handover resource-cap receipt changed before formal closure"
        )
    signal_receipt_path = _absolute(
        state_root / "signal_receipt.json",
        label="20k handover signal receipt",
        existing=True,
        directory=False,
    )
    signal_receipt = _object(
        signal_receipt_path,
        label="20k handover signal receipt",
    )
    process_before = resource_cap_receipt.get("process_before_signal")
    if (
        not isinstance(process_before, Mapping)
        or signal_receipt.get("signal") != "SIGTERM"
        or signal_receipt.get("signal_number") != int(signal.SIGTERM)
        or signal_receipt.get("exited_within_wait") is not True
        or signal_receipt.get("sigkill_used") is not False
        or signal_receipt.get("exact_pid") != process_before.get("pid")
        or signal_receipt.get("exact_start_ticks")
        != process_before.get("start_ticks")
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "20k handover signal receipt does not prove exact graceful exit"
        )
    checkpoint_manifest = checkpoint.validation.manifest
    checkpoint_manifest_path = _absolute(
        checkpoint.validation.checkpoint_dir / "checkpoint_manifest.json",
        label="20k checkpoint manifest",
        existing=True,
        directory=False,
    )
    if _object(checkpoint_manifest_path, label="20k checkpoint manifest") != dict(
        checkpoint_manifest
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "20k checkpoint manifest changed after checkpoint reload"
        )
    files = checkpoint_manifest.get("files")
    generation_state = (
        files.get("generation_state.pt") if isinstance(files, Mapping) else None
    )
    rng_container_sha = (
        generation_state.get("sha256")
        if isinstance(generation_state, Mapping)
        else None
    )
    generation_state_path = _absolute(
        checkpoint.validation.checkpoint_dir / "generation_state.pt",
        label="committed 20k generation state",
        existing=True,
        directory=False,
    )
    candidate_universe_sha = materialized_manifest.get("counterfactuals_sha256")
    if (
        SHA256_RE.fullmatch(str(rng_container_sha or "")) is None
        or SHA256_RE.fullmatch(str(candidate_universe_sha or "")) is None
    ):
        raise BaceComRecGCResourceCapExecutorError(
            "20k exclusion receipt lacks checkpoint RNG or candidate-universe SHA256"
        )
    if sha256_file(generation_state_path) != rng_container_sha:
        raise BaceComRecGCResourceCapExecutorError(
            "committed 20k generation-state bytes changed after checkpoint reload"
        )
    candidate_path = _absolute(
        str(materialized_manifest.get("counterfactuals_path") or ""),
        label="materialized official 20k candidate universe",
        existing=True,
        directory=False,
    )
    if sha256_file(candidate_path) != candidate_universe_sha:
        raise BaceComRecGCResourceCapExecutorError(
            "materialized official 20k candidate-universe SHA256 changed"
        )
    progress_path = source / "progress.json"
    progress_sha = (
        sha256_file(progress_path)
        if progress_path.is_file() and not progress_path.is_symlink()
        else None
    )
    common = {
        "M_CONFIGURED_MAX": int(resource_cap_receipt["M_configured_max"]),
        "M_EFFECTIVE": int(resource_cap_receipt["M_effective"]),
        "checkpoint_dir": str(checkpoint.validation.checkpoint_dir),
        "checkpoint_SHA": str(checkpoint.validation.checkpoint_digest),
        "checkpoint_manifest_SHA256": sha256_file(checkpoint_manifest_path),
        "RNG_state_SHA": str(rng_container_sha),
        "RNG_state_SHA_scope": (
            "committed generation_state.pt container containing the exact "
            "python/numpy/torch_cpu/torch_cuda RNG state"
        ),
        "candidate_universe_SHA": str(candidate_universe_sha),
        "candidate_universe_SHA_scope": "materialized official 20k counterfactuals.pt",
        "valid_unique_rules": int(resource_cap_receipt["valid_unique_count"]),
        "lineage_errors": int(resource_cap_receipt["lineage_error_count"]),
        "stop_reason": "RESOURCE_CAP_20000",
        "post_20k_uncommitted_outputs_excluded": True,
        "post_20k_uncommitted_outputs_preserved": True,
        "handover_resource_cap_receipt": str(handover_receipt_path),
        "handover_resource_cap_receipt_SHA256": sha256_file(
            handover_receipt_path
        ),
        "handover_signal_receipt": str(signal_receipt_path),
        "handover_signal_receipt_SHA256": sha256_file(signal_receipt_path),
        "handover_exact_pid": int(signal_receipt["exact_pid"]),
        "handover_exact_start_ticks": int(signal_receipt["exact_start_ticks"]),
        "handover_graceful_exit": True,
        "sigkill_used": False,
        "source_generation_root": str(source),
        "source_progress_present_after_signal": progress_sha is not None,
        "source_progress_path": str(progress_path) if progress_sha else None,
        "source_progress_SHA256_after_signal": progress_sha,
        "formal_generation_root": str(candidate_path.parent),
        "scientific_result_adopted_through_step": 20_000,
        "later_partial_rows_adopted": False,
        "later_temporary_outputs_deleted": False,
        "created_at": utc_now(),
    }
    cap_path = state_root / "bace_comrecgc_20k_resource_cap_receipt.json"
    excluded_path = state_root / "excluded_after_20k.json"
    for path in (cap_path, excluded_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace existing receipt: {path}")
    _atomic_json_new(
        cap_path,
        {
            **common,
            "schema_version": "bace_comrecgc_20k_resource_cap_receipt_v1",
        },
    )
    _atomic_json_new(
        excluded_path,
        {
            **common,
            "schema_version": "bace_comrecgc_excluded_after_20k_v1",
        },
    )
    return cap_path, excluded_path


def build_postprocess_fragment(
    *,
    python: str | Path,
    project_root: str | Path,
    output_root: str | Path,
    gnn_checkpoint: str | Path,
    dataset_dir: str | Path,
    calibration_split: str | Path,
    test_split: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    neurosed_checkpoint: str | Path,
    official_root: str | Path,
    resource_cap_receipt: str | Path,
) -> dict[str, Any]:
    """Build only post-generation BACE tasks in one fresh namespace."""

    root = _absolute(output_root, label="output_root")
    generation = root / "train_generation"
    if not (generation / "_RUN_COMPLETE.json").is_file():
        raise BaceComRecGCResourceCapExecutorError(
            "resource-cap generation materialization is incomplete"
        )
    fragment = build_bace_baseline_controller_fragment(
        method="ComRecGC",
        python=python,
        project_root=project_root,
        output_root=root,
        gnn_checkpoint=gnn_checkpoint,
        dataset_dir=dataset_dir,
        calibration_split=calibration_split,
        test_split=test_split,
        molclr_root=molclr_root,
        molclr_checkpoint=molclr_checkpoint,
        neurosed_checkpoint=neurosed_checkpoint,
        official_root=official_root,
    )
    generation_id = "bace_comrecgc_train_generation"
    preflight_id = "bace_comrecgc_preflight"
    filtered: list[dict[str, Any]] = []
    for original in fragment["tasks"]:
        task = dict(original)
        if task["task_id"] == generation_id:
            continue
        if task["task_id"] == "bace_comrecgc_train_common_recourse":
            task["dependencies"] = [preflight_id]
            task["inputs"] = [
                str(generation) if value == str(generation) else value
                for value in task["inputs"]
            ]
        if task["task_id"] == "bace_comrecgc_train_candidates":
            task["argv"] = [*task["argv"], "--minimum-candidates", "10"]
        filtered.append(task)
    final_id = "bace_comrecgc_final_freeze"
    final_root = root / "final"
    standard_root = root / "standardized"
    project = _absolute(
        project_root, label="project_root", existing=True, directory=True
    )
    py = _absolute(python, label="python", existing=True, directory=False)
    checkpoint = _absolute(
        gnn_checkpoint,
        label="gnn_checkpoint",
        existing=True,
        directory=True,
    )
    filtered.append(
        {
            "task_id": "bace_comrecgc_standardized",
            "dataset": "bace",
            "method": "ComRecGC",
            "resource": {"kind": "cpu", "gpus": 0},
            "argv": [
                str(py),
                str(project / "scripts/autodl/standardize_bace_frozen_cell.py"),
                "--method",
                "ComRecGC",
                "--source-final-root",
                str(final_root),
                "--gnn-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(standard_root),
            ],
            "env": {
                "PYTHONPATH": str(project),
                "PYTHONHASHSEED": "0",
                "RUN_TASTEMOLNET": "0",
            },
            "controller_injected_env": [],
            "inputs": [str(final_root)],
            "output_root": str(standard_root),
            "fresh_output_required": True,
            "required_markers": ["PASS", "_FINALIZED.json"],
            "dependencies": [final_id],
            "max_transient_retries": 0,
            "max_oom_retries": 0,
            "retry_policy": "no_automatic_retry",
        }
    )
    receipt = _absolute(
        resource_cap_receipt,
        label="resource_cap_receipt",
        existing=True,
        directory=False,
    )
    return {
        **fragment,
        "schema_version": POSTPROCESS_SCHEMA,
        "tasks": filtered,
        "terminal_task_ids": ["bace_comrecgc_standardized"],
        "resource_cap_receipt": str(receipt),
        "resource_cap_receipt_sha256": sha256_file(receipt),
        "generation_adopted_from_checkpoint": True,
        "generation_task_omitted": generation_id,
        "test_decision_input": False,
    }


@dataclass(frozen=True, slots=True)
class ResourceCapExecutorInputs:
    handover_request: str
    checkpoint_dir: str
    source_trace_root: str
    source_resolved_config: str
    dataset_dir: str
    output_root: str
    python: str
    project_root: str
    gnn_checkpoint: str
    calibration_split: str
    test_split: str
    molclr_root: str
    molclr_checkpoint: str
    neurosed_checkpoint: str
    official_root: str
    process: ProcessContract
    poll_seconds: int = 60
    exit_wait_seconds: int = 300


class ResourceCapExecutor:
    """Persistent cap executor; before eligibility it only writes heartbeat."""

    def __init__(self, inputs: ResourceCapExecutorInputs) -> None:
        self.inputs = inputs
        if int(inputs.poll_seconds) <= 0 or int(inputs.exit_wait_seconds) <= 0:
            raise BaceComRecGCResourceCapExecutorError(
                "poll and exit-wait intervals must be positive"
            )
        self.root = _absolute(inputs.output_root, label="output_root")
        self.state_root = self.root / "executor"
        self.state_root.mkdir(parents=True, exist_ok=True)

    def _state(self, state: str, **fields: Any) -> dict[str, Any]:
        value = {
            "schema_version": EXECUTOR_SCHEMA,
            "state": state,
            "pid": os.getpid(),
            "heartbeat_at": utc_now(),
            "signals_sent": [],
            "sigkill_used": False,
            **fields,
        }
        atomic_write_json(self.state_root / "state.json", value)
        atomic_write_json(self.state_root / "heartbeat.json", value)
        return value

    def tick(self) -> dict[str, Any]:
        request_path = Path(self.inputs.handover_request).expanduser()
        if not request_path.is_absolute():
            raise BaceComRecGCResourceCapExecutorError(
                "handover_request must be absolute"
            )
        if not request_path.is_file():
            return self._state("WAITING_RESOURCE_CAP_REQUEST")
        request_path = request_path.resolve(strict=True)
        request = _object(request_path, label="handover request")
        checkpoint = load_generation_checkpoint(self.inputs.checkpoint_dir)
        decision = _validate_request(
            request,
            checkpoint_digest=checkpoint.validation.checkpoint_digest,
            completed_step=checkpoint.completed_step,
        )
        scientific_failure = (
            decision["status"]
            == "SCI_FAILED_ELIGIBLE_FOR_EXACT_GRACEFUL_STOP"
        )
        process_before = verify_exact_process(self.inputs.process)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": decision["status"],
            "receipt_state": (
                "RESOURCE_CAP_SCIENTIFIC_FAILURE_STOP_COMMITTED"
                if scientific_failure
                else "RESOURCE_CAP_HANDOVER_COMMITTED"
            ),
            "request_path": str(request_path),
            "request_sha256": sha256_file(request_path),
            "checkpoint_dir": str(checkpoint.validation.checkpoint_dir),
            "checkpoint_digest": checkpoint.validation.checkpoint_digest,
            "checkpoint_reload_pass": True,
            "M_configured_max": 20_000,
            "M_fallback_max": 25_000,
            "M_effective": checkpoint.completed_step,
            "m_effective": checkpoint.completed_step,
            "resource_cap_used": decision["reason"].startswith("RESOURCE_CAP_"),
            "early_stop_used": decision["reason"]
            == "PREREGISTERED_CONVERGENCE_PASS",
            "stop_reason": decision["reason"],
            "reason": decision["reason"],
            "valid_unique_count": decision["valid_unique_count"],
            "lineage_error_count": decision["lineage_error_count"],
            "calibration_loaded": False,
            "test_loaded": False,
            "process_before_signal": process_before,
            "requested_signal": "SIGTERM_EXACT_PID_ONLY",
            "sigkill_used": False,
            "created_at": utc_now(),
        }
        receipt_path = self.state_root / "resource_cap_receipt.json"
        if receipt_path.is_file():
            existing = _object(receipt_path, label="resource cap receipt")
            stable = (
                "checkpoint_digest",
                "M_effective",
                "stop_reason",
                "valid_unique_count",
                "lineage_error_count",
            )
            if any(existing.get(key) != receipt.get(key) for key in stable):
                raise BaceComRecGCResourceCapExecutorError(
                    "existing resource-cap receipt differs"
                )
            receipt = existing
        else:
            _atomic_json_new(receipt_path, receipt)
        # Reopen after receipt publication.  No PID lookup or fuzzy matching is
        # permitted between this identity proof and the exact signal call.
        process_immediate = verify_exact_process(self.inputs.process)
        os.kill(int(self.inputs.process.pid), signal.SIGTERM)
        signal_at = utc_now()
        deadline = time.monotonic() + int(self.inputs.exit_wait_seconds)
        exited = False
        while time.monotonic() < deadline:
            try:
                current = _parse_proc_stat(
                    Path("/proc") / str(self.inputs.process.pid) / "stat"
                )
            except (FileNotFoundError, OSError, BaceComRecGCResourceCapExecutorError):
                exited = True
                break
            if current["state"] in {"Z", "X", "x"}:
                exited = True
                break
            time.sleep(min(1.0, max(deadline - time.monotonic(), 0.0)))
        signal_receipt = {
            "signal": "SIGTERM",
            "signal_number": int(signal.SIGTERM),
            "signal_at": signal_at,
            "exact_pid": int(self.inputs.process.pid),
            "exact_start_ticks": int(self.inputs.process.start_ticks),
            "identity_immediately_before_signal": process_immediate,
            "exited_within_wait": exited,
            "sigkill_used": False,
        }
        atomic_write_json(self.state_root / "signal_receipt.json", signal_receipt)
        if not exited:
            return self._state(
                "SIGTERM_TIMEOUT",
                signals_sent=["SIGTERM"],
                signal_receipt=signal_receipt,
                manual_intervention_required=True,
                reason="exact worker did not exit; SIGKILL is forbidden",
            )
        if scientific_failure:
            return self._state(
                "SCIENTIFIC_FAILED_AT_ABSOLUTE_CAP",
                signals_sent=["SIGTERM"],
                signal_receipt=signal_receipt,
                decision=decision,
                checkpoint_digest=checkpoint.validation.checkpoint_digest,
                M_effective=checkpoint.completed_step,
                postprocess_started=False,
            )
        generation_root = self.root / "train_generation"
        manifest = materialize_resource_cap_checkpoint(
            checkpoint_dir=checkpoint.validation.checkpoint_dir,
            expected_checkpoint_digest=checkpoint.validation.checkpoint_digest,
            source_trace_root=self.inputs.source_trace_root,
            source_resolved_config=self.inputs.source_resolved_config,
            dataset_dir=self.inputs.dataset_dir,
            output_dir=generation_root,
            resource_cap_receipt=receipt,
            loaded_checkpoint=checkpoint,
        )
        formal_receipt_state: dict[str, Any] = {}
        materialized_at_20k = manifest.get("M_effective") == 20_000
        receipt_is_20k_cap = receipt.get("stop_reason") == "RESOURCE_CAP_20000"
        if materialized_at_20k != receipt_is_20k_cap:
            raise BaceComRecGCResourceCapExecutorError(
                "materialized generation and formal 20k receipt policy disagree"
            )
        if materialized_at_20k:
            cap_receipt_path, exclusion_path = _write_post20k_exclusion_receipts(
                state_root=self.state_root,
                source_generation_root=self.inputs.process.output_root,
                checkpoint=checkpoint,
                materialized_manifest=manifest,
                resource_cap_receipt=receipt,
            )
            formal_receipt_state = {
                "formal_resource_cap_receipt": str(cap_receipt_path),
                "formal_resource_cap_receipt_sha256": sha256_file(cap_receipt_path),
                "excluded_after_20k": str(exclusion_path),
                "excluded_after_20k_sha256": sha256_file(exclusion_path),
            }
        postprocess = build_postprocess_fragment(
            python=self.inputs.python,
            project_root=self.inputs.project_root,
            output_root=self.root,
            gnn_checkpoint=self.inputs.gnn_checkpoint,
            dataset_dir=self.inputs.dataset_dir,
            calibration_split=self.inputs.calibration_split,
            test_split=self.inputs.test_split,
            molclr_root=self.inputs.molclr_root,
            molclr_checkpoint=self.inputs.molclr_checkpoint,
            neurosed_checkpoint=self.inputs.neurosed_checkpoint,
            official_root=self.inputs.official_root,
            resource_cap_receipt=receipt_path,
        )
        fragment_path = self.state_root / "postprocess.tasks.json"
        _atomic_json_new(fragment_path, postprocess)
        return self._state(
            "POSTPROCESS_QUEUE_READY",
            signals_sent=["SIGTERM"],
            signal_receipt=signal_receipt,
            generation_root=str(generation_root),
            generation_manifest_sha256=sha256_file(
                generation_root / "run_manifest.json"
            ),
            M_effective=manifest["M_effective"],
            **formal_receipt_state,
            postprocess_fragment=str(fragment_path),
            postprocess_fragment_sha256=sha256_file(fragment_path),
        )

    def run(self, *, once: bool = False) -> int:
        while True:
            state = self.tick()
            if once or state["state"] not in {"WAITING_RESOURCE_CAP_REQUEST"}:
                return 0
            time.sleep(int(self.inputs.poll_seconds))


__all__ = [
    "BaceComRecGCResourceCapExecutorError",
    "ProcessContract",
    "ResourceCapExecutor",
    "ResourceCapExecutorInputs",
    "build_postprocess_fragment",
    "classify_liveness",
    "materialize_resource_cap_checkpoint",
    "verify_exact_process",
]
