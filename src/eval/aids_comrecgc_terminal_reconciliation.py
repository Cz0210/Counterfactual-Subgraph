"""Read-only terminal wrapper for completed zero-flip AIDS ComRecGC science.

This module does not repair or copy scientific output.  It closes the narrow
case where every typed recovery stage and the complete standardized science
root are immutable, but the recovery controller exited before publishing its
outer terminal.  A fresh wrapper records that orchestration fact and preserves
the scientifically valid zero-strict-flip result without numeric imputation.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Mapping

from src.eval.am_legacy_standardization import scan_live_writers
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    CONTROLLER_ID,
    STAGE_ORDER,
    STATE_SCHEMA,
    _gate_path,
    load_bound_controller_manifest,
    open_typed_recovery_gate,
)


RECONCILIATION_SCHEMA = "aids_comrecgc_terminal_publication_reconciliation_v1"
RECONCILIATION_AUDIT_SCHEMA = (
    "aids_comrecgc_terminal_publication_reconciliation_audit_v1"
)
PASS_BYTES = b"PASS\n"
_ZERO_FILES = (
    "standardized/run_manifest.json",
    "standardized/summary.json",
    "standardized/final_artifact_audit.json",
    "standardized/freeze_manifest.json",
    "standardized/_FINALIZED.json",
    "standardized/prefix_metrics.csv",
    "standardized/prefix_metrics.json",
    "standardized/figure3_coverage_vs_k.csv",
    "standardized/figure4_coverage_vs_threshold.csv",
    "standardized/table2_comrecgc_k10.csv",
    "standardized/table2_comrecgc_k20.csv",
)


class AIDSComRecGCTerminalReconciliationError(RuntimeError):
    """A completed AIDS science root cannot be reconciled read-only."""


def _stable_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_file(path_like: str | Path, *, label: str) -> Path:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} must be an absolute physical file"
        )
    try:
        path = logical.resolve(strict=True)
    except OSError as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} is absent: {logical}"
        ) from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} is not a nonempty file: {path}"
        )
    return path


def _physical_directory(path_like: str | Path, *, label: str) -> Path:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} must be an absolute physical directory"
        )
    try:
        root = logical.resolve(strict=True)
    except OSError as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} is absent: {logical}"
        ) from exc
    if not root.is_dir():
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} is not a directory: {root}"
        )
    return root


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"invalid {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} must contain one JSON object"
        )
    return dict(value)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _zero(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _unavailable(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "na", "n/a", "none", "null", "nan"}


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _inventory(root: Path, names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = _physical_file(root / name, label=f"AIDS source artifact {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    return result


def validate_zero_strict_flip_science(source_root: str | Path) -> dict[str, Any]:
    """Reopen the real zero-result exports without inventing candidates/cost."""

    root = _physical_directory(source_root, label="AIDS source science root")
    standardized = _physical_directory(
        root / "standardized", label="AIDS standardized source"
    )
    run = _json(standardized / "run_manifest.json", label="AIDS standardized run")
    summary = _json(standardized / "summary.json", label="AIDS standardized summary")
    audit = _json(
        standardized / "final_artifact_audit.json", label="AIDS standardized audit"
    )
    freeze = _json(
        standardized / "freeze_manifest.json", label="AIDS standardized freeze"
    )
    gate_path = _physical_file(
        str(freeze.get("source_gate_result_path") or ""),
        label="AIDS full science gate",
    )
    gate = _json(gate_path, label="AIDS full science gate")
    for label, value in (("run", run), ("summary", summary)):
        if (
            value.get("dataset") != "AIDS"
            or value.get("dataset_key") != "aids"
            or value.get("method")
            != "COMRECGC-Adapted-DeterministicChemRepair"
            or value.get("cf_mode") != "strict_flip"
            or value.get("scientific_output_empty") is not True
            or value.get("strict_flip_status") != "STRICT_FLIP_NOT_OBSERVED"
        ):
            raise AIDSComRecGCTerminalReconciliationError(
                f"AIDS {label} is not the frozen zero-strict-flip result"
            )
    # The production slot evaluator intentionally writes dataset identity only
    # to summary/run_manifest.  The final audit still carries the method and
    # complete strict-flip contract, and is hash-bound by the freeze manifest.
    if (
        audit.get("audit_passed") is not True
        or audit.get("run_complete") is not True
        or audit.get("method") != "COMRECGC-Adapted-DeterministicChemRepair"
        or audit.get("cf_mode") != "strict_flip"
        or audit.get("scientific_output_empty") is not True
        or audit.get("strict_flip_status") != "STRICT_FLIP_NOT_OBSERVED"
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS zero-strict-flip science audit did not pass"
        )
    if (
        gate.get("status") != "FULL_EXECUTION_PASS"
        or gate.get("audit_passed") is not True
        or gate.get("run_complete") is not True
        or gate.get("dataset") != "aids"
        or gate.get("scientific_output_empty") is not True
        or gate.get("scientific_output_status") != "SCIENTIFIC_OUTPUT_EMPTY"
        or gate.get("strict_flip_status") != "STRICT_FLIP_NOT_OBSERVED"
        or _sha(gate_path) != freeze.get("source_gate_result_sha256")
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS zero-strict-flip full gate changed"
        )
    prefix_value = _json(
        standardized / "prefix_metrics.json", label="AIDS prefix metrics"
    )
    prefixes = prefix_value.get("prefix_metrics")
    if (
        not isinstance(prefixes, list)
        or [row.get("k") for row in prefixes if isinstance(row, Mapping)]
        != list(range(1, 21))
    ):
        raise AIDSComRecGCTerminalReconciliationError("AIDS zero-result K grid changed")
    for row in prefixes:
        if (
            not isinstance(row, Mapping)
            or not _zero(row.get("close_cf_coverage"))
            or int(row.get("num_any_strict_flip_parents") or 0) != 0
            or not _unavailable(row.get("conditional_mean_cost"))
            or not _unavailable(row.get("conditional_median_cost"))
        ):
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS prefix contains a flip, nonzero coverage, or imputed conditional cost"
            )
        if (
            row.get("method") != "COMRECGC-Adapted-DeterministicChemRepair"
            or not _finite_nonnegative(row.get("fixed_capped_mean_cost"))
            or not _finite_nonnegative(row.get("fixed_capped_median_cost"))
        ):
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS prefix lacks its truthful fixed-capped zero-result cost"
            )
    prefix_csv = _read_csv(
        _physical_file(
            standardized / "prefix_metrics.csv", label="AIDS prefix metrics CSV"
        )
    )
    if [int(row.get("k") or 0) for row in prefix_csv] != list(range(1, 21)):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS prefix CSV K grid changed"
        )
    for row in prefix_csv:
        if (
            row.get("method") != "COMRECGC-Adapted-DeterministicChemRepair"
            or not _zero(row.get("close_cf_coverage"))
            or int(row.get("num_any_strict_flip_parents") or 0) != 0
            or not _unavailable(row.get("conditional_mean_cost"))
            or not _unavailable(row.get("conditional_median_cost"))
            or not _finite_nonnegative(row.get("fixed_capped_mean_cost"))
            or not _finite_nonnegative(row.get("fixed_capped_median_cost"))
        ):
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS prefix CSV is not the truthful zero-strict-flip result"
            )
    for relative, expected_rows, coverage_fields in (
        ("standardized/figure3_coverage_vs_k.csv", 20, ("close_cf_coverage", "coverage")),
        ("standardized/figure4_coverage_vs_threshold.csv", None, ("close_cf_coverage", "coverage")),
        ("standardized/table2_comrecgc_k10.csv", 1, ("coverage", "ccrcov")),
        ("standardized/table2_comrecgc_k20.csv", 1, ("coverage", "ccrcov")),
    ):
        rows = _read_csv(_physical_file(root / relative, label=relative))
        if not rows or (expected_rows is not None and len(rows) != expected_rows):
            raise AIDSComRecGCTerminalReconciliationError(
                f"AIDS zero-result export row count changed: {relative}"
            )
        if relative.endswith("figure3_coverage_vs_k.csv") and [
            int(row.get("k") or 0) for row in rows
        ] != list(range(1, 21)):
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS Figure 3 K grid changed"
            )
        if relative.endswith("figure4_coverage_vs_threshold.csv"):
            try:
                thresholds = [float(row.get("threshold")) for row in rows]
            except (TypeError, ValueError) as exc:
                raise AIDSComRecGCTerminalReconciliationError(
                    "AIDS Figure 4 threshold grid is invalid"
                ) from exc
            if (
                any(not math.isfinite(value) for value in thresholds)
                or thresholds != sorted(set(thresholds))
            ):
                raise AIDSComRecGCTerminalReconciliationError(
                    "AIDS Figure 4 threshold grid is invalid"
                )
        for row in rows:
            observed = [row[field] for field in coverage_fields if field in row]
            if (
                row.get("method")
                != "COMRECGC-Adapted-DeterministicChemRepair"
                or not observed
                or any(not _zero(value) for value in observed)
            ):
                raise AIDSComRecGCTerminalReconciliationError(
                    f"AIDS zero-result export contains nonzero coverage: {relative}"
                )
            for field in ("conditional_mean_cost", "conditional_median_cost", "cost"):
                if field in row and not _unavailable(row[field]):
                    raise AIDSComRecGCTerminalReconciliationError(
                        f"AIDS zero-result export imputes conditional cost: {relative}"
                    )
            if not relative.endswith(
                ("figure3_coverage_vs_k.csv", "figure4_coverage_vs_threshold.csv")
            ):
                expected_k = 10 if relative.endswith("k10.csv") else 20
                if (
                    int(row.get("k") or 0) != expected_k
                    or row.get("dataset") != "AIDS"
                    or not _finite_nonnegative(row.get("fixed_capped_mean_cost"))
                    or not _finite_nonnegative(row.get("fixed_capped_median_cost"))
                ):
                    raise AIDSComRecGCTerminalReconciliationError(
                        f"AIDS zero-result Table 2 contract changed: {relative}"
                    )
    return {
        "status": "PASS",
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "coverage": 0.0,
        "conditional_cost_available": False,
        "numeric_imputation_used": False,
        "full_gate_path": str(gate_path),
        "full_gate_sha256": _sha(gate_path),
        "source_inventory": _inventory(root, _ZERO_FILES),
    }


def _proc_start_ticks(proc_root: Path, pid: int) -> int | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        return int(raw[closing + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def validate_missing_controller_terminal(
    controller_manifest_path: str | Path, *, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    """Prove all typed stages passed and only the outer terminal is absent."""

    manifest_path = _physical_file(
        controller_manifest_path, label="AIDS recovery controller manifest"
    )
    manifest = load_bound_controller_manifest(manifest_path)
    controller_root = _physical_directory(
        manifest["controller_root"], label="AIDS recovery controller root"
    )
    if (controller_root / "terminal.json").exists() or (controller_root / "PASS").exists():
        raise AIDSComRecGCTerminalReconciliationError(
            "ordinary AIDS controller terminal exists; reconciliation is forbidden"
        )
    state_path = _physical_file(
        controller_root / "state.json", label="AIDS recovery controller state"
    )
    state = _json(state_path, label="AIDS recovery controller state")
    stage_states = state.get("stages")
    if not isinstance(stage_states, Mapping) or set(stage_states) != set(STAGE_ORDER):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller stage projection is malformed"
        )
    final_stage = STAGE_ORDER[-1]
    state_projection_is_terminal_gap = bool(
        state.get("status") in {"RUNNING", "BLOCKED"}
        and state.get("current_stage") is None
        and dict(stage_states) == {stage: "PASS" for stage in STAGE_ORDER}
        and (
            state.get("status") == "RUNNING"
            or isinstance(state.get("last_error"), Mapping)
        )
    )
    state_projection_is_publish_failure = bool(
        state.get("status") == "BLOCKED"
        and state.get("current_stage") == final_stage
        and all(stage_states.get(stage) == "PASS" for stage in STAGE_ORDER[:-1])
        and stage_states.get(final_stage) == "BLOCKED"
        and isinstance(state.get("last_error"), Mapping)
    )
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("controller_id") != CONTROLLER_ID
        or state.get("controller_manifest_sha256") != manifest["manifest_sha256"]
        or state.get("worker") is not None
        or state.get("startup_barrier") is not None
        or not (
            state_projection_is_terminal_gap
            or state_projection_is_publish_failure
        )
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller state is not a narrow post-science terminal-publication gap"
        )
    proc = _physical_directory(proc_root, label="proc root")
    controller_process = state.get("controller_process")
    if not isinstance(controller_process, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller process identity is absent"
        )
    try:
        pid = int(controller_process["pid"])
        ticks = int(controller_process["start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller process identity is malformed"
        ) from exc
    if pid <= 0 or ticks <= 0:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller process identity is malformed"
        )
    if _proc_start_ticks(proc, pid) == ticks:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS recovery controller is still alive"
        )
    pending = [
        str(path)
        for pattern in ("*.publish.tmp", "*.replace.tmp")
        for path in controller_root.rglob(pattern)
    ]
    if pending:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller has an interrupted immutable publication"
        )
    lock_path = controller_root / ".controller.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller lock identity is absent"
        )
    # Open read-only: reconciliation never acquires a write-capable descriptor
    # on the original controller root.
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS recovery controller lock is still held"
            ) from exc
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    writer = scan_live_writers(controller_root, proc_root=proc)
    if (
        writer.get("procfs_verified") is not True
        or writer.get("writable_fd_count") != 0
        or writer.get("writers") != []
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS recovery controller root still has a live writer"
        )
    gates: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        gate = open_typed_recovery_gate(manifest, stage)
        path = _physical_file(_gate_path(manifest, stage), label=f"AIDS {stage} gate")
        gates[stage] = {
            "path": str(path),
            "sha256": _sha(path),
            "gate_sha256": gate["gate_sha256"],
        }
    return {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "controller_manifest_path": str(manifest_path),
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "controller_root": str(controller_root),
        "controller_state_path": str(state_path),
        "controller_state_sha256": _sha(state_path),
        "controller_terminal_present": False,
        "controller_pass_marker_present": False,
        "controller_process_alive": False,
        "controller_lock_held": False,
        "all_typed_stages_pass": True,
        "mutable_state_projection": (
            "POST_SCIENCE_TERMINAL_GAP"
            if state_projection_is_terminal_gap
            else "FINAL_PUBLICATION_FAILURE_AFTER_TYPED_PASS"
        ),
        "typed_stage_gates": gates,
        "controller_restart_performed": False,
    }


def science_terminal_projection(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Select only immutable science identities from the matrix validator."""

    standardized = evidence.get("standardized")
    inventory = evidence.get("inventory")
    if not isinstance(standardized, Mapping) or not isinstance(inventory, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS science evidence lacks its immutable closure"
        )
    fields = (
        "root",
        "controller_manifest_path",
        "controller_manifest_sha256",
        "exact_stage_receipt_path",
        "exact_stage_receipt_sha256",
        "final_stage_receipt_path",
        "final_stage_receipt_sha256",
        "continuation_terminal_sha256",
        "common_terminal_sha256",
        "source_generation_root",
        "source_integrity_final_sha256",
    )
    projection = {field: evidence.get(field) for field in fields}
    projection["standardized"] = {
        field: standardized.get(field)
        for field in (
            "root",
            "source_evaluation_root",
            "run_manifest_sha256",
            "final_artifact_audit_sha256",
            "freeze_manifest_sha256",
            "identities",
        )
    }
    projection["inventory"] = dict(inventory)
    if any(value in (None, "") for value in projection.values()):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS science evidence projection is incomplete"
        )
    return projection


def publish_reconciliation(
    *,
    output_root: str | Path,
    science_projection: Mapping[str, Any],
    controller_evidence: Mapping[str, Any],
    zero_evidence: Mapping[str, Any],
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Atomically publish a fresh wrapper; the source root is never written."""

    output_logical = Path(output_root).expanduser()
    if not output_logical.is_absolute() or output_logical.is_symlink():
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation output must be an absolute fresh physical path"
        )
    output = output_logical.resolve(strict=False)
    if output.exists():
        raise AIDSComRecGCTerminalReconciliationError(
            f"reconciliation output already exists: {output}"
        )
    gates = controller_evidence.get("typed_stage_gates")
    if (
        controller_evidence.get("status") != "PASS"
        or controller_evidence.get("controller_terminal_present") is not False
        or controller_evidence.get("controller_pass_marker_present") is not False
        or controller_evidence.get("controller_process_alive") is not False
        or controller_evidence.get("controller_lock_held") is not False
        or controller_evidence.get("all_typed_stages_pass") is not True
        or controller_evidence.get("controller_restart_performed") is not False
        or not isinstance(gates, Mapping)
        or set(gates) != set(STAGE_ORDER)
        or zero_evidence.get("status") != "PASS"
        or zero_evidence.get("scientific_output_empty") is not True
        or zero_evidence.get("strict_flip_status")
        != "STRICT_FLIP_NOT_OBSERVED"
        or zero_evidence.get("coverage") != 0.0
        or zero_evidence.get("conditional_cost_available") is not False
        or zero_evidence.get("numeric_imputation_used") is not False
        or science_projection.get("controller_manifest_path")
        != controller_evidence.get("controller_manifest_path")
        or science_projection.get("controller_manifest_sha256")
        != controller_evidence.get("controller_manifest_sha256")
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation inputs are not the validated terminal-publication gap"
        )
    source = _physical_directory(
        science_projection.get("root", ""), label="AIDS source science root"
    )
    if output == source or output in source.parents or source in output.parents:
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation output overlaps the read-only source"
        )
    controller_root = _physical_directory(
        controller_evidence.get("controller_root", ""),
        label="AIDS recovery controller root",
    )
    if (
        output == controller_root
        or output in controller_root.parents
        or controller_root in output.parents
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation output overlaps the read-only controller root"
        )
    source_before = _inventory(source, _ZERO_FILES)
    if source_before != zero_evidence.get("source_inventory"):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS source changed before reconciliation publication"
        )
    payload: dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA,
        "status": "PASS",
        "run_complete": True,
        "dataset": "AIDS",
        "dataset_key": "aids",
        "method": "ComRecGC",
        "source_science_root": str(source),
        "source_science_read_only": True,
        "source_science_files_copied": False,
        "source_science_files_modified": False,
        "science_terminal_projection": dict(science_projection),
        "controller_terminal_reconciliation": dict(controller_evidence),
        "zero_strict_flip_evidence": dict(zero_evidence),
        "scientific_output_empty": True,
        "strict_flip_status": "STRICT_FLIP_NOT_OBSERVED",
        "coverage": 0.0,
        "conditional_cost_available": False,
        "numeric_imputation_used": False,
        "scientific_metrics_recomputed": False,
        "controller_restart_performed": False,
        "matrix_authority_bypassed": False,
    }
    payload["reconciliation_sha256"] = _stable_sha256(payload)
    audit = {
        "schema_version": RECONCILIATION_AUDIT_SCHEMA,
        "status": "PASS",
        "audit_passed": True,
        "source_science_root": str(source),
        "source_science_read_only": True,
        "zero_strict_flip_result_preserved": True,
        "numeric_imputation_used": False,
        "scientific_metrics_recomputed": False,
        "controller_restart_performed": False,
        "reconciliation_sha256": payload["reconciliation_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{secrets.token_hex(16)}"
    try:
        staging.mkdir()
        encoded = _json_bytes(payload)
        _write_new(staging / "terminal_reconciliation.json", encoded)
        _write_new(staging / "run_manifest.json", encoded)
        _write_new(staging / "final_artifact_audit.json", _json_bytes(audit))
        if _inventory(source, _ZERO_FILES) != source_before:
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS source changed during reconciliation publication"
            )
        _write_new(staging / "PASS", PASS_BYTES)
        atomic_rename_directory_noreplace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reopened = validate_reconciliation_root(output, proc_root=proc_root)
    if reopened != payload:
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation payload changed on reopen"
        )
    return reopened


def validate_reconciliation_root(
    root_like: str | Path, *, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    root = _physical_directory(root_like, label="AIDS reconciliation root")
    if _physical_file(root / "PASS", label="AIDS reconciliation PASS").read_bytes() != PASS_BYTES:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS reconciliation PASS bytes changed"
        )
    receipt_path = _physical_file(
        root / "terminal_reconciliation.json", label="AIDS reconciliation receipt"
    )
    receipt = _json(receipt_path, label="AIDS reconciliation receipt")
    run = _json(
        _physical_file(root / "run_manifest.json", label="AIDS reconciliation run"),
        label="AIDS reconciliation run",
    )
    audit = _json(
        _physical_file(
            root / "final_artifact_audit.json", label="AIDS reconciliation audit"
        ),
        label="AIDS reconciliation audit",
    )
    claimed = receipt.get("reconciliation_sha256")
    unsigned = {
        key: value for key, value in receipt.items() if key != "reconciliation_sha256"
    }
    if (
        receipt.get("schema_version") != RECONCILIATION_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("run_complete") is not True
        or receipt.get("dataset") != "AIDS"
        or receipt.get("method") != "ComRecGC"
        or claimed != _stable_sha256(unsigned)
        or run != receipt
        or audit.get("schema_version") != RECONCILIATION_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("audit_passed") is not True
        or audit.get("reconciliation_sha256") != claimed
        or receipt.get("source_science_read_only") is not True
        or receipt.get("source_science_files_copied") is not False
        or receipt.get("source_science_files_modified") is not False
        or receipt.get("scientific_output_empty") is not True
        or receipt.get("strict_flip_status") != "STRICT_FLIP_NOT_OBSERVED"
        or receipt.get("coverage") != 0.0
        or receipt.get("conditional_cost_available") is not False
        or receipt.get("numeric_imputation_used") is not False
        or receipt.get("scientific_metrics_recomputed") is not False
        or receipt.get("controller_restart_performed") is not False
        or receipt.get("matrix_authority_bypassed") is not False
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS reconciliation contract changed"
        )
    source = _physical_directory(
        receipt.get("source_science_root", ""), label="AIDS source science root"
    )
    zero = validate_zero_strict_flip_science(source)
    if zero != receipt.get("zero_strict_flip_evidence"):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS zero-strict-flip evidence changed"
        )
    controller = receipt.get("controller_terminal_reconciliation")
    if not isinstance(controller, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller reconciliation evidence is absent"
        )
    reopened_controller = validate_missing_controller_terminal(
        controller.get("controller_manifest_path", ""), proc_root=proc_root
    )
    if reopened_controller != dict(controller):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller reconciliation evidence changed"
        )
    return receipt


__all__ = [
    "AIDSComRecGCTerminalReconciliationError",
    "RECONCILIATION_SCHEMA",
    "publish_reconciliation",
    "science_terminal_projection",
    "validate_missing_controller_terminal",
    "validate_reconciliation_root",
    "validate_zero_strict_flip_science",
]
