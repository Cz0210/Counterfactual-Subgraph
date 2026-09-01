"""Read-only terminal wrapper for completed zero-flip AIDS ComRecGC science.

This module does not repair or copy scientific output.  It closes the narrow
historical case where the original controller died with a stale BLOCKED exact
projection, while a posthoc checkpoint-adoption receipt, an exact-recovery PASS
receipt, and a separately completed final root form a fully hash-bound science
closure.  A fresh wrapper records that orchestration fact and preserves the
scientifically valid zero-strict-flip result without numeric imputation.
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

from scripts.autodl.run_comrecgc_standardized_continuation import (
    _validate_common_recourse_completion,
)
from src.baselines.comrecgc.external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    _validate_component_recovery_closure,
)
from src.eval.am_legacy_standardization import scan_live_writers
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    CONTROLLER_ID,
    EXACT_CHECKPOINT_ADOPTION_SCHEMA,
    EXACT_STAGE,
    EXACT_STAGE_RECEIPT_SCHEMA,
    EXPECTED_ROWS,
    STAGE_ORDER,
    STATE_SCHEMA,
    _frozen_stage_environment,
    _process_group_member_pids,
    _read_handover_artifact,
    _validate_checkpoint_observation_shape,
    _validated_exact_checkpoint_snapshot_and_artifact,
    load_bound_controller_manifest,
    validate_stage_terminal,
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


def _stage_spec(manifest: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest.get("stages", ())
        if isinstance(row, Mapping) and row.get("stage_id") == stage_id
    ]
    if len(rows) != 1:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS controller stage identity is not unique: {stage_id}"
        )
    return rows[0]


def _dead_process(
    identity: Mapping[str, Any], *, proc_root: Path, label: str
) -> dict[str, int]:
    try:
        pid = int(identity["pid"])
        ticks = int(identity["start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} process identity is malformed"
        ) from exc
    if pid <= 0 or ticks <= 0:
        raise AIDSComRecGCTerminalReconciliationError(
            f"{label} process identity is malformed"
        )
    if _proc_start_ticks(proc_root, pid) == ticks:
        raise AIDSComRecGCTerminalReconciliationError(f"{label} is still alive")
    return {"pid": pid, "start_ticks": ticks}


def _validate_exact_receipt(
    *, manifest: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    stage = _stage_spec(manifest, EXACT_STAGE)
    expected = _physical_file(
        str(stage.get("terminal_path") or ""), label="AIDS exact stage terminal"
    )
    if receipt_path != expected:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact PASS receipt is not the manifest-bound stage terminal"
        )
    receipt = _json(receipt_path, label="AIDS exact PASS receipt")
    try:
        typed_exact = validate_stage_terminal(manifest, stage_id=EXACT_STAGE)
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS exact PASS receipt full scientific reopen failed: {exc}"
        ) from exc
    if (
        Path(str(typed_exact.get("path") or "")).resolve(strict=True)
        != receipt_path
        or typed_exact.get("sha256") != _sha(receipt_path)
        or typed_exact.get("stage_receipt") != receipt
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact PASS receipt identity changed on full scientific reopen"
        )
    dbscan_path = _physical_file(
        str(receipt.get("dbscan_manifest_path") or ""),
        label="AIDS exact DBSCAN manifest",
    )
    dbscan = _json(dbscan_path, label="AIDS exact DBSCAN manifest")
    if (
        receipt.get("schema_version") != EXACT_STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("run_complete") is not True
        or receipt.get("recovery_only") is not True
        or receipt.get("ordinary_pass_dependency_eligible") is not False
        or receipt.get("dbscan_partition_proven") is not True
        or receipt.get("observed_environment") != _frozen_stage_environment(manifest)
        or receipt.get("dbscan_manifest_path") != str(dbscan_path)
        or receipt.get("dbscan_manifest_sha256") != _sha(dbscan_path)
        or dbscan.get("run_complete") is not True
        or dbscan.get("clustering_path") != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
        or dbscan.get("approximation_used") is not False
        or int(dbscan.get("num_samples", -1)) != EXPECTED_ROWS
        or int(dbscan.get("core_count", -1)) != EXPECTED_ROWS
        or int(dbscan.get("noise_count", -1)) != 0
        or typed_exact.get("manifest") != dbscan
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact PASS receipt/DBSCAN contract changed"
        )
    try:
        _validate_component_recovery_closure(
            manifest=dbscan, root=dbscan_path.parent
        )
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS exact DBSCAN closure failed: {exc}"
        ) from exc
    proof_path = _physical_file(
        str(dbscan.get("shortcut_proof_path") or ""),
        label="AIDS exact DBSCAN proof",
    )
    proof = _json(proof_path, label="AIDS exact DBSCAN proof")
    if (
        proof.get("unique_seed_component_proven") is not True
        or int(proof.get("seed_component_count", -1)) != 1
        or proof.get("all_points_core_proven") is not True
        or proof.get("exact_multicomponent_partition_proven") is not True
        or proof.get("all_progress_prefixes_complete") is not True
        or typed_exact.get("proof") != proof
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact component proof theorem is incomplete"
        )
    linked: dict[str, dict[str, Any]] = {}
    for path_field, hash_field in (
        ("promotion_manifest_path", "promotion_manifest_sha256"),
        ("source_evidence_receipt_path", "source_evidence_receipt_sha256"),
        ("continuation_bootstrap_path", "continuation_bootstrap_sha256"),
    ):
        path = _physical_file(
            str(receipt.get(path_field) or ""), label=f"AIDS exact {path_field}"
        )
        digest = _sha(path)
        if digest != receipt.get(hash_field):
            raise AIDSComRecGCTerminalReconciliationError(
                f"AIDS exact receipt binding changed: {path_field}"
            )
        linked[path_field] = {"path": str(path), "sha256": digest}
    return {
        "status": "PASS",
        "path": str(receipt_path),
        "sha256": _sha(receipt_path),
        "dbscan_manifest_path": str(dbscan_path),
        "dbscan_manifest_sha256": _sha(dbscan_path),
        "proof_path": str(proof_path),
        "proof_sha256": _sha(proof_path),
        "linked_artifacts": linked,
    }


def validate_historical_controller_exact_authority(
    controller_manifest_path: str | Path,
    *,
    exact_receipt_path: str | Path,
    exact_adoption_gate_path: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Bind dead historical orchestration to its posthoc exact science."""

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
    startup_barrier = state.get("startup_barrier")
    exact_index = STAGE_ORDER.index(EXACT_STAGE)
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("controller_id") != CONTROLLER_ID
        or state.get("controller_manifest_sha256") != manifest["manifest_sha256"]
        or state.get("status") != "BLOCKED"
        or state.get("current_stage") != EXACT_STAGE
        or not isinstance(stage_states, Mapping)
        or set(stage_states) != set(STAGE_ORDER)
        or any(stage_states.get(stage) != "PASS" for stage in STAGE_ORDER[:exact_index])
        or stage_states.get(EXACT_STAGE) != "BLOCKED"
        or any(
            stage_states.get(stage) != "PENDING"
            for stage in STAGE_ORDER[exact_index + 1 :]
        )
        or not isinstance(startup_barrier, Mapping)
        or startup_barrier.get("stage_id", EXACT_STAGE) != EXACT_STAGE
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller is not the frozen historical BLOCKED-exact projection"
        )
    proc = _physical_directory(proc_root, label="proc root")
    controller_process = state.get("controller_process")
    if not isinstance(controller_process, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller process identity is absent"
        )
    dead_controller = _dead_process(
        controller_process, proc_root=proc, label="AIDS recovery controller"
    )
    worker = state.get("worker")
    if not isinstance(worker, Mapping) or worker.get("stage_id") != EXACT_STAGE:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS historical exact worker projection is absent"
        )
    dead_worker = _dead_process(worker, proc_root=proc, label="AIDS exact worker")
    try:
        process_group_id = int(worker.get("process_group_id", worker["pid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact process-group identity is malformed"
        ) from exc
    members = _process_group_member_pids(process_group_id, proc_root=proc)
    if members:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS exact process group is still alive: {members[:16]}"
        )
    lock_path = controller_root / ".controller.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller lock identity is absent"
        )
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

    adoption_path = _physical_file(
        exact_adoption_gate_path, label="AIDS posthoc exact checkpoint adoption"
    )
    try:
        adoption_raw, adoption_artifact = _read_handover_artifact(
            adoption_path, label="AIDS posthoc exact checkpoint adoption"
        )
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS posthoc exact checkpoint adoption reopen failed: {exc}"
        ) from exc
    projected = dict(adoption_raw)
    receipt_sha = projected.pop("receipt_sha256", None)
    snapshot = adoption_raw.get("checkpoint_snapshot")
    expected_adoption_fields = {
        "schema_version",
        "controller_manifest_sha256",
        "stage_id",
        "checkpoint_snapshot",
        "expected_progress_rows",
        "science_writer_absent",
        "publication_sequence",
        "signals_sent",
        "verified_at",
        "receipt_sha256",
    }
    if (
        set(adoption_raw) != expected_adoption_fields
        or adoption_raw.get("schema_version") != EXACT_CHECKPOINT_ADOPTION_SCHEMA
        or adoption_raw.get("controller_manifest_sha256")
        != manifest["manifest_sha256"]
        or adoption_raw.get("stage_id") != EXACT_STAGE
        or adoption_raw.get("science_writer_absent") is not True
        or adoption_raw.get("publication_sequence")
        != [
            "producer_os_replace",
            "producer_parent_fsync",
            "verifier_o_nofollow_open",
            "verifier_fstat",
            "verifier_fd_sha256",
        ]
        or adoption_raw.get("signals_sent") != []
        or not isinstance(adoption_raw.get("verified_at"), str)
        or receipt_sha != _stable_sha256(projected)
        or not isinstance(snapshot, Mapping)
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc exact checkpoint adoption contract changed"
        )
    try:
        adoption_snapshot = _validate_checkpoint_observation_shape(
            manifest=manifest, checkpoint=snapshot, require_positive=True
        )
        expected_progress = int(adoption_raw["expected_progress_rows"])
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS posthoc adoption-time checkpoint closure failed: {exc}"
        ) from exc
    if (
        expected_progress != int(adoption_snapshot["progress_rows"])
        or adoption_path.parent != controller_root / "gates"
        or adoption_path.name
        != "89_exact_checkpoint_adoption_"
        f"{str(adoption_snapshot['sha256_at_observation'])[:16]}.json"
        or adoption_artifact.get("path") != str(adoption_path)
        or adoption_artifact.get("content_sha256") != _sha(adoption_path)
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc exact checkpoint adoption-time identity changed"
        )
    exact_stage = _stage_spec(manifest, EXACT_STAGE)
    exact_science_root = _physical_directory(
        str(exact_stage.get("output_dir") or ""),
        label="AIDS exact recovery science root",
    )
    exact_writer = scan_live_writers(exact_science_root, proc_root=proc)
    if (
        exact_writer.get("procfs_verified") is not True
        or exact_writer.get("writable_fd_count") != 0
        or exact_writer.get("writers") != []
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact recovery science root still has a live writer"
        )
    checkpoint_path = _physical_file(
        str(adoption_snapshot.get("path") or ""),
        label="AIDS final exact checkpoint",
    )
    exact_path = _physical_file(
        exact_receipt_path, label="AIDS exact recovery PASS receipt"
    )
    exact = _validate_exact_receipt(manifest=manifest, receipt_path=exact_path)
    try:
        final_snapshot, final_checkpoint_artifact = (
            _validated_exact_checkpoint_snapshot_and_artifact(exact_stage)
        )
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS final exact checkpoint reopen failed: {exc}"
        ) from exc
    if (
        final_snapshot.get("path") != str(checkpoint_path)
        or final_checkpoint_artifact.get("path") != str(checkpoint_path)
        or final_checkpoint_artifact.get("content_sha256") != _sha(checkpoint_path)
        or final_snapshot.get("sha256_at_observation")
        != final_checkpoint_artifact.get("content_sha256")
        or final_snapshot.get("identity_sha256")
        != adoption_snapshot.get("identity_sha256")
        or final_snapshot.get("vectors_sha256")
        != adoption_snapshot.get("vectors_sha256")
        or int(final_snapshot.get("progress_rows", -1)) < expected_progress
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact checkpoint continuation regressed or changed scientific input"
        )
    return {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "controller_manifest_path": str(manifest_path),
        "controller_manifest_sha256": manifest["manifest_sha256"],
        "controller_root": str(controller_root),
        "controller_state_path": str(state_path),
        "controller_state_sha256": _sha(state_path),
        "exact_science_root": str(exact_science_root),
        "historical_state": "BLOCKED_EXACT_COMPONENT_RECOVERY",
        "stale_worker_projection_preserved": True,
        "stale_startup_barrier_preserved": True,
        "controller_terminal_present": False,
        "controller_pass_marker_present": False,
        "controller_process_alive": False,
        "exact_worker_alive": False,
        "exact_process_group_alive": False,
        "controller_lock_held": False,
        "controller_process": dead_controller,
        "exact_worker_process": dead_worker,
        "exact_process_group_id": process_group_id,
        "posthoc_exact_adoption_path": str(adoption_path),
        "posthoc_exact_adoption_sha256": _sha(adoption_path),
        "adoption_checkpoint_path": str(checkpoint_path),
        "adoption_checkpoint_sha256": adoption_snapshot[
            "sha256_at_observation"
        ],
        "adoption_checkpoint_progress_rows": expected_progress,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "checkpoint_progress_rows": int(final_snapshot["progress_rows"]),
        "checkpoint_identity_sha256": final_snapshot["identity_sha256"],
        "checkpoint_vectors_sha256": final_snapshot["vectors_sha256"],
        "checkpoint_monotonic_from_adoption": True,
        "science_writer_absent": True,
        "exact_receipt": exact,
        "old_state_modified": False,
        "controller_restart_performed": False,
    }


def validate_reconciled_final_science(
    source_root: str | Path,
    *,
    controller_evidence: Mapping[str, Any],
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
) -> dict[str, Any]:
    """Reopen the posthoc final root without consulting old stage gates."""

    # These mature validators already enforce the RF/split/MolCLR/freeze and
    # procfs writer contracts.  Importing locally avoids a module-load cycle:
    # non_taste_matrix_append dispatches reconciliation terminals back here.
    from src.eval.non_taste_matrix_append import (
        _critical_inventory,
        _validate_aids_standardized,
        _writer_audit,
    )

    root = _physical_directory(source_root, label="AIDS posthoc final science root")
    if any((root / name).exists() for name in ("FAILED", "FAILED.json", "FAIL.json")):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc final root contains a failure sentinel"
        )
    if _physical_file(root / "PASS", label="AIDS posthoc final PASS").read_bytes() != PASS_BYTES:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc final PASS bytes changed"
        )
    continuation_path = _physical_file(
        root / "_RUN_COMPLETE.json", label="AIDS posthoc continuation terminal"
    )
    continuation = _json(continuation_path, label="AIDS posthoc continuation terminal")
    common_path = _physical_file(
        root / "common_recourse/_RUN_COMPLETE.json",
        label="AIDS posthoc common-recourse terminal",
    )
    common = _json(common_path, label="AIDS posthoc common-recourse terminal")
    try:
        _validate_common_recourse_completion(marker=common_path, terminal=common)
    except Exception as exc:
        raise AIDSComRecGCTerminalReconciliationError(
            f"AIDS posthoc common-recourse closure failed: {exc}"
        ) from exc
    expected_continuation = {
        "schema_version": 1,
        "status": "PASS",
        "run_complete": True,
        "dataset": "aids",
        "method": "COMRECGC",
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "generation_adopted": True,
        "generation_rerun": False,
        "ordering_adopted": False,
        "evaluation_adopted": False,
        "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein",
        "standardized_output_root": str(root / "standardized"),
    }
    if any(continuation.get(key) != value for key, value in expected_continuation.items()):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc continuation terminal contract changed"
        )
    standardized = _validate_aids_standardized(root)
    run_path = _physical_file(root / "run_manifest.json", label="AIDS posthoc run")
    gate_path = _physical_file(root / "final_gate.json", label="AIDS posthoc final gate")
    run = _json(run_path, label="AIDS posthoc run")
    final_gate = _json(gate_path, label="AIDS posthoc final gate")
    expected_outer = dict(continuation)
    expected_outer.pop("run_complete", None)
    if run != expected_outer or final_gate != expected_outer:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc run/final gate diverged from PASS continuation"
        )
    if (
        run.get("standardized_run_manifest_sha256")
        != standardized["run_manifest_sha256"]
        or run.get("freeze_manifest_sha256")
        != standardized["freeze_manifest_sha256"]
        or run.get("teacher_sha256") != standardized["identities"]["oracle_hash"]
        or run.get("molclr_checkpoint_sha256")
        != standardized["identities"]["molclr_checkpoint_hash"]
        or run.get("dataset_csv_sha256")
        != standardized["identities"]["dataset_hash"]
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc final/standardized identities changed"
        )
    source_generation = _physical_directory(
        str(run.get("source_generation_root") or ""),
        label="AIDS frozen generation root",
    )
    source_integrity = _physical_file(
        root / "source_integrity_final.json",
        label="AIDS final source-integrity receipt",
    )
    if run.get("source_integrity_final_sha256") != _sha(source_integrity):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc source-integrity binding changed"
        )
    exact = controller_evidence.get("exact_receipt")
    if not isinstance(exact, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact receipt evidence is absent"
        )
    common_manifest_path = _physical_file(
        root / "common_recourse/run_manifest.json",
        label="AIDS posthoc common-recourse manifest",
    )
    common_manifest = _json(common_manifest_path, label="AIDS common-recourse manifest")
    external = common_manifest.get("external_memory_artifacts")
    if not isinstance(external, Mapping) or external.get("dbscan_adopted_read_only") is not True:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc final did not adopt exact DBSCAN read-only"
        )
    adoption_path = _physical_file(
        str(external.get("dbscan_adoption_manifest") or ""),
        label="AIDS posthoc DBSCAN adoption manifest",
    )
    adoption = _json(adoption_path, label="AIDS posthoc DBSCAN adoption manifest")
    if (
        external.get("dbscan_adoption_manifest_sha256") != _sha(adoption_path)
        or adoption.get("status") != "PASS"
        or adoption.get("run_complete") is not True
        or adoption.get("source_access") != "read_only"
        or adoption.get("source_mutated") is not False
        or adoption.get("dbscan_recomputed") is not False
        or adoption.get("pair_store_recomputed") is not False
        or adoption.get("sklearn_float64_semantics_preserved") is not True
        or adoption.get("exact_recovery_receipt_path") != exact.get("path")
        or adoption.get("exact_recovery_receipt_sha256") != exact.get("sha256")
        or adoption.get("source_manifest_path") != exact.get("dbscan_manifest_path")
        or adoption.get("source_manifest_sha256")
        != exact.get("dbscan_manifest_sha256")
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS posthoc final/exact receipt binding changed"
        )
    zero = validate_zero_strict_flip_science(root)
    writer = _writer_audit(
        root, proc_root=proc_root, required=require_writer_audit
    )
    return {
        "terminal_kind": "AIDS_POSTHOC_SELF_CLOSED_SCIENCE_FINAL",
        "root": str(root),
        "controller_manifest_path": controller_evidence[
            "controller_manifest_path"
        ],
        "controller_manifest_sha256": controller_evidence[
            "controller_manifest_sha256"
        ],
        "posthoc_exact_adoption_path": controller_evidence[
            "posthoc_exact_adoption_path"
        ],
        "posthoc_exact_adoption_sha256": controller_evidence[
            "posthoc_exact_adoption_sha256"
        ],
        "adoption_checkpoint_path": controller_evidence[
            "adoption_checkpoint_path"
        ],
        "adoption_checkpoint_sha256": controller_evidence[
            "adoption_checkpoint_sha256"
        ],
        "adoption_checkpoint_progress_rows": controller_evidence[
            "adoption_checkpoint_progress_rows"
        ],
        "checkpoint_path": controller_evidence["checkpoint_path"],
        "checkpoint_sha256": controller_evidence["checkpoint_sha256"],
        "checkpoint_progress_rows": controller_evidence[
            "checkpoint_progress_rows"
        ],
        "checkpoint_identity_sha256": controller_evidence[
            "checkpoint_identity_sha256"
        ],
        "checkpoint_vectors_sha256": controller_evidence[
            "checkpoint_vectors_sha256"
        ],
        "checkpoint_monotonic_from_adoption": controller_evidence[
            "checkpoint_monotonic_from_adoption"
        ],
        "exact_receipt_path": exact["path"],
        "exact_receipt_sha256": exact["sha256"],
        "exact_dbscan_manifest_path": exact["dbscan_manifest_path"],
        "exact_dbscan_manifest_sha256": exact["dbscan_manifest_sha256"],
        "continuation_terminal_sha256": _sha(continuation_path),
        "common_terminal_sha256": _sha(common_path),
        "run_manifest_sha256": _sha(run_path),
        "final_gate_sha256": _sha(gate_path),
        "standardized": standardized,
        "source_generation_root": str(source_generation),
        "source_integrity_final_sha256": _sha(source_integrity),
        "dbscan_adoption_manifest_path": str(adoption_path),
        "dbscan_adoption_manifest_sha256": _sha(adoption_path),
        "zero_strict_flip_evidence": zero,
        "writer_audit": writer,
        "inventory": _critical_inventory(
            root,
            (
                "PASS",
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "common_recourse/_RUN_COMPLETE.json",
                "common_recourse/run_manifest.json",
                "common_recourse/external_memory/dbscan_adoption/run_manifest.json",
                "standardized/run_manifest.json",
                "standardized/summary.json",
                "standardized/final_artifact_audit.json",
                "standardized/freeze_manifest.json",
                "standardized/_FINALIZED.json",
                "source_integrity_final.json",
            ),
        ),
    }


def science_terminal_projection(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Select the immutable posthoc science identities for the wrapper."""

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
        "posthoc_exact_adoption_path",
        "posthoc_exact_adoption_sha256",
        "adoption_checkpoint_path",
        "adoption_checkpoint_sha256",
        "adoption_checkpoint_progress_rows",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_progress_rows",
        "checkpoint_identity_sha256",
        "checkpoint_vectors_sha256",
        "checkpoint_monotonic_from_adoption",
        "exact_receipt_path",
        "exact_receipt_sha256",
        "exact_dbscan_manifest_path",
        "exact_dbscan_manifest_sha256",
        "continuation_terminal_sha256",
        "common_terminal_sha256",
        "run_manifest_sha256",
        "final_gate_sha256",
        "source_generation_root",
        "source_integrity_final_sha256",
        "dbscan_adoption_manifest_path",
        "dbscan_adoption_manifest_sha256",
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
    exact = controller_evidence.get("exact_receipt")
    if (
        controller_evidence.get("status") != "PASS"
        or controller_evidence.get("historical_state")
        != "BLOCKED_EXACT_COMPONENT_RECOVERY"
        or controller_evidence.get("stale_worker_projection_preserved") is not True
        or controller_evidence.get("stale_startup_barrier_preserved") is not True
        or controller_evidence.get("controller_terminal_present") is not False
        or controller_evidence.get("controller_pass_marker_present") is not False
        or controller_evidence.get("controller_process_alive") is not False
        or controller_evidence.get("exact_worker_alive") is not False
        or controller_evidence.get("exact_process_group_alive") is not False
        or controller_evidence.get("controller_lock_held") is not False
        or controller_evidence.get("science_writer_absent") is not True
        or controller_evidence.get("old_state_modified") is not False
        or controller_evidence.get("controller_restart_performed") is not False
        or not isinstance(exact, Mapping)
        or exact.get("status") != "PASS"
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
        or science_projection.get("posthoc_exact_adoption_path")
        != controller_evidence.get("posthoc_exact_adoption_path")
        or science_projection.get("posthoc_exact_adoption_sha256")
        != controller_evidence.get("posthoc_exact_adoption_sha256")
        or science_projection.get("adoption_checkpoint_path")
        != controller_evidence.get("adoption_checkpoint_path")
        or science_projection.get("adoption_checkpoint_sha256")
        != controller_evidence.get("adoption_checkpoint_sha256")
        or science_projection.get("adoption_checkpoint_progress_rows")
        != controller_evidence.get("adoption_checkpoint_progress_rows")
        or science_projection.get("checkpoint_path")
        != controller_evidence.get("checkpoint_path")
        or science_projection.get("checkpoint_sha256")
        != controller_evidence.get("checkpoint_sha256")
        or science_projection.get("checkpoint_progress_rows")
        != controller_evidence.get("checkpoint_progress_rows")
        or science_projection.get("checkpoint_identity_sha256")
        != controller_evidence.get("checkpoint_identity_sha256")
        or science_projection.get("checkpoint_vectors_sha256")
        != controller_evidence.get("checkpoint_vectors_sha256")
        or science_projection.get("checkpoint_monotonic_from_adoption") is not True
        or science_projection.get("exact_receipt_path") != exact.get("path")
        or science_projection.get("exact_receipt_sha256") != exact.get("sha256")
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
    exact_science_root = _physical_directory(
        controller_evidence.get("exact_science_root", ""),
        label="AIDS exact recovery science root",
    )
    if (
        output == exact_science_root
        or output in exact_science_root.parents
        or exact_science_root in output.parents
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "reconciliation output overlaps the read-only exact science root"
        )
    source_before = _inventory(source, _ZERO_FILES)
    if source_before != zero_evidence.get("source_inventory"):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS source changed before reconciliation publication"
        )
    projected_inventory = science_projection.get("inventory")
    if not isinstance(projected_inventory, Mapping) or not projected_inventory:
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS science closure inventory is absent"
        )
    source_closure_before = _inventory(source, tuple(projected_inventory))
    if source_closure_before != dict(projected_inventory):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS science closure changed before reconciliation publication"
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
        if _inventory(source, tuple(projected_inventory)) != source_closure_before:
            raise AIDSComRecGCTerminalReconciliationError(
                "AIDS science closure changed during reconciliation publication"
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
        or audit.get("source_science_root") != receipt.get("source_science_root")
        or audit.get("source_science_read_only") is not True
        or audit.get("zero_strict_flip_result_preserved") is not True
        or audit.get("numeric_imputation_used") is not False
        or audit.get("scientific_metrics_recomputed") is not False
        or audit.get("controller_restart_performed") is not False
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
    controller = receipt.get("controller_terminal_reconciliation")
    if not isinstance(controller, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller reconciliation evidence is absent"
        )
    exact = controller.get("exact_receipt")
    if not isinstance(exact, Mapping):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS exact receipt evidence is absent"
        )
    reopened_controller = validate_historical_controller_exact_authority(
        controller.get("controller_manifest_path", ""),
        exact_receipt_path=exact.get("path", ""),
        exact_adoption_gate_path=controller.get("posthoc_exact_adoption_path", ""),
        proc_root=proc_root,
    )
    if reopened_controller != dict(controller):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS controller reconciliation evidence changed"
        )
    reopened_science = validate_reconciled_final_science(
        source,
        controller_evidence=reopened_controller,
        proc_root=proc_root,
        require_writer_audit=True,
    )
    if science_terminal_projection(reopened_science) != receipt.get(
        "science_terminal_projection"
    ):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS reconciled science-terminal projection changed"
        )
    zero = reopened_science["zero_strict_flip_evidence"]
    if zero != receipt.get("zero_strict_flip_evidence"):
        raise AIDSComRecGCTerminalReconciliationError(
            "AIDS zero-strict-flip evidence changed"
        )
    return receipt


__all__ = [
    "AIDSComRecGCTerminalReconciliationError",
    "RECONCILIATION_SCHEMA",
    "publish_reconciliation",
    "science_terminal_projection",
    "validate_historical_controller_exact_authority",
    "validate_reconciled_final_science",
    "validate_reconciliation_root",
    "validate_zero_strict_flip_science",
]
