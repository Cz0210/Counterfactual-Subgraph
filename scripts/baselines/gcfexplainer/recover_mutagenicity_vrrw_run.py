#!/usr/bin/env python3
"""Recover a completed Mutagenicity VRRW artifact after persistence ENOSPC.

This command never reruns GCFExplainer.  It validates one explicitly selected
official ``counterfactuals.pt`` artifact and the failed run's frozen lineage,
then materializes a summary-compatible run directory without touching the
failed run.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import (  # noqa: E402
    stable_json_sha256,
)


RECOVERY_SCHEMA_VERSION = 1
EXPECTED_DATASET = "Mutagenicity"
EXPECTED_PROFILE = "full"
EXPECTED_PARENT_LIMIT = 1448
EXPECTED_M = 50000
EXPECTED_ALPHA = 1.0
EXPECTED_THETA = 0.05
EXPECTED_SEED = 13
VRRW_ALPHA_ENDPOINT_PATCH = "vrrw_alpha_endpoint_none_safe_v1"
ORIGINAL_FAILURE_REASON = "enospc_during_artifact_persistence"
REDUNDANT_COPY_RELATIVE_PATH = Path(
    "official_runtime/results/mutagenicity/runs/counterfactuals.pt"
)
VISITED_UNIVERSE_NAME = "visited_graph_universe.pt"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GCFExplainerVRRWRecoveryError(RuntimeError):
    """Raised when legacy VRRW artifacts cannot be safely recovered."""


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    failed_run_dir: Path
    counterfactuals_path: Path
    output_dir: Path
    expected_profile: str
    expected_parent_limit: int
    expected_m: int
    expected_alpha: float
    expected_theta: float
    expected_seed: int
    expected_job_id: str
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class PayloadAudit:
    payload_keys: tuple[str, ...]
    candidate_count: int
    graph_map_count: int
    graph_index_map_count: int
    traversed_step_count: int
    input_graph_covered_count: int
    candidate_order_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-run-dir", required=True)
    parser.add_argument("--counterfactuals-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-parent-limit", type=int, required=True)
    parser.add_argument("--expected-m", type=int, required=True)
    parser.add_argument("--expected-alpha", type=float, required=True)
    parser.add_argument("--expected-theta", type=float, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser


def _request_from_args(args: argparse.Namespace) -> RecoveryRequest:
    return RecoveryRequest(
        failed_run_dir=Path(args.failed_run_dir).expanduser().resolve(),
        counterfactuals_path=Path(args.counterfactuals_path).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        expected_profile=str(args.expected_profile),
        expected_parent_limit=int(args.expected_parent_limit),
        expected_m=int(args.expected_m),
        expected_alpha=float(args.expected_alpha),
        expected_theta=float(args.expected_theta),
        expected_seed=int(args.expected_seed),
        expected_job_id=str(args.expected_job_id).strip(),
        expected_bytes=int(args.expected_bytes),
        expected_sha256=str(args.expected_sha256).strip().lower(),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise GCFExplainerVRRWRecoveryError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GCFExplainerVRRWRecoveryError(
            f"Cannot read {label} as JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GCFExplainerVRRWRecoveryError(
            f"{label} must be a JSON object: {path}"
        )
    return payload


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
            raise
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _materialize_counterfactuals(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"Recovery target already exists: {destination}")
    try:
        os.link(source, destination)
        _fsync_directory(destination.parent)
        return "hardlink"
    except OSError as exc:
        fallback_errors = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        }
        if exc.errno not in fallback_errors:
            raise
    _atomic_copy(source, destination)
    return "atomic_copy"


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - HPC dependency gate
        raise GCFExplainerVRRWRecoveryError(
            "VRRW recovery requires PyTorch to validate counterfactuals.pt."
        ) from exc
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise GCFExplainerVRRWRecoveryError(
            f"torch.load failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _tensor_element_count(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    try:
        return len(value)
    except TypeError as exc:
        raise GCFExplainerVRRWRecoveryError(
            "payload input_graphs_covered has no auditable element count."
        ) from exc


def inspect_counterfactual_payload(
    payload: Any,
    *,
    expected_parent_count: int,
) -> PayloadAudit:
    if not isinstance(payload, dict) or not payload:
        raise GCFExplainerVRRWRecoveryError(
            "counterfactuals payload must be a non-empty dict."
        )
    required = {
        "graph_map",
        "graph_index_map",
        "counterfactual_candidates",
        "MAX_COUNTERFACTUAL_SIZE",
        "traversed_hashes",
        "input_graphs_covered",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise GCFExplainerVRRWRecoveryError(
            f"counterfactuals payload is missing official keys: {missing}"
        )
    graph_map = payload["graph_map"]
    graph_index_map = payload["graph_index_map"]
    candidates = payload["counterfactual_candidates"]
    traversed_hashes = payload["traversed_hashes"]
    if not isinstance(graph_map, Mapping) or not graph_map:
        raise GCFExplainerVRRWRecoveryError("payload graph_map must be non-empty.")
    if not isinstance(graph_index_map, Mapping):
        raise GCFExplainerVRRWRecoveryError("payload graph_index_map must be a mapping.")
    if not isinstance(candidates, list) or not candidates:
        raise GCFExplainerVRRWRecoveryError(
            "payload counterfactual_candidates must be a non-empty list."
        )
    if not isinstance(traversed_hashes, (list, tuple)) or not traversed_hashes:
        raise GCFExplainerVRRWRecoveryError(
            "payload traversed_hashes must be a non-empty sequence."
        )
    if len(graph_map) != len(graph_index_map) or len(candidates) != len(graph_map):
        raise GCFExplainerVRRWRecoveryError(
            "payload graph_map, graph_index_map, and candidate counts disagree."
        )
    candidate_hashes: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise GCFExplainerVRRWRecoveryError(
                f"counterfactual candidate {index} is not a mapping."
            )
        if "graph_hash" not in candidate:
            raise GCFExplainerVRRWRecoveryError(
                f"counterfactual candidate {index} has no graph_hash."
            )
        graph_hash = candidate["graph_hash"]
        if graph_hash not in graph_map:
            raise GCFExplainerVRRWRecoveryError(
                f"candidate {index} graph_hash is absent from graph_map."
            )
        if graph_hash not in graph_index_map or int(graph_index_map[graph_hash]) != index:
            raise GCFExplainerVRRWRecoveryError(
                f"candidate {index} disagrees with graph_index_map ordering."
            )
        candidate_hashes.append(str(graph_hash))
    input_graph_count = _tensor_element_count(payload["input_graphs_covered"])
    if input_graph_count != int(expected_parent_count):
        raise GCFExplainerVRRWRecoveryError(
            "payload input_graphs_covered count mismatch: "
            f"expected={expected_parent_count}, actual={input_graph_count}."
        )
    capacity = int(payload["MAX_COUNTERFACTUAL_SIZE"])
    if capacity < len(candidates):
        raise GCFExplainerVRRWRecoveryError(
            "payload candidate count exceeds MAX_COUNTERFACTUAL_SIZE."
        )
    return PayloadAudit(
        payload_keys=tuple(sorted(str(key) for key in payload)),
        candidate_count=len(candidates),
        graph_map_count=len(graph_map),
        graph_index_map_count=len(graph_index_map),
        traversed_step_count=len(traversed_hashes),
        input_graph_covered_count=input_graph_count,
        candidate_order_sha256=stable_json_sha256(candidate_hashes),
    )


def _validate_preregistered_request(request: RecoveryRequest) -> None:
    expected = {
        "expected_profile": (request.expected_profile, EXPECTED_PROFILE),
        "expected_parent_limit": (
            request.expected_parent_limit,
            EXPECTED_PARENT_LIMIT,
        ),
        "expected_m": (request.expected_m, EXPECTED_M),
        "expected_alpha": (request.expected_alpha, EXPECTED_ALPHA),
        "expected_theta": (request.expected_theta, EXPECTED_THETA),
        "expected_seed": (request.expected_seed, EXPECTED_SEED),
    }
    mismatches = [
        f"{field}: actual={actual!r}, required={required!r}"
        for field, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise GCFExplainerVRRWRecoveryError(
            "Recovery is restricted to the preregistered full VRRW run; "
            + "; ".join(mismatches)
        )
    if not request.expected_job_id or not request.expected_job_id.isdigit():
        raise GCFExplainerVRRWRecoveryError("expected-job-id must be numeric.")
    if request.expected_bytes <= 0:
        raise GCFExplainerVRRWRecoveryError("expected-bytes must be positive.")
    if not _SHA256_RE.fullmatch(request.expected_sha256):
        raise GCFExplainerVRRWRecoveryError(
            "expected-sha256 must be a lowercase 64-character SHA256."
        )


def _validate_paths(request: RecoveryRequest) -> None:
    if not request.failed_run_dir.is_dir():
        raise GCFExplainerVRRWRecoveryError(
            f"Failed run directory is missing: {request.failed_run_dir}"
        )
    source = request.counterfactuals_path
    if not source.is_file() or source.is_symlink():
        raise GCFExplainerVRRWRecoveryError(
            f"Counterfactual artifact must be a regular non-symlink file: {source}"
        )
    if request.output_dir == request.failed_run_dir:
        raise GCFExplainerVRRWRecoveryError(
            "Recovery output must differ from the failed run directory."
        )
    if request.output_dir == source or request.output_dir in source.parents:
        raise GCFExplainerVRRWRecoveryError(
            "Recovery output cannot replace or contain the source artifact path."
        )
    if request.output_dir.exists():
        if not request.output_dir.is_dir():
            raise GCFExplainerVRRWRecoveryError(
                f"Recovery output is not a directory: {request.output_dir}"
            )
        if any(request.output_dir.iterdir()):
            raise GCFExplainerVRRWRecoveryError(
                f"Recovery output directory must be empty: {request.output_dir}"
            )


def _validate_source_artifact(request: RecoveryRequest) -> str:
    actual_bytes = request.counterfactuals_path.stat().st_size
    if actual_bytes != request.expected_bytes:
        raise GCFExplainerVRRWRecoveryError(
            "counterfactuals size mismatch: "
            f"expected={request.expected_bytes}, actual={actual_bytes}."
        )
    actual_sha256 = _sha256_file(request.counterfactuals_path)
    if actual_sha256 != request.expected_sha256:
        raise GCFExplainerVRRWRecoveryError(
            "counterfactuals SHA256 mismatch: "
            f"expected={request.expected_sha256}, actual={actual_sha256}."
        )
    return actual_sha256


def _validate_resolved_config(
    request: RecoveryRequest,
) -> tuple[dict[str, Any], list[str], str]:
    config_path = request.failed_run_dir / "resolved_config.json"
    config = _read_json(config_path, label="source resolved_config.json")
    required_values = {
        "dataset": EXPECTED_DATASET,
        "profile": request.expected_profile,
        "parent_limit": request.expected_parent_limit,
        "generation_source_parent_rows": request.expected_parent_limit,
        "M": request.expected_m,
        "alpha": request.expected_alpha,
        "alpha_endpoint_branch": "individual_only",
        "theta": request.expected_theta,
        "seed": request.expected_seed,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    for field, expected in required_values.items():
        actual = config.get(field)
        if actual != expected:
            raise GCFExplainerVRRWRecoveryError(
                f"resolved_config mismatch for {field}: "
                f"expected={expected!r}, actual={actual!r}."
            )
    patches = config.get("official_compatibility_patches")
    if not isinstance(patches, list) or VRRW_ALPHA_ENDPOINT_PATCH not in patches:
        raise GCFExplainerVRRWRecoveryError(
            "resolved_config does not record the required alpha endpoint patch."
        )
    raw_ids = config.get("generation_parent_ids")
    if not isinstance(raw_ids, list):
        raise GCFExplainerVRRWRecoveryError(
            "resolved_config generation_parent_ids must be a list."
        )
    parent_ids = [str(value).strip() for value in raw_ids]
    if len(parent_ids) != request.expected_parent_limit:
        raise GCFExplainerVRRWRecoveryError(
            "generation parent ID count mismatch: "
            f"expected={request.expected_parent_limit}, actual={len(parent_ids)}."
        )
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise GCFExplainerVRRWRecoveryError(
            "generation parent IDs must be non-empty and unique."
        )
    parent_ids_sha256 = stable_json_sha256(parent_ids)
    lineage_hashes = {
        "generation_source_cohort_hash": config.get(
            "generation_source_cohort_hash"
        ),
    }
    if "generation_parent_ids_sha256" in config:
        lineage_hashes["generation_parent_ids_sha256"] = config.get(
            "generation_parent_ids_sha256"
        )
    for field, expected_hash in lineage_hashes.items():
        if str(expected_hash) != parent_ids_sha256:
            raise GCFExplainerVRRWRecoveryError(
                f"generation parent ID hash mismatch in {field}: "
                f"expected={expected_hash}, actual={parent_ids_sha256}."
            )
    fingerprint = str(config.get("config_fingerprint", ""))
    fingerprint_payload = {
        key: value for key, value in config.items() if key != "config_fingerprint"
    }
    actual_fingerprint = stable_json_sha256(fingerprint_payload)
    if not fingerprint or fingerprint != actual_fingerprint:
        raise GCFExplainerVRRWRecoveryError(
            "resolved_config fingerprint does not match its frozen lineage."
        )
    return config, parent_ids, parent_ids_sha256


def _resolve_checkpoint(config: Mapping[str, Any], field: str) -> Path:
    raw = str(config.get(field, "")).strip()
    if not raw:
        raise GCFExplainerVRRWRecoveryError(f"resolved_config is missing {field}.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise GCFExplainerVRRWRecoveryError(
            f"resolved_config {field} must be an absolute frozen path: {raw}"
        )
    path = path.resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise GCFExplainerVRRWRecoveryError(
            f"Frozen checkpoint is missing or invalid for {field}: {path}"
        )
    expected_hash = str(config.get(f"{field}_sha256", ""))
    actual_hash = _sha256_file(path)
    if not expected_hash or actual_hash != expected_hash:
        raise GCFExplainerVRRWRecoveryError(
            f"Checkpoint SHA256 mismatch for {field}: "
            f"expected={expected_hash}, actual={actual_hash}."
        )
    return path


def _read_internal_predictions(
    failed_run_dir: Path,
    parent_ids: Sequence[str],
) -> tuple[Path, list[dict[str, Any]], str, dict[str, int]]:
    path = failed_run_dir / "internal_gnn_predictions.jsonl"
    if not path.is_file() or path.is_symlink():
        raise GCFExplainerVRRWRecoveryError(
            f"Missing internal GNN predictions: {path}"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise GCFExplainerVRRWRecoveryError(
                    f"Blank internal prediction row at line {line_number}."
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GCFExplainerVRRWRecoveryError(
                    f"Invalid prediction JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise GCFExplainerVRRWRecoveryError(
                    f"Prediction row {line_number} is not a JSON object."
                )
            rows.append(row)
    if len(rows) != len(parent_ids):
        raise GCFExplainerVRRWRecoveryError(
            "internal GNN prediction row count mismatch: "
            f"expected={len(parent_ids)}, actual={len(rows)}."
        )
    prediction_ids = [str(row.get("molecule_id", "")) for row in rows]
    if prediction_ids != list(parent_ids):
        raise GCFExplainerVRRWRecoveryError(
            "internal GNN prediction parent order does not match generation_parent_ids."
        )
    counts = Counter(str(row.get("official_gnn_prediction")) for row in rows)
    return path, rows, _sha256_file(path), dict(sorted(counts.items()))


def _failure_text(payloads: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        json.dumps(dict(payload), sort_keys=True, default=str).lower()
        for payload in payloads
    )


def _validate_original_failure(
    request: RecoveryRequest,
) -> tuple[Path, dict[str, Any], str]:
    marker_path = request.failed_run_dir / "_RUN_FAILED.json"
    marker = _read_json(marker_path, label="source _RUN_FAILED.json")
    if marker.get("run_complete") is True:
        raise GCFExplainerVRRWRecoveryError(
            "Source failure marker incorrectly reports run_complete=true."
        )
    payloads: list[Mapping[str, Any]] = [marker]
    failure_summary_path = request.failed_run_dir / "failure_summary.json"
    if failure_summary_path.is_file():
        payloads.append(
            _read_json(failure_summary_path, label="source failure_summary.json")
        )
    recorded_job_ids = {
        str(payload.get(field))
        for payload in payloads
        for field in ("job_id", "slurm_job_id", "failed_job_id")
        if payload.get(field) not in (None, "")
    }
    if recorded_job_ids and recorded_job_ids != {request.expected_job_id}:
        raise GCFExplainerVRRWRecoveryError(
            "Source failure job ID disagrees with expected-job-id: "
            f"recorded={sorted(recorded_job_ids)}, expected={request.expected_job_id}."
        )
    text = _failure_text(payloads)
    direct_enospc = any(
        token in text
        for token in ("enospc", "errno 28", "no space left on device")
    )
    persistence_error = any(
        token in text
        for token in (
            "artifact persistence",
            "file write failed",
            "pytorchstreamwriter",
            "unexpected pos",
            "visited_graph_universe",
            "failed finding central directory",
        )
    )
    corrupt_visited = False
    visited_path = request.failed_run_dir / VISITED_UNIVERSE_NAME
    if visited_path.is_file() and visited_path.stat().st_size > 0:
        try:
            _torch_load(visited_path)
        except GCFExplainerVRRWRecoveryError:
            corrupt_visited = True
    if direct_enospc:
        evidence_mode = "failure_marker_enospc"
    elif persistence_error and corrupt_visited:
        evidence_mode = "persistence_error_plus_corrupt_visited_universe"
    else:
        raise GCFExplainerVRRWRecoveryError(
            "Source failure is not proven to be ENOSPC during artifact persistence; "
            "algorithm/runtime failures cannot be recovered as success."
        )
    return marker_path, marker, evidence_mode


def summary_requires_visited_graph_universe() -> bool:
    """Return the checked summary dependency contract.

    ``build_native_summary`` reads the recovered run manifest and the official
    counterfactual payload's ``graph_map``/``counterfactual_candidates``.  It
    never reads ``visited_graph_universe.pt``.
    """

    return False


def _validate_redundant_copy_status(failed_run_dir: Path) -> str:
    redundant = failed_run_dir / REDUNDANT_COPY_RELATIVE_PATH
    return "present_not_required" if redundant.exists() else "missing_not_required"


def recover_vrrw_run(request: RecoveryRequest) -> dict[str, Any]:
    _validate_preregistered_request(request)
    _validate_paths(request)
    source_sha256 = _validate_source_artifact(request)
    config, parent_ids, parent_ids_sha256 = _validate_resolved_config(request)
    gnn_checkpoint = _resolve_checkpoint(config, "gnn_checkpoint")
    neurosed_checkpoint = _resolve_checkpoint(config, "neurosed_checkpoint")
    predictions_path, _prediction_rows, predictions_sha256, prediction_counts = (
        _read_internal_predictions(request.failed_run_dir, parent_ids)
    )
    failure_marker_path, _failure_marker, failure_evidence_mode = (
        _validate_original_failure(request)
    )
    payload = _torch_load(request.counterfactuals_path)
    payload_audit = inspect_counterfactual_payload(
        payload,
        expected_parent_count=request.expected_parent_limit,
    )
    if summary_requires_visited_graph_universe():
        raise GCFExplainerVRRWRecoveryError(
            "Native summary requires visited_graph_universe.pt, but the corrupt "
            "legacy artifact cannot be deterministically reconstructed. Rerun VRRW."
        )
    visited_status = "optional_not_required"
    redundant_status = _validate_redundant_copy_status(request.failed_run_dir)

    request.output_dir.mkdir(parents=True, exist_ok=True)
    target_counterfactuals = request.output_dir / "counterfactuals.pt"
    materialization_mode = _materialize_counterfactuals(
        request.counterfactuals_path,
        target_counterfactuals,
    )
    materialized_bytes = target_counterfactuals.stat().st_size
    materialized_sha256 = _sha256_file(target_counterfactuals)
    if (
        materialized_bytes != request.expected_bytes
        or materialized_sha256 != source_sha256
    ):
        raise GCFExplainerVRRWRecoveryError(
            "Materialized counterfactual artifact failed post-write integrity check."
        )

    target_predictions = request.output_dir / "internal_gnn_predictions.jsonl"
    _atomic_copy(predictions_path, target_predictions)
    if _sha256_file(target_predictions) != predictions_sha256:
        raise GCFExplainerVRRWRecoveryError(
            "Materialized internal predictions failed integrity check."
        )
    _atomic_write_json(request.output_dir / "resolved_config.json", config)

    recovered_job_id: int | str = (
        int(request.expected_job_id)
        if request.expected_job_id.isdigit()
        else request.expected_job_id
    )
    recovery_manifest = {
        "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_validation_passed": True,
        "recovered_from_failed_job": recovered_job_id,
        "source_failed_run_dir": str(request.failed_run_dir),
        "source_failure_marker": str(failure_marker_path),
        "source_failure_marker_sha256": _sha256_file(failure_marker_path),
        "source_failure_evidence_mode": failure_evidence_mode,
        "source_counterfactuals_path": str(request.counterfactuals_path),
        "source_counterfactuals_sha256": source_sha256,
        "source_counterfactuals_bytes": request.expected_bytes,
        "redundant_official_copy_status": redundant_status,
        "artifact_materialization_mode": materialization_mode,
        "materialized_counterfactuals_path": str(target_counterfactuals),
        "materialized_counterfactuals_sha256": materialized_sha256,
        "profile": request.expected_profile,
        "parent_limit": request.expected_parent_limit,
        "M": request.expected_m,
        "alpha": request.expected_alpha,
        "theta": request.expected_theta,
        "seed": request.expected_seed,
        "candidate_count": payload_audit.candidate_count,
        "candidate_order_sha256": payload_audit.candidate_order_sha256,
        "candidate_order_unchanged": True,
        "payload_type": type(payload).__name__,
        "payload_keys": list(payload_audit.payload_keys),
        "graph_map_count": payload_audit.graph_map_count,
        "graph_index_map_count": payload_audit.graph_index_map_count,
        "traversed_step_count": payload_audit.traversed_step_count,
        "input_graph_covered_count": payload_audit.input_graph_covered_count,
        "algorithm_rerun": False,
        "original_run_failed": True,
        "original_failure_reason": ORIGINAL_FAILURE_REASON,
        "visited_graph_universe_status": visited_status,
        "gnn_checkpoint": str(gnn_checkpoint),
        "gnn_checkpoint_sha256": config["gnn_checkpoint_sha256"],
        "neurosed_checkpoint": str(neurosed_checkpoint),
        "neurosed_checkpoint_sha256": config["neurosed_checkpoint_sha256"],
        "generation_parent_ids_sha256": parent_ids_sha256,
        "internal_gnn_predictions_sha256": predictions_sha256,
        "calibration_loaded": False,
        "test_loaded": False,
        "remote_source_modified": False,
        "run_complete": True,
    }
    recovery_manifest_path = request.output_dir / "recovery_manifest.json"
    _atomic_write_json(recovery_manifest_path, recovery_manifest)

    run_manifest = {
        **config,
        "generation_parent_ids_sha256": parent_ids_sha256,
        "internal_gnn_prediction_counts": prediction_counts,
        "internal_gnn_predictions_path": str(target_predictions),
        "internal_gnn_predictions_sha256": predictions_sha256,
        "visited_graph_count": payload_audit.graph_map_count,
        "counterfactual_candidate_count": payload_audit.candidate_count,
        "traversed_step_count": payload_audit.traversed_step_count,
        "counterfactual_payload_type": type(payload).__name__,
        "counterfactual_payload_keys": list(payload_audit.payload_keys),
        "counterfactuals_path": str(target_counterfactuals),
        "counterfactuals_sha256": materialized_sha256,
        "counterfactuals_bytes": materialized_bytes,
        "official_algorithms_reused": [
            "vrrw.counterfactual_summary_with_randomwalk",
            "vrrw.move_to_next_graph",
            "vrrw.populate_counterfactual_candidates",
            "vrrw.dynamic_teleportation_probabilities",
            "importance.prepare_and_get",
            "distance.load_neurosed",
        ],
        "lineage_wrapper_changes_graph_tensors": False,
        "official_compatibility_patches": [VRRW_ALPHA_ENDPOINT_PATCH],
        "alpha_endpoint_branch": "individual_only",
        "recovered_run": True,
        "algorithm_rerun": False,
        "model_training_performed": False,
        "original_run_failed": True,
        "recovered_from_failed_job": recovered_job_id,
        "original_failure_reason": ORIGINAL_FAILURE_REASON,
        "candidate_order_unchanged": True,
        "candidate_order_sha256": payload_audit.candidate_order_sha256,
        "artifact_materialization_mode": materialization_mode,
        "redundant_official_copy_status": redundant_status,
        "visited_graph_universe_status": visited_status,
        "recovery_manifest_path": str(recovery_manifest_path),
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": True,
    }
    run_manifest_path = request.output_dir / "run_manifest.json"
    _atomic_write_json(run_manifest_path, run_manifest)
    complete = {
        "run_complete": True,
        "recovered_run": True,
        "algorithm_rerun": False,
        "counterfactuals_sha256": materialized_sha256,
        "run_manifest_sha256": _sha256_file(run_manifest_path),
        "recovery_manifest_sha256": _sha256_file(recovery_manifest_path),
        "calibration_loaded": False,
        "test_loaded": False,
    }
    _atomic_write_json(request.output_dir / "_RUN_COMPLETE.json", complete)
    return run_manifest


def _write_failure_artifacts(
    output_dir: Path,
    *,
    error: BaseException,
    request: RecoveryRequest,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_validation_passed": False,
        "run_complete": False,
        "stage": "vrrw_artifact_recovery",
        "error_type": type(error).__name__,
        "error": str(error),
        "source_failed_run_dir": str(request.failed_run_dir),
        "source_counterfactuals_path": str(request.counterfactuals_path),
        "output_dir": str(request.output_dir),
        "recovered_from_failed_job": request.expected_job_id,
        "algorithm_rerun": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    _atomic_write_json(output_dir / "failure_summary.json", payload)
    _atomic_write_json(output_dir / "_RUN_FAILED.json", payload)


def main(argv: list[str] | None = None) -> int:
    request = _request_from_args(build_parser().parse_args(argv))
    output_was_safe_for_this_run = (
        request.output_dir != request.failed_run_dir
        and request.output_dir != request.counterfactuals_path
        and (
            not request.output_dir.exists()
            or (
                request.output_dir.is_dir()
                and not any(request.output_dir.iterdir())
            )
        )
    )
    try:
        manifest = recover_vrrw_run(request)
    except Exception as exc:
        if output_was_safe_for_this_run:
            _write_failure_artifacts(request.output_dir, error=exc, request=request)
        print("[MUTAGENICITY_GCFEXPLAINER_VRRW_RECOVERY_ERROR]", file=sys.stderr)
        print(f"error_type={type(exc).__name__}", file=sys.stderr)
        print(f"error={exc}", file=sys.stderr)
        return 2
    print("[MUTAGENICITY_GCFEXPLAINER_VRRW_RECOVERY_OK]", flush=True)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
