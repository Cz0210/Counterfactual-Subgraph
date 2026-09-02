"""Read-only train-side convergence audit for TasteMolNet T14 ComRecGC.

The auditor consumes only atomically completed checkpoint directories.  It
never opens the active or checkpointed SQLite graph store, never writes below
the T14 run root, and exposes no process-signalling API.  Large
``generation_state.pt`` payloads are loaded one at a time only after the
minimum 5k/10k/12.5k evidence set exists.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from src.baselines.comrecgc.contracts import stable_json_sha256
from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256


AUDIT_SCHEMA = "tastemolnet_t14_external_convergence_audit_v1"
RECEIPT_SCHEMA = "tastemolnet_t14_external_convergence_receipt_v1"
CHECKPOINT_SCHEMA = "comrecgc_generation_checkpoint_v2"
STATE_SCHEMA = "comrecgc_generation_state_v2"
RUNTIME_SCHEMA = "tastemolnet_t14_bounded_runtime_v1"
BOUNDARY = "after_fully_completed_step_v1"
ALLOWED_STEPS = (5_000, 10_000, 12_500, 15_000, 17_500, 20_000)
_CHECKPOINT_NAME = re.compile(r"step-(?P<step>[0-9]{12})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class T14ExternalConvergenceError(RuntimeError):
    """Committed T14 convergence evidence is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class T14ConvergencePolicy:
    top100_jaccard_min: float = 0.99
    top20_jaccard_min: float = 0.95
    rank_spearman_min: float = 0.99
    absolute_train_coverage_gain_max: float = 0.005
    minimum_valid_unique_count: int = 10
    required_consecutive_windows: int = 2


POLICY = T14ConvergencePolicy()


@dataclass(frozen=True, slots=True)
class CommittedCheckpoint:
    step: int
    root: Path
    state_path: Path
    state_bytes: int
    state_sha256: str
    sqlite_path: Path
    sqlite_bytes: int
    sqlite_sha256: str
    checkpoint_digest: str
    manifest_sha256: str
    provenance: Mapping[str, str]
    provenance_sha256: str


def _physical_regular(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise T14ExternalConvergenceError(f"Missing checkpoint evidence: {path}") from exc
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise T14ExternalConvergenceError(
            f"Checkpoint evidence must be a physical regular file: {path}"
        )
    return value


def _stable_bytes(path: Path, maximum: int = 16 * 1024 * 1024) -> bytes:
    before = _physical_regular(path)
    if before.st_size <= 0 or before.st_size > maximum:
        raise T14ExternalConvergenceError(f"Small evidence has invalid size: {path}")
    payload = path.read_bytes()
    after = _physical_regular(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise T14ExternalConvergenceError(f"Evidence changed while read: {path}")
    return payload


def _json(path: Path) -> tuple[dict[str, Any], str]:
    payload = _stable_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14ExternalConvergenceError(f"Invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise T14ExternalConvergenceError(f"Expected JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise T14ExternalConvergenceError(f"{field} is not a lowercase SHA-256")
    return value


def _normalize_provenance(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise T14ExternalConvergenceError("Checkpoint provenance is absent")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise T14ExternalConvergenceError("Checkpoint provenance is not string typed")
        key, item = raw_key.strip(), raw_value.strip()
        if not key or not item or key in result:
            raise T14ExternalConvergenceError("Checkpoint provenance is ambiguous")
        result[key] = item
    return dict(sorted(result.items()))


def _declared_file(checkpoint: Path, name: str, value: Any) -> tuple[Path, int, str]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        raise T14ExternalConvergenceError(f"Malformed checkpoint file declaration: {name}")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise T14ExternalConvergenceError(f"Invalid checkpoint file size: {name}")
    digest = _sha(value.get("sha256"), f"files.{name}.sha256")
    path = checkpoint / name
    observed = _physical_regular(path)
    if observed.st_size != size:
        raise T14ExternalConvergenceError(f"Checkpoint file size changed: {path}")
    return path, size, digest


def validate_committed_checkpoint(checkpoint: Path) -> CommittedCheckpoint:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise T14ExternalConvergenceError(
            f"Checkpoint must be one physical directory: {checkpoint}"
        )
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise T14ExternalConvergenceError(f"Invalid checkpoint directory: {checkpoint}")
    step = int(match.group("step"))
    if step not in ALLOWED_STEPS:
        raise T14ExternalConvergenceError(f"Checkpoint is off audit cadence: {step}")
    manifest, manifest_sha = _json(checkpoint / "checkpoint_manifest.json")
    complete, _ = _json(checkpoint / "_CHECKPOINT_COMPLETE.json")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or manifest.get("state_schema_version") != STATE_SCHEMA
        or manifest.get("boundary") != BOUNDARY
        or manifest.get("atomic_complete") is not True
        or manifest.get("file_digest_algorithm") != "sha256"
        or manifest.get("checkpoint_digest_scheme") != "stable_json_sha256_v1"
        or manifest.get("checkpoint_dir") != checkpoint.name
        or manifest.get("completed_step") != step
        or manifest.get("next_step") != step + 1
        or manifest.get("total_steps") != 25_000
    ):
        raise T14ExternalConvergenceError(
            f"Checkpoint is not an atomic T14 completed-step boundary: {checkpoint}"
        )
    provenance = _normalize_provenance(manifest.get("provenance_fingerprints"))
    provenance_sha = stable_json_sha256(provenance)
    required = {
        "dataset": "tastemolnet",
        "method": "comrecgc",
        "stage": "T14_COMRECGC_FULL",
        "runtime_state_schema": RUNTIME_SCHEMA,
        "total_steps": "25000",
    }
    mismatches = [key for key, expected in required.items() if provenance.get(key) != expected]
    if (
        mismatches
        or provenance_sha != manifest.get("provenance_sha256")
        or not _SHA256.fullmatch(str(provenance.get("train_csv_sha256") or ""))
    ):
        raise T14ExternalConvergenceError(
            "Checkpoint provenance is not the frozen train-only T14 route: "
            + ",".join(mismatches)
        )
    scientific_argv = manifest.get("scientific_argv")
    if not isinstance(scientific_argv, list) or not scientific_argv:
        raise T14ExternalConvergenceError("Checkpoint scientific argv is absent")
    forbidden = {"--calibration", "--calibration-csv", "--test", "--test-csv"}
    if any(str(item).split("=", 1)[0] in forbidden for item in scientific_argv):
        raise T14ExternalConvergenceError("Calibration/test appeared in T14 generation argv")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "generation_state.pt",
        "authoritative_graph_store.sqlite3",
    }:
        raise T14ExternalConvergenceError("Checkpoint file inventory is incomplete")
    state_path, state_bytes, state_sha = _declared_file(
        checkpoint, "generation_state.pt", files["generation_state.pt"]
    )
    # Metadata only.  This path is intentionally never opened by this module.
    sqlite_path, sqlite_bytes, sqlite_sha = _declared_file(
        checkpoint,
        "authoritative_graph_store.sqlite3",
        files["authoritative_graph_store.sqlite3"],
    )
    checkpoint_digest = _sha(manifest.get("checkpoint_digest"), "checkpoint_digest")
    unsigned = {key: value for key, value in manifest.items() if key != "checkpoint_digest"}
    if checkpoint_digest != stable_json_sha256(unsigned):
        raise T14ExternalConvergenceError("Checkpoint manifest digest changed")
    if complete != {
        "checkpoint_digest": checkpoint_digest,
        "manifest_sha256": manifest_sha,
        "schema_version": CHECKPOINT_SCHEMA,
    }:
        raise T14ExternalConvergenceError("Checkpoint completion marker changed")
    return CommittedCheckpoint(
        step=step,
        root=checkpoint.resolve(),
        state_path=state_path.resolve(),
        state_bytes=state_bytes,
        state_sha256=state_sha,
        sqlite_path=sqlite_path.resolve(),
        sqlite_bytes=sqlite_bytes,
        sqlite_sha256=sqlite_sha,
        checkpoint_digest=checkpoint_digest,
        manifest_sha256=manifest_sha,
        provenance=provenance,
        provenance_sha256=provenance_sha,
    )


def discover_committed_checkpoints(root: Path) -> list[CommittedCheckpoint]:
    if root.is_symlink() or not root.is_dir():
        raise T14ExternalConvergenceError(f"Checkpoint root is invalid: {root}")
    rows: list[CommittedCheckpoint] = []
    for step in ALLOWED_STEPS:
        path = root / f"step-{step:012d}"
        if not path.exists():
            continue
        rows.append(validate_committed_checkpoint(path))
    return rows


def _torch_loader(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - AutoDL dependency
        raise T14ExternalConvergenceError("PyTorch is required for T14 summaries") from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - pinned older torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise T14ExternalConvergenceError("T14 checkpoint state is not a mapping")
    return value


def _native_list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise T14ExternalConvergenceError(f"{field} is not tensor/list-like")
    return value


def _frequency(value: Any) -> int:
    if isinstance(value, bool):
        raise T14ExternalConvergenceError("Candidate frequency is boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise T14ExternalConvergenceError("Candidate frequency is invalid") from exc
    if result <= 0:
        raise T14ExternalConvergenceError("Candidate frequency must be positive")
    return result


def _ranking(frequencies: Mapping[str, int], allowed: set[str] | None = None) -> list[str]:
    items = (
        (key, value)
        for key, value in frequencies.items()
        if allowed is None or key in allowed
    )
    return [key for key, _ in sorted(items, key=lambda item: (-item[1], item[0]))]


def summarize_checkpoint(
    checkpoint: CommittedCheckpoint,
    *,
    state_loader: Callable[[Path], Mapping[str, Any]] = _torch_loader,
) -> dict[str, Any]:
    state = state_loader(checkpoint.state_path)
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("boundary") != BOUNDARY
        or state.get("fully_completed_step") is not True
        or state.get("completed_step") != checkpoint.step
        or state.get("next_step") != checkpoint.step + 1
        or state.get("provenance_sha256") != checkpoint.provenance_sha256
    ):
        raise T14ExternalConvergenceError("T14 state/manifest boundary mismatch")
    algorithm = state.get("algorithm_state")
    if not isinstance(algorithm, Mapping) or algorithm.get("schema_version") != RUNTIME_SCHEMA:
        raise T14ExternalConvergenceError("T14 runtime state schema changed")
    official = algorithm.get("official_state")
    bridge = algorithm.get("bridge_state")
    if not isinstance(official, Mapping) or not isinstance(bridge, Mapping):
        raise T14ExternalConvergenceError("T14 checkpoint lacks official/bridge state")
    if bridge.get("schema_version") != "tastemolnet_comrecgc_bridge_checkpoint_v3":
        raise T14ExternalConvergenceError("T14 bridge state schema changed")

    raw_candidates = official.get("counterfactual_candidates")
    records = bridge.get("records")
    collisions = bridge.get("graph_collision_payloads")
    lineages = bridge.get("lineage_occurrences")
    if not isinstance(raw_candidates, list) or not all(
        isinstance(value, Mapping) for value in (records, collisions, lineages)
    ):
        raise T14ExternalConvergenceError("T14 candidate/lineage state is incomplete")
    frequencies: Counter[str] = Counter()
    for row in raw_candidates:
        if not isinstance(row, Mapping):
            raise T14ExternalConvergenceError("Official candidate row is malformed")
        graph_hash = _sha(row.get("graph_hash"), "candidate.graph_hash")
        frequencies[graph_hash] += _frequency(row.get("frequency", 1))

    valid: set[str] = set()
    lineage_errors: list[dict[str, Any]] = []
    for raw_key, raw_record in records.items():
        key = str(raw_key)
        failures: list[str] = []
        if _SHA256.fullmatch(key) is None or not isinstance(raw_record, Mapping):
            failures.append("record_identity")
        else:
            collision = collisions.get(key)
            lineage = lineages.get(key)
            try:
                collision_matches = (
                    isinstance(collision, dict)
                    and _identity_graph_sha256(collision) == key
                )
            except Exception:
                collision_matches = False
            if raw_record.get("graph_identity_sha256") != key:
                failures.append("record_graph_identity")
            if not collision_matches:
                failures.append("collision_payload")
            if (
                not isinstance(lineage, Mapping)
                or not lineage
                or any(
                    _SHA256.fullmatch(str(parent)) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                    for parent, count in lineage.items()
                )
            ):
                failures.append("lineage")
            candidate = (
                raw_record.get("valid_fullgraph") is True
                and raw_record.get("candidate") is True
                and raw_record.get("prediction") in (0, 2)
                and isinstance(raw_record.get("canonical_graph"), str)
                and bool(raw_record.get("canonical_graph"))
            )
            if candidate and not failures:
                valid.add(key)
        if failures:
            lineage_errors.append({"graph_identity_sha256": key, "failures": failures})

    coverage_values = _native_list(official.get("input_graphs_covered"), "coverage")
    if not coverage_values:
        raise T14ExternalConvergenceError("T14 train coverage vector is empty")
    normalized_coverage: list[float] = []
    for value in coverage_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise T14ExternalConvergenceError("T14 coverage value is non-numeric")
        item = float(value)
        if not math.isfinite(item) or item < 0:
            raise T14ExternalConvergenceError("T14 coverage value is invalid")
        normalized_coverage.append(item)
    top100 = _ranking(frequencies)[:100]
    top20 = _ranking(frequencies, valid)[:20]
    result = {
        "schema_version": "tastemolnet_t14_checkpoint_summary_v1",
        "step": checkpoint.step,
        "checkpoint_root": str(checkpoint.root),
        "checkpoint_digest": checkpoint.checkpoint_digest,
        "state_path": str(checkpoint.state_path),
        "state_declared_sha256": checkpoint.state_sha256,
        "state_declared_bytes": checkpoint.state_bytes,
        "sqlite_path": str(checkpoint.sqlite_path),
        "sqlite_accessed": False,
        "split": "train",
        "calibration_loaded": False,
        "test_loaded": False,
        "candidate_frequency": dict(sorted(frequencies.items())),
        "top100_candidate_hashes": top100,
        "top20_provisional_rule_hashes": top20,
        "valid_unique_rule_count": len(valid),
        "valid_unique_rule_hashes": sorted(valid),
        "lineage_error_count": len(lineage_errors),
        "lineage_errors": lineage_errors,
        "train_coverage": sum(item > 0 for item in normalized_coverage)
        / len(normalized_coverage),
        "train_parent_count": len(normalized_coverage),
    }
    # Release the 10s-of-GiB object before loading another checkpoint.
    del state, algorithm, official, bridge, raw_candidates, records, collisions, lineages
    gc.collect()
    return result


def _jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    left, right = set(first), set(second)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _rank_map(frequencies: Mapping[str, int], universe: set[str]) -> dict[str, int]:
    ordered = _ranking({key: int(frequencies.get(key, 0)) for key in universe})
    return {key: index + 1 for index, key in enumerate(ordered)}


def _spearman(first: Mapping[str, int], second: Mapping[str, int]) -> float:
    universe = set(first) | set(second)
    if len(universe) <= 1:
        return 1.0
    first_rank, second_rank = _rank_map(first, universe), _rank_map(second, universe)
    xs = [float(first_rank[key]) for key in sorted(universe)]
    ys = [float(second_rank[key]) for key in sorted(universe)]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    left = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    right = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return 1.0 if left == 0 and right == 0 else numerator / (left * right)


def compare_summaries(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if int(after["step"]) <= int(before["step"]):
        raise T14ExternalConvergenceError("Convergence window steps are not increasing")
    top100 = _jaccard(before["top100_candidate_hashes"], after["top100_candidate_hashes"])
    top20 = _jaccard(
        before["top20_provisional_rule_hashes"],
        after["top20_provisional_rule_hashes"],
    )
    spearman = _spearman(before["candidate_frequency"], after["candidate_frequency"])
    coverage_gain = float(after["train_coverage"]) - float(before["train_coverage"])
    checks = {
        "top100_jaccard": top100 >= POLICY.top100_jaccard_min,
        "top20_jaccard": top20 >= POLICY.top20_jaccard_min,
        "rank_spearman": spearman >= POLICY.rank_spearman_min,
        "absolute_train_coverage_gain": abs(coverage_gain)
        <= POLICY.absolute_train_coverage_gain_max,
        "valid_unique_rule_count": int(after["valid_unique_rule_count"])
        >= POLICY.minimum_valid_unique_count,
        "lineage_error_count": int(after["lineage_error_count"]) == 0,
    }
    return {
        "before_step": int(before["step"]),
        "after_step": int(after["step"]),
        "top100_jaccard": top100,
        "top20_jaccard": top20,
        "rank_spearman": spearman,
        "absolute_train_coverage_gain": abs(coverage_gain),
        "signed_train_coverage_gain": coverage_gain,
        "valid_unique_rule_count": int(after["valid_unique_rule_count"]),
        "lineage_error_count": int(after["lineage_error_count"]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_t14_external_convergence(
    checkpoint_root: Path,
    *,
    state_loader: Callable[[Path], Mapping[str, Any]] = _torch_loader,
) -> dict[str, Any]:
    checkpoints = discover_committed_checkpoints(checkpoint_root)
    steps = [row.step for row in checkpoints]
    minimum = {5_000, 10_000, 12_500}
    evidence = [
        {
            **asdict(row),
            "root": str(row.root),
            "state_path": str(row.state_path),
            "sqlite_path": str(row.sqlite_path),
            "provenance": dict(row.provenance),
        }
        for row in checkpoints
    ]
    if not minimum.issubset(steps):
        return {
            "schema_version": AUDIT_SCHEMA,
            "status": "WAITING_FOR_12500",
            "policy": asdict(POLICY),
            "available_steps": steps,
            "required_initial_steps": sorted(minimum),
            "checkpoint_evidence": evidence,
            "checkpoint_state_loaded": False,
            "sqlite_accessed": False,
            "converged": False,
            "safe_stop_authorized": False,
        }
    summaries: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        if checkpoint.step < 5_000:
            continue
        summaries.append(summarize_checkpoint(checkpoint, state_loader=state_loader))
    windows = [
        compare_summaries(first, second)
        for first, second in zip(summaries, summaries[1:])
    ]
    consecutive = 0
    for window in windows:
        consecutive = consecutive + 1 if window["pass"] else 0
    converged = consecutive >= POLICY.required_consecutive_windows
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "CONVERGED_EARLY_STOP" if converged else "CONTINUE_T14",
        "policy": asdict(POLICY),
        "available_steps": steps,
        "checkpoint_evidence": evidence,
        "checkpoint_summaries": summaries,
        "windows": windows,
        "consecutive_passing_windows": consecutive,
        "checkpoint_state_loaded": True,
        "sqlite_accessed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "converged": converged,
        # A separate exact-PID controller may consume a PASS receipt.  This
        # read-only module itself cannot send a signal.
        "safe_stop_authorized": converged,
        "process_signal_api_present": False,
    }
