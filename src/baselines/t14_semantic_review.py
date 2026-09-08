"""T14-only saved-ledger review; no model loads, RNG draws, or dispatch.

Opaque mixed-state/RNG hashes do not identify a scientific difference. They
remain evidence gaps rather than being discarded or promoted to parity PASS.
This module intentionally has no configurable tolerance: the old bridge's
cohort-probability tolerance is not a blanket trajectory tolerance.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

SCHEMA = "tastemolnet_t14_componentwise_parity_v2"
OBSERVATIONAL = frozenset({
    "pid", "owner_pid", "walltime_seconds", "elapsed_seconds", "written_at",
    "timestamp", "heartbeat_at", "cache_hit_count", "output_root", "ledger_path",
})
# These contain numerical payloads and cannot be interpreted from a differing
# digest alone. Purely discrete identities (e.g. candidate_universe) stay exact.
OPAQUE = frozenset({
    "rng_state_sha256", "candidate_order_frequency_sha256",
    "candidate_records_sha256", "record_semantics_sha256",
    "active_graph_state_sha256", "transition_state_sha256",
})
REQUIRED = frozenset({
    "schema_version", "completed_step", "sequence_id", "rng_state_sha256",
    "selected", "candidate_universe_sha256", "candidate_covering_lists_sha256",
    "module_scientific_state_sha256", "lineage_sha256", "graph_index_sha256",
    "test_loaded", "calibration_loaded",
})


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def atomic_receipt(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as handle:
        for index, raw in enumerate(handle, 1):
            digest.update(raw)
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("completed_step") != index or row.get("sequence_id") != index:
                raise ValueError(f"noncontiguous/invalid ledger row: {path}:{index}")
            if row.get("test_loaded") is True or row.get("calibration_loaded") is True:
                raise ValueError(f"non-train evidence: {path}:{index}")
            rows.append(row)
    return rows, digest.hexdigest()


def _decode(value: Any) -> Any:
    if not isinstance(value, dict) or "type" not in value:
        return value
    kind = value["type"]
    if kind == "mapping":
        items = [(_decode(row["key"]), _decode(row["value"])) for row in value["items"]]
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate semantic mapping key")
        return dict(items)
    if kind in {"list", "tuple", "set", "frozenset"}:
        # Preserve container kind; do not reorder actions or candidate sequences.
        return {"container_type": kind, "values": [_decode(row) for row in value["items"]]}
    if kind == "float":
        parsed = float.fromhex(value["value"])
        if not math.isfinite(parsed):
            raise ValueError("nonfinite semantic number")
        return parsed
    if kind in {"int", "str", "bool"}:
        return value["value"]
    if kind == "none":
        return None
    return value  # array descriptors retain shape, dtype, bytes and digest.


def _differences(left: Any, right: Any, path: str):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.get("type") == "array" or right.get("type") == "array":
            metadata = ("type", "shape", "dtype", "bytes")
            if any(left.get(key) != right.get(key) for key in metadata):
                yield "discrete", path + ".array_metadata", left, right
            elif left.get("sha256") != right.get("sha256"):
                yield "missing_raw", path, left, right
            return
        for key in sorted(set(left) | set(right), key=str):
            subpath = f"{path}.{key}"
            if key not in left or key not in right:
                yield "discrete", subpath, left.get(key), right.get(key)
            else:
                yield from _differences(left[key], right[key], subpath)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield "discrete", path + ".length", len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right)):
            yield from _differences(a, b, f"{path}[{index}]")
    elif type(left) is not type(right):
        yield "discrete", path + ".type", type(left).__name__, type(right).__name__
    elif isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            yield "discrete", path + ".nonfinite", str(left), str(right)
        elif left != right:
            yield "numeric_unqualified", path, left, right
    elif left != right:
        yield "discrete", path, left, right


def compare_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *,
    start_step: int, end_step: int, require_complete_schema: bool = False,
) -> dict[str, Any]:
    if start_step < 1 or end_step < start_step:
        raise ValueError("invalid T14 interval")
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    differing_steps: dict[str, set[int]] = {}
    first_by_category: dict[str, dict[str, Any]] = {}
    top_counts: Counter[str] = Counter()
    missing_scope = []
    for step in range(start_step, end_step + 1):
        if step > len(left) or step > len(right):
            missing_scope.append(step)
            differences = [("discrete", "missing_step", step > len(left), step > len(right))]
        else:
            a, b = left[step - 1], right[step - 1]
            differences = []
            if require_complete_schema:
                for role, row in (("reference", a), ("candidate", b)):
                    for key in sorted(REQUIRED - set(row)):
                        differences.append(("missing_raw", f"{role}.missing_required.{key}", None, None))
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(("discrete", key, a.get(key), b.get(key)))
                elif a[key] != b[key]:
                    top_counts[key] += 1
                    if key in OBSERVATIONAL:
                        differences.append(("observational", key, a[key], b[key]))
                    elif key in OPAQUE:
                        differences.append(("missing_raw", key, a[key], b[key]))
                    else:
                        differences.extend(_differences(_decode(a[key]), _decode(b[key]), key))
        for category, path, a, b in differences:
            row = {"step": step, "field": path, "reference": a, "candidate": b}
            if category == "numeric_unqualified":
                row.update(max_abs_error=abs(a - b), max_relative_error=abs(a - b) / max(abs(a), 1e-300),
                           tolerance_result="UNQUALIFIED_NO_TRAJECTORY_TOLERANCE_DEFINED")
            first_by_category.setdefault(category, row)
            token = (category, path)
            if token not in summaries:
                summaries[token] = {"category": category, "field": path, "count": 0,
                                    "first_step": step, "last_step": step, "first_difference": row}
            summary = summaries[token]
            summary["count"] += 1
            summary["last_step"] = step
            if category == "numeric_unqualified":
                summary["max_abs_error"] = max(summary.get("max_abs_error", 0.0), abs(a - b))
            differing_steps.setdefault(category, set()).add(step)
    scientific_categories = {"discrete", "missing_raw", "numeric_unqualified"}
    scientific_steps = sorted(set().union(*(differing_steps.get(key, set()) for key in scientific_categories)))
    state = "FAILED" if "discrete" in first_by_category else (
        "EVIDENCE_INCOMPLETE" if scientific_steps else "PASS")
    return {
        "schema_version": SCHEMA, "status": state,
        "start_step": start_step, "end_step": end_step,
        "examined_step_count": end_step - start_step + 1,
        "missing_steps": missing_scope,
        "first_semantic_divergence_step": scientific_steps[0] if scientific_steps else None,
        "first_true_discrete_difference": first_by_category.get("discrete"),
        "first_numeric_difference": first_by_category.get("numeric_unqualified"),
        "first_missing_raw_evidence": first_by_category.get("missing_raw"),
        "discrete_state_exact": "discrete" not in first_by_category,
        "differing_fields": sorted(top_counts), "top_level_differing_step_counts": dict(top_counts),
        "category_step_counts": {key: len(value) for key, value in differing_steps.items()},
        "field_review": [summaries[key] for key in sorted(summaries)],
        "numeric_tolerance_changed": False,
        "numeric_policy": "EXACT_VALUES_OR_EXISTING_SCOPE_SPECIFIC_CONTRACT_REQUIRED",
        "opaque_hash_mismatch_is_scientific_failure": False,
        "all_original_required_conditions_pass": state == "PASS",
        "new_science_transitions": 0,
    }


def review_three_ledgers(reference: Path, continuous: Path, reload: Path, *, output_root: Path) -> dict[str, Any]:
    """Persist each comparison before aggregate disposition, including failures."""
    output_root.mkdir(parents=True, exist_ok=False)
    source = {"reference": reference, "continuous": continuous, "reload": reload}
    data = {key: read_ledger(path) for key, path in source.items()}
    receipts = {}
    scopes = (("reference_vs_lowmemory_1_500", "reference", "continuous", 1, 500),
              ("continuous_vs_reload_1_500", "continuous", "reload", 1, 500),
              ("continuous_vs_reload_501_510", "continuous", "reload", 501, 510))
    for name, left, right, start, end in scopes:
        result = compare_rows(data[left][0], data[right][0], start_step=start, end_step=end, require_complete_schema=True)
        result.update(reference=str(source[left]), candidate=str(source[right]),
                      reference_sha256=data[left][1], candidate_sha256=data[right][1],
                      origin="reconstructed-from-existing-evidence", created_at=datetime.now(timezone.utc).isoformat())
        result = seal(result)
        atomic_receipt(output_root / f"{name}.json", result)
        receipts[name] = {"path": str(output_root / f"{name}.json"), "status": result["status"],
                          "receipt_sha256": result["receipt_sha256"],
                          "first_true_discrete_difference": result["first_true_discrete_difference"],
                          "category_step_counts": result["category_step_counts"]}
    status = "FAILED" if any(row["status"] == "FAILED" for row in receipts.values()) else (
        "PASS" if all(row["status"] == "PASS" for row in receipts.values()) else "EVIDENCE_INCOMPLETE")
    summary = seal({"schema_version": SCHEMA, "status": status, "receipts": receipts,
                    "all_three_receipts_written_before_disposition": True,
                    "original_artifacts_modified": False, "formal_dispatch_authorized": False,
                    "new_science_transitions": 0, "created_at": datetime.now(timezone.utc).isoformat()})
    atomic_receipt(output_root / "review.json", summary)
    return summary


def diagnostic_transition_admission(*, start_boundaries: Mapping[str, int], end_step: int, used: int = 0) -> dict[str, Any]:
    """Count access replay too; a named endpoint is not an executable start."""
    if not start_boundaries or used < 0 or used > 64:
        raise ValueError("invalid T14 diagnostic budget")
    rows = []
    for arm, boundary in start_boundaries.items():
        if type(boundary) is not int or boundary < 0 or end_step <= boundary:
            raise ValueError("invalid diagnostic checkpoint boundary")
        rows.append({"arm": arm, "restored_completed_step": boundary,
                     "first_new_transition": boundary + 1, "last_new_transition": end_step,
                     "new_transitions": end_step - boundary})
    requested = sum(row["new_transitions"] for row in rows)
    return {"status": "PASS" if used + requested <= 64 else "BLOCKED_DIAGNOSTIC_TRANSITION_BUDGET",
            "hard_cap": 64, "already_used": used, "requested": requested,
            "shortfall": max(0, used + requested - 64), "arms": rows,
            "access_replay_included": True, "science_launched": False}
