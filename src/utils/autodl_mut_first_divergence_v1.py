"""Read-only first-divergence audit for sealed Mut ComRecGC 500-step runs.

The historical checkpoint-instrumentation comparator raised before it could
write a structured failure when candidate counts differed.  This module reads
only sealed JSON/JSONL evidence, verifies trace chunk hashes, and makes the
scientific distinction explicit:

* a real candidate/action difference is ``SCIENTIFIC_STATE_DIVERGENCE``;
* a pair with different commits, resume modes, or trace modes is not valid
  evidence about the trace flag, even when a scientific divergence is found.

It deliberately does not import or unpickle ``counterfactuals.pt``.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "mut_first_divergence_audit_v1"
CONTRACT_SCHEMA = "mut_contract_comparison_v1"
CLASSIFICATIONS = frozenset(
    {
        "TRACE_ALIAS_ONLY",
        "CANONICAL_SERIALIZATION_ONLY",
        "OUTPUT_ORDER_ONLY",
        "COMPARATOR_BUG",
        "SCIENTIFIC_STATE_DIVERGENCE",
    }
)
NON_SCIENTIFIC_TRACE_FIELDS = frozenset(
    {
        "source_official_hash",
        "target_official_hash",
        "official_graph_hash",
        "timestamp",
        "written_at",
        "pid",
        "runtime_seconds",
    }
)
TRACE_SCIENCE_FIELDS = (
    "event",
    "move_index",
    "head_index",
    "parent_id",
    "source_graph_sha256",
    "action",
    "target_graph_sha256",
    "prediction",
    "source_probability",
    "strict_flip",
    "accept_reject",
    "duplicate_key",
    "candidate_frequency_digest",
    "registry_digest",
    "lineage_digest",
    "eviction_count",
)


class MutDivergenceAuditError(RuntimeError):
    """The sealed audit evidence is incomplete or inconsistent."""


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(dict(value)) + b"\n")


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MutDivergenceAuditError(f"missing physical JSON evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MutDivergenceAuditError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise MutDivergenceAuditError(f"expected one JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise MutDivergenceAuditError(f"missing physical JSONL evidence: {path}")
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise MutDivergenceAuditError(
            f"invalid JSONL evidence at {path}:{line_number}"
        ) from exc
    return rows


def _trace_root(run_root: Path) -> Path:
    for candidate in (run_root / "trace", run_root / "_native_aux/trace"):
        if (candidate / "selected_action_trace_manifest.json").is_file():
            return candidate
    raise MutDivergenceAuditError(f"selected-action trace missing below {run_root}")


def _selected_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_root = _trace_root(run_root)
    manifest_path = trace_root / "selected_action_trace_manifest.json"
    manifest = _json(manifest_path)
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise MutDivergenceAuditError("selected trace manifest has no chunk list")
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for raw in chunks:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise MutDivergenceAuditError("selected trace chunk row is malformed")
        relative = Path(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise MutDivergenceAuditError("selected trace chunk escapes trace root")
        path = trace_root / relative
        observed_sha = file_sha256(path)
        expected_sha = str(raw.get("sha256") or "")
        if expected_sha and observed_sha != expected_sha:
            raise MutDivergenceAuditError(f"selected trace chunk SHA changed: {path}")
        chunk_rows = _jsonl(path)
        if len(chunk_rows) != int(raw.get("row_count", -1)):
            raise MutDivergenceAuditError(f"selected trace chunk count changed: {path}")
        rows.extend(chunk_rows)
        inventory.append(
            {
                "path": str(path),
                "sha256": observed_sha,
                "row_count": len(chunk_rows),
            }
        )
    if len(rows) != int(manifest.get("row_count", -1)):
        raise MutDivergenceAuditError("selected trace manifest total changed")
    return rows, {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "row_count": len(rows),
        "chunks": inventory,
    }


def _lineage_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _trace_root(run_root) / "candidate_action_lineage_index.jsonl"
    rows = _jsonl(path)
    return rows, {"path": str(path), "sha256": file_sha256(path), "row_count": len(rows)}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _contains_not_recorded(value: Any) -> bool:
    if value == "NOT_RECORDED":
        return True
    if isinstance(value, Mapping):
        return any(_contains_not_recorded(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_not_recorded(item) for item in value)
    return False


def _arm_contract(
    run_root: Path,
    *,
    task_spec: Mapping[str, Any] | None,
    dataset_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    manifest = _json(manifest_path)
    receipt_path = run_root / "semantic_lineage_finalizer_receipt.json"
    receipt = _json(receipt_path)
    dataset = _mapping(manifest.get("dataset_audit"))
    parameters = _mapping(manifest.get("parameters"))
    runtime = _mapping(manifest.get("runtime_environment"))
    trace = _mapping(manifest.get("trace_summary"))
    closure = _mapping(trace.get("frozen_payload_closure"))
    science_contract = _mapping(
        task_spec.get("science_contract") if task_spec is not None else None
    )
    input_hashes = _mapping(
        task_spec.get("input_hashes") if task_spec is not None else None
    )
    summary = dataset_summary or {}
    split_hashes = {
        "train_ids_sha256": summary.get("train_ids_hash", "NOT_RECORDED"),
        "validation_ids_sha256": summary.get("val_ids_hash", "NOT_RECORDED"),
    }
    resumed_from = manifest.get("resumed_from_checkpoint")
    resume_mode = "RESUMED_FROM_CHECKPOINT" if resumed_from else "FRESH_CONTINUOUS"
    return {
        "root": str(run_root),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": dataset.get("dataset_fingerprint", "NOT_RECORDED"),
        "split_hashes": split_hashes,
        "source_cohort_sha256": manifest.get(
            "generation_parent_ids_sha256",
            dataset.get("generation_parent_ids_sha256", "NOT_RECORDED"),
        ),
        "rf_oracle_sha256": science_contract.get(
            "rf_oracle_sha256", input_hashes.get("mut_rf_oracle", "NOT_RECORDED")
        ),
        "gnn_checkpoint_sha256": _mapping(manifest.get("gnn")).get(
            "checkpoint_sha256", "NOT_RECORDED"
        ),
        "distance_checkpoint_sha256": _mapping(manifest.get("distance_model")).get(
            "checkpoint_sha256", "NOT_RECORDED"
        ),
        "config_sha256": manifest.get("config_sha256", "NOT_RECORDED"),
        "algorithm_project_commit": manifest.get("project_commit", "NOT_RECORDED"),
        "source_algorithm_commit": receipt.get(
            "source_algorithm_commit", "NOT_RECORDED"
        ),
        "upstream_commit": manifest.get("upstream_commit", "NOT_RECORDED"),
        "pythonhashseed": runtime.get("pythonhashseed", "NOT_RECORDED"),
        "seed": parameters.get("seed", "NOT_RECORDED"),
        "fresh_resume_mode": resume_mode,
        "resumed_from_checkpoint": resumed_from,
        "candidate_capacity": parameters.get("candidate_capacity", "NOT_RECORDED"),
        "trace_mode": "on" if manifest.get("trace_enabled") is True else "off",
        "graph_identity_codec": trace.get("graph_identity_mode", "NOT_RECORDED"),
        "graph_serialization_codec": closure.get(
            "graph_serialization_version", closure.get("policy", "NOT_RECORDED")
        ),
        "scientific_parameters": dict(parameters),
        "scientific_parameters_sha256": stable_sha256(parameters),
        "test_loaded": manifest.get("test_loaded"),
        "calibration_loaded": manifest.get("calibration_loaded"),
        "candidate_count": manifest.get("counterfactual_candidate_count"),
        "semantic_transition_count": receipt.get("semantic_transition_count"),
        "semantic_transition_alias_event_count": receipt.get(
            "semantic_transition_alias_event_count"
        ),
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def _contract_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    # ``trace_mode`` is the single intentional treatment difference.  It must
    # be an on/off pair, but requiring equality here would make the causal gate
    # impossible to satisfy by construction.
    same_contract_fields = (
        "dataset_sha256",
        "split_hashes",
        "source_cohort_sha256",
        "rf_oracle_sha256",
        "gnn_checkpoint_sha256",
        "distance_checkpoint_sha256",
        "config_sha256",
        "algorithm_project_commit",
        "source_algorithm_commit",
        "upstream_commit",
        "pythonhashseed",
        "seed",
        "fresh_resume_mode",
        "candidate_capacity",
        "graph_identity_codec",
        "graph_serialization_codec",
        "scientific_parameters_sha256",
    )
    rows = {
        field: {
            "legacy": left.get(field),
            "instrumented": right.get(field),
            "equal": left.get(field) == right.get(field),
        }
        for field in (*same_contract_fields, "trace_mode")
    }
    mismatches = [field for field in same_contract_fields if not rows[field]["equal"]]
    not_recorded = [
        field
        for field in same_contract_fields
        for row in (rows[field],)
        if _contains_not_recorded(row["legacy"])
        or _contains_not_recorded(row["instrumented"])
    ]
    same_scientific_contract = not mismatches and not not_recorded
    trace_pair_is_on_off = {left.get("trace_mode"), right.get("trace_mode")} == {
        "on",
        "off",
    }
    result = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "PASS" if same_scientific_contract and trace_pair_is_on_off else "FAIL",
        "legacy": dict(left),
        "instrumented": dict(right),
        "field_comparison": rows,
        "mismatch_fields": mismatches,
        "not_recorded_fields": not_recorded,
        "same_scientific_contract_except_trace_mode": same_scientific_contract,
        # Compatibility alias: exact here means exact modulo the declared
        # treatment variable, never that trace_mode itself is equal.
        "exact_contract_match": same_scientific_contract,
        "trace_pair_is_on_off": trace_pair_is_on_off,
        "eligible_as_trace_mode_equivalence_evidence": (
            same_scientific_contract and trace_pair_is_on_off
        ),
    }
    result["report_sha256"] = stable_sha256(result)
    return result


def _science_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in TRACE_SCIENCE_FIELDS
        if field in row
    }


def _field_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {"legacy": left.get(key), "instrumented": right.get(key)}
        for key in sorted((set(left) | set(right)) - NON_SCIENTIFIC_TRACE_FIELDS)
        if left.get(key) != right.get(key)
    }


def _first_trace_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for index, (legacy, instrumented) in enumerate(zip(left, right, strict=False)):
        differences = _field_delta(legacy, instrumented)
        if differences:
            move = legacy.get("move_index", instrumented.get("move_index"))
            return {
                "row_index": index,
                "generation_step": int(move) + 1 if isinstance(move, int) else None,
                "move_index": move,
                "head_index": legacy.get(
                    "head_index", instrumented.get("head_index")
                ),
                "parent_id": legacy.get("parent_id", instrumented.get("parent_id")),
                "differences": differences,
                "legacy_science": _science_projection(legacy),
                "instrumented_science": _science_projection(instrumented),
            }
    if len(left) != len(right):
        return {
            "row_index": min(len(left), len(right)),
            "generation_step": None,
            "differences": {
                "trace_row_count": {"legacy": len(left), "instrumented": len(right)}
            },
        }
    return None


def _candidate_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    parent = str(row.get("parent_id") or "")
    stable = str(row.get("stable_graph_sha256") or "")
    if not parent or len(stable) != 64:
        raise MutDivergenceAuditError("candidate lineage lacks parent/canonical SHA")
    return parent, stable


def _first_candidate_sequence_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for index, (legacy, instrumented) in enumerate(zip(left, right, strict=False)):
        if _candidate_identity(legacy) != _candidate_identity(instrumented):
            return {
                "candidate_index": index,
                "legacy_identity": list(_candidate_identity(legacy)),
                "instrumented_identity": list(_candidate_identity(instrumented)),
            }
    if len(left) != len(right):
        return {
            "candidate_index": min(len(left), len(right)),
            "legacy_identity": None,
            "instrumented_identity": None,
            "row_count_difference": True,
        }
    return None


def _counter(rows: Iterable[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(_candidate_identity(row) for row in rows)


def _multiset_digest(counter: Counter[tuple[str, str]]) -> str:
    return stable_sha256(
        [
            {"parent_id": parent, "stable_graph_sha256": graph, "count": count}
            for (parent, graph), count in sorted(counter.items())
        ]
    )


def _write_candidate_delta(
    path: Path,
    legacy: Counter[tuple[str, str]],
    instrumented: Counter[tuple[str, str]],
) -> None:
    rows: list[dict[str, Any]] = []
    for side, delta in (
        ("LEGACY_ONLY", legacy - instrumented),
        ("INSTRUMENTED_ONLY", instrumented - legacy),
    ):
        for (parent, graph), count in sorted(delta.items()):
            rows.append(
                {
                    "side": side,
                    "parent_id": parent,
                    "stable_graph_sha256": graph,
                    "multiplicity": count,
                }
            )
    buffer = ["side,parent_id,stable_graph_sha256,multiplicity\n"]
    for row in rows:
        buffer.append(
            f'{row["side"]},{row["parent_id"]},{row["stable_graph_sha256"]},{row["multiplicity"]}\n'
        )
    _atomic_bytes(path, "".join(buffer).encode("utf-8"))


def _write_transition_delta(
    path: Path,
    legacy: Sequence[Mapping[str, Any]],
    instrumented: Sequence[Mapping[str, Any]],
) -> int:
    fields = (
        "row_index",
        "generation_step",
        "move_index",
        "head_index",
        "parent_id",
        "field",
        "legacy",
        "instrumented",
    )
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(legacy, instrumented, strict=False)):
        move = left.get("move_index", right.get("move_index"))
        for field, values in _field_delta(left, right).items():
            rows.append(
                {
                    "row_index": index,
                    "generation_step": int(move) + 1 if isinstance(move, int) else "",
                    "move_index": move,
                    "head_index": left.get("head_index", right.get("head_index")),
                    "parent_id": left.get("parent_id", right.get("parent_id")),
                    "field": field,
                    "legacy": json.dumps(values["legacy"], sort_keys=True),
                    "instrumented": json.dumps(
                        values["instrumented"], sort_keys=True
                    ),
                }
            )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def audit_mut_first_divergence(
    *,
    legacy_root: str | Path,
    instrumented_root: str | Path,
    output_dir: str | Path,
    task_spec_path: str | Path | None = None,
    dataset_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit two sealed roots and atomically publish a fail-closed diagnosis."""

    legacy = Path(legacy_root).expanduser().resolve()
    instrumented = Path(instrumented_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"audit output must be fresh: {output}")
    output.mkdir(parents=True)
    task_spec = _json(Path(task_spec_path).expanduser().resolve()) if task_spec_path else None
    dataset_summary = (
        _json(Path(dataset_summary_path).expanduser().resolve())
        if dataset_summary_path
        else None
    )

    legacy_contract = _arm_contract(
        legacy, task_spec=task_spec, dataset_summary=dataset_summary
    )
    instrumented_contract = _arm_contract(
        instrumented, task_spec=task_spec, dataset_summary=dataset_summary
    )
    contract = _contract_comparison(legacy_contract, instrumented_contract)
    atomic_json(output / "mut_contract_comparison.json", contract)

    legacy_trace, legacy_trace_manifest = _selected_rows(legacy)
    instrumented_trace, instrumented_trace_manifest = _selected_rows(instrumented)
    legacy_lineage, legacy_lineage_manifest = _lineage_rows(legacy)
    instrumented_lineage, instrumented_lineage_manifest = _lineage_rows(instrumented)
    first_transition = _first_trace_difference(legacy_trace, instrumented_trace)
    first_candidate = _first_candidate_sequence_difference(
        legacy_lineage, instrumented_lineage
    )
    legacy_candidates = _counter(legacy_lineage)
    instrumented_candidates = _counter(instrumented_lineage)
    legacy_only = legacy_candidates - instrumented_candidates
    instrumented_only = instrumented_candidates - legacy_candidates
    multiset_equal = legacy_candidates == instrumented_candidates
    _write_candidate_delta(
        output / "mut_candidate_delta.csv", legacy_candidates, instrumented_candidates
    )
    transition_delta_count = _write_transition_delta(
        output / "mut_transition_delta.csv", legacy_trace, instrumented_trace
    )

    if not multiset_equal or first_transition is not None:
        classification = "SCIENTIFIC_STATE_DIVERGENCE"
    elif first_candidate is not None:
        classification = "OUTPUT_ORDER_ONLY"
    else:
        any_raw_difference = any(
            left != right
            for left, right in zip(legacy_trace, instrumented_trace, strict=True)
        )
        if any_raw_difference:
            classification = "TRACE_ALIAS_ONLY"
        elif legacy_contract["graph_serialization_codec"] != instrumented_contract[
            "graph_serialization_codec"
        ]:
            classification = "CANONICAL_SERIALIZATION_ONLY"
        else:
            classification = "COMPARATOR_BUG"
    if classification not in CLASSIFICATIONS:  # pragma: no cover - invariant
        raise AssertionError(classification)

    eligible = bool(contract["eligible_as_trace_mode_equivalence_evidence"])
    route_b_admissible = classification == "SCIENTIFIC_STATE_DIVERGENCE" and eligible
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS_AUDIT_COMPLETE",
        "classification": classification,
        "legacy_root": str(legacy),
        "instrumented_root": str(instrumented),
        "contract_comparison_path": str(output / "mut_contract_comparison.json"),
        "contract_comparison_sha256": file_sha256(
            output / "mut_contract_comparison.json"
        ),
        "selected_trace": {
            "legacy": legacy_trace_manifest,
            "instrumented": instrumented_trace_manifest,
        },
        "candidate_lineage": {
            "legacy": legacy_lineage_manifest,
            "instrumented": instrumented_lineage_manifest,
        },
        "first_transition_divergence": first_transition,
        "first_candidate_sequence_divergence": first_candidate,
        "candidate_universe": {
            "legacy_count": len(legacy_lineage),
            "instrumented_count": len(instrumented_lineage),
            "delta_instrumented_minus_legacy": len(instrumented_lineage)
            - len(legacy_lineage),
            "legacy_multiset_sha256": _multiset_digest(legacy_candidates),
            "instrumented_multiset_sha256": _multiset_digest(
                instrumented_candidates
            ),
            "multiset_equal": multiset_equal,
            "legacy_only_multiplicity": sum(legacy_only.values()),
            "instrumented_only_multiplicity": sum(instrumented_only.values()),
        },
        "transition_delta_row_count": transition_delta_count,
        "candidate_delta_path": str(output / "mut_candidate_delta.csv"),
        "transition_delta_path": str(output / "mut_transition_delta.csv"),
        "2250_vs_2255_explicitly_accounted_for": (
            len(legacy_lineage) == 2250 and len(instrumented_lineage) == 2255
        ),
        "output_order_only_permitted": multiset_equal,
        "current_pair_trace_mode_causal_claim_permitted": eligible,
        "route_b_gate": {
            "route_b_admissible": route_b_admissible,
            "requires_scientific_state_divergence": True,
            "requires_same_contract_trace_on_off_pair": True,
            "next_required_action": (
                "BUILD_ROUTE_B_FRESH_50K_SPEC"
                if route_b_admissible
                else "RUN_FRESH_SAME_COMMIT_SEQUENTIAL_TRACE_ON_OFF_A_B"
            ),
            "fresh_50k_started": False,
        },
        "interpretation": (
            "A real scientific transition and candidate-universe divergence is "
            "present, but this legacy/instrumented pair has different execution "
            "contracts and both arms are trace-on. It cannot identify the trace "
            "flag as the cause."
            if classification == "SCIENTIFIC_STATE_DIVERGENCE" and not eligible
            else "Classification follows the sealed JSON trace and lineage evidence."
        ),
        "paper_eligible": False,
    }
    report["report_sha256"] = stable_sha256(report)
    atomic_json(output / "mut_first_divergence.json", report)
    atomic_json(
        output / "terminal.json",
        {
            "schema_version": "mut_first_divergence_terminal_v1",
            "status": "PASS_AUDIT_COMPLETE",
            "classification": classification,
            "report_path": str(output / "mut_first_divergence.json"),
            "report_sha256": file_sha256(output / "mut_first_divergence.json"),
            "route_b_admissible": route_b_admissible,
            "fresh_50k_started": False,
        },
    )
    _atomic_bytes(output / "PASS", b"Mut first-divergence audit complete.\n")
    return report


__all__ = [
    "CLASSIFICATIONS",
    "CONTRACT_SCHEMA",
    "MutDivergenceAuditError",
    "SCHEMA",
    "audit_mut_first_divergence",
    "file_sha256",
    "stable_sha256",
]
