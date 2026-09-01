"""Small fail-closed primitives for the fast Mutagenicity ComRecGC route.

This module deliberately contains no scheduler, signal, or experiment-writing
code.  It normalizes the evidence needed by the authorized Mut continuation:

* cgroup-v1/v2 memory snapshots with explicit missing fields;
* supersession of the unmeasured 440-GiB launch gate;
* the preregistered empirical-memory formula;
* conditional adoption of the *truthful trace-on* historical 50k result after
  a complete 500-step semantic-equivalence receipt;
* an explicit payload -> pair store -> DBSCAN transitive binding;
* read-only common-recourse adoption, convergence, and capacity contracts.

The historical manifest never gains fields it did not originally contain.
In particular, conditional adoption records ``trace_enabled=true``,
``trace_parity_passed=false``, and ``traceoff_reference_rerun=false``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


GIB = 1024**3
STATIC_440_GIB = 440
STATIC_440_BYTES = STATIC_440_GIB * GIB
EMPIRICAL_ADMISSION_AUTHORITY = "ALLOW_MUT_EMPIRICAL_MEMORY_ADMISSION=1"
ADOPTION_WITHOUT_FULL_PARITY_AUTHORITY = (
    "ALLOW_MUT_ARTIFACT_ADOPTION_WITHOUT_FULL_50K_PARITY_RERUN=1"
)
MUT_STEPS = 50_000
MUT_CANDIDATE_CAPACITY = 100_000
MUT_SEED = 0
EQUIVALENCE_STEPS = 500

_SHA256 = re.compile(r"[0-9a-f]{64}")


class MutFastAccurateV2Error(RuntimeError):
    """Evidence is malformed or cannot support the requested conclusion."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if not _is_int(value) or int(value) < minimum:
        raise MutFastAccurateV2Error(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MutFastAccurateV2Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MutFastAccurateV2Error(f"{label} is outside its allowed range")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MutFastAccurateV2Error(
            f"{label} must be one lowercase 64-character SHA-256 digest"
        )
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MutFastAccurateV2Error(f"{label} must be one object")
    return value


def _at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(value: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        result = _at(value, path)
        if result is not None:
            return result
    return None


def _required_false(value: Mapping[str, Any], field: str) -> None:
    if _at(value, field) is not False:
        raise MutFastAccurateV2Error(f"{field} must be explicitly false")


def _required_true(value: Mapping[str, Any], field: str) -> None:
    if _at(value, field) is not True:
        raise MutFastAccurateV2Error(f"{field} must be explicitly true")


def _raw_text(value: Any, *, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MutFastAccurateV2Error(f"{label} is not UTF-8") from exc
    if not isinstance(value, str):
        raise MutFastAccurateV2Error(f"{label} must be text")
    return value


def _parse_counter(raw: str, *, label: str, allow_max: bool = False) -> int | None:
    text = raw.strip()
    if allow_max and text == "max":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise MutFastAccurateV2Error(f"{label} is not an integer") from exc
    if value < 0:
        raise MutFastAccurateV2Error(f"{label} must be nonnegative")
    return value


def _parse_key_values(raw: str, *, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2 or fields[0] in result:
            raise MutFastAccurateV2Error(
                f"{label} has a malformed or duplicate row at line {line_number}"
            )
        result[fields[0]] = _parse_counter(
            fields[1], label=f"{label}.{fields[0]}"
        ) or 0
    return result


def _parse_pressure(raw: str, *, label: str) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        kind = fields[0]
        if kind not in {"some", "full"} or kind in result:
            raise MutFastAccurateV2Error(
                f"{label} has an invalid pressure row at line {line_number}"
            )
        row: dict[str, float | int] = {}
        for token in fields[1:]:
            if "=" not in token:
                raise MutFastAccurateV2Error(
                    f"{label} has a malformed token at line {line_number}"
                )
            key, raw_value = token.split("=", 1)
            if key in row:
                raise MutFastAccurateV2Error(f"{label}.{kind}.{key} is duplicated")
            try:
                row[key] = int(raw_value) if key == "total" else float(raw_value)
            except ValueError as exc:
                raise MutFastAccurateV2Error(
                    f"{label}.{kind}.{key} is malformed"
                ) from exc
        result[kind] = row
    return result


_V2_FILES = (
    "memory.max",
    "memory.high",
    "memory.current",
    "memory.peak",
    "memory.stat",
    "memory.events",
    "memory.events.local",
    "memory.pressure",
    "pids.current",
    "pids.max",
)
_V1_FILES = (
    "memory.limit_in_bytes",
    "memory.soft_limit_in_bytes",
    "memory.usage_in_bytes",
    "memory.max_usage_in_bytes",
    "memory.stat",
    "memory.failcnt",
    "memory.oom_control",
    "memory.pressure",
    "pids.current",
    "pids.max",
)


def parse_cgroup_snapshot(
    files: Mapping[str, str | bytes | None], *, version: str | int = "auto"
) -> dict[str, Any]:
    """Parse one cgroup memory snapshot without hiding absent kernel fields.

    ``files`` maps basename to file contents.  Missing basenames and explicit
    ``None`` values are both retained in ``missing_fields`` and ``field_state``.
    Malformed present fields raise instead of being treated as zero.
    """

    if version == "auto":
        has_v2 = any(files.get(name) is not None for name in _V2_FILES[:4])
        has_v1 = any(files.get(name) is not None for name in _V1_FILES[:4])
        if has_v2 == has_v1:
            raise MutFastAccurateV2Error(
                "cgroup version is ambiguous; provide version='v1' or version='v2'"
            )
        normalized_version = "v2" if has_v2 else "v1"
    else:
        normalized_version = str(version).lower().replace("cgroup", "").strip()
        if normalized_version in {"1", "v1"}:
            normalized_version = "v1"
        elif normalized_version in {"2", "v2"}:
            normalized_version = "v2"
        else:
            raise MutFastAccurateV2Error(f"unsupported cgroup version: {version!r}")

    expected = _V2_FILES if normalized_version == "v2" else _V1_FILES
    raw: dict[str, str | None] = {}
    for name in expected:
        item = files.get(name)
        raw[name] = None if item is None else _raw_text(item, label=name)
    missing = sorted(name for name, item in raw.items() if item is None)
    field_state = {
        name: {"present": item is not None, "raw": item}
        for name, item in raw.items()
    }

    def counter(name: str, *, allow_max: bool = False) -> int | None:
        item = raw.get(name)
        if item is None:
            return None
        return _parse_counter(item, label=name, allow_max=allow_max)

    if normalized_version == "v2":
        memory_max = counter("memory.max", allow_max=True)
        memory_high = counter("memory.high", allow_max=True)
        memory_current = counter("memory.current")
        memory_peak = counter("memory.peak")
        stat_values = (
            _parse_key_values(raw["memory.stat"], label="memory.stat")
            if raw["memory.stat"] is not None
            else {}
        )
        events = (
            _parse_key_values(raw["memory.events"], label="memory.events")
            if raw["memory.events"] is not None
            else {}
        )
        events_local = (
            _parse_key_values(
                raw["memory.events.local"], label="memory.events.local"
            )
            if raw["memory.events.local"] is not None
            else {}
        )
        anon = stat_values.get("anon")
        file_bytes = stat_values.get("file")
        slab_reclaimable = stat_values.get("slab_reclaimable")
        failcnt = None
        oom_control: dict[str, int] = {}
    else:
        memory_max = counter("memory.limit_in_bytes")
        memory_high = counter("memory.soft_limit_in_bytes")
        memory_current = counter("memory.usage_in_bytes")
        memory_peak = counter("memory.max_usage_in_bytes")
        stat_values = (
            _parse_key_values(raw["memory.stat"], label="memory.stat")
            if raw["memory.stat"] is not None
            else {}
        )
        failcnt = counter("memory.failcnt")
        oom_control = (
            _parse_key_values(raw["memory.oom_control"], label="memory.oom_control")
            if raw["memory.oom_control"] is not None
            else {}
        )
        # v1 has no memory.events files.  Keep the closest counters explicit,
        # but never synthesize oom/oom_kill values from failcnt.
        events = {}
        events_local = {}
        anon = stat_values.get("total_rss", stat_values.get("rss"))
        file_bytes = stat_values.get("total_cache", stat_values.get("cache"))
        slab_reclaimable = stat_values.get(
            "total_slab_reclaimable", stat_values.get("slab_reclaimable")
        )

    pressure = (
        _parse_pressure(raw["memory.pressure"], label="memory.pressure")
        if raw.get("memory.pressure") is not None
        else {}
    )
    pids_current = counter("pids.current")
    pids_max = counter("pids.max", allow_max=True)
    headroom = (
        max(0, memory_max - memory_current)
        if memory_max is not None and memory_current is not None
        else None
    )
    event_source = events_local if events_local else events
    return {
        "schema_version": "mut_cgroup_snapshot_v2",
        "cgroup_version": normalized_version,
        "field_state": field_state,
        "missing_fields": missing,
        "memory_max_bytes": memory_max,
        "memory_high_bytes": memory_high,
        "memory_current_bytes": memory_current,
        "memory_peak_bytes": memory_peak,
        "memory_headroom_bytes": headroom,
        "memory_stat": stat_values,
        "anon_bytes": anon,
        "file_bytes": file_bytes,
        "inactive_file_bytes": stat_values.get("inactive_file"),
        "slab_reclaimable_bytes": slab_reclaimable,
        "memory_events": events,
        "memory_events_local": events_local,
        "effective_event_source": (
            "memory.events.local"
            if events_local
            else ("memory.events" if events else None)
        ),
        "events_high": event_source.get("high"),
        "events_max": event_source.get("max"),
        "events_oom": event_source.get("oom"),
        "events_oom_kill": event_source.get("oom_kill"),
        "memory_failcnt": failcnt,
        "memory_oom_control": oom_control,
        "memory_pressure": pressure,
        "pressure_some_avg10": _at(pressure, "some.avg10"),
        "pressure_full_avg10": _at(pressure, "full.avg10"),
        "pids_current": pids_current,
        "pids_max": pids_max,
    }


def read_cgroup_snapshot(
    root: str | Path, *, version: str | int = "auto"
) -> dict[str, Any]:
    """Read the small cgroup files needed by :func:`parse_cgroup_snapshot`."""

    directory = Path(root).expanduser()
    if not directory.is_dir():
        raise MutFastAccurateV2Error(f"cgroup root is not a directory: {directory}")
    selected = version
    if version == "auto":
        if (directory / "memory.max").is_file():
            selected = "v2"
        elif (directory / "memory.limit_in_bytes").is_file():
            selected = "v1"
        else:
            raise MutFastAccurateV2Error("cannot detect cgroup version")
    normalized = str(selected).lower().replace("cgroup", "").strip()
    expected = _V2_FILES if normalized in {"2", "v2"} else _V1_FILES
    values: dict[str, str | None] = {}
    for name in expected:
        path = directory / name
        try:
            values[name] = path.read_text(encoding="utf-8") if path.is_file() else None
        except OSError as exc:
            raise MutFastAccurateV2Error(f"cannot read cgroup field: {path}") from exc
    result = parse_cgroup_snapshot(values, version=selected)
    result["cgroup_root"] = str(directory.resolve())
    return result


def supersede_static_440g_gate(
    *,
    authority: str | bool,
    old_required_free_bytes: int | None = None,
    old_required_free_gib: int | float | None = None,
) -> dict[str, Any]:
    """Close the exact old 440-GiB gate without inventing a replacement."""

    authorized = authority is True or authority == EMPIRICAL_ADMISSION_AUTHORITY
    if not authorized:
        raise MutFastAccurateV2Error("empirical-memory admission is not authorized")
    if old_required_free_bytes is not None and old_required_free_gib is not None:
        raise MutFastAccurateV2Error("provide the old gate in bytes or GiB, not both")
    if old_required_free_bytes is None and old_required_free_gib is None:
        old_required_free_bytes = STATIC_440_BYTES
    if old_required_free_gib is not None:
        gib = _number(old_required_free_gib, label="old_required_free_gib", minimum=0)
        if not gib.is_integer():
            raise MutFastAccurateV2Error("old_required_free_gib must be integral")
        old_required_free_bytes = int(gib) * GIB
    old_bytes = _integer(
        old_required_free_bytes,
        label="old_required_free_bytes",
        minimum=1,
    )
    if old_bytes != STATIC_440_BYTES:
        raise MutFastAccurateV2Error(
            "only the exact unmeasured 440-GiB gate may be superseded"
        )
    return {
        "schema_version": "mut_static_memory_gate_supersession_v2",
        "state": "SUPERSEDED_UNMEASURED_STATIC_GATE",
        "authority": EMPIRICAL_ADMISSION_AUTHORITY,
        "old_required_free_memory_gib": STATIC_440_GIB,
        "old_required_free_memory_bytes": STATIC_440_BYTES,
        "old_gate_enforced": False,
        "replacement_static_gate_bytes": None,
        "empirical_memory_admission_required": True,
    }


# A spelling that reads naturally in callers and remains discoverable.
build_static_440_supersession = supersede_static_440g_gate


def derive_empirical_memory_admission(
    *,
    cgroup_memory_peak_bytes: int,
    process_peak_rss_bytes: int,
    checkpoint_peak_bytes: int,
    memory_event_deltas: Mapping[str, int] | None = None,
    protected_task_slowdown_fraction: float = 0.0,
    semantic_equivalence_pass: bool = True,
    checkpoint_reload_pass: bool = True,
) -> dict[str, Any]:
    """Apply the preregistered measured-peak formula and health gates."""

    peaks = {
        "cgroup_memory_peak_bytes": _integer(
            cgroup_memory_peak_bytes,
            label="cgroup_memory_peak_bytes",
            minimum=0,
        ),
        "process_peak_rss_bytes": _integer(
            process_peak_rss_bytes,
            label="process_peak_rss_bytes",
            minimum=0,
        ),
        "checkpoint_peak_bytes": _integer(
            checkpoint_peak_bytes,
            label="checkpoint_peak_bytes",
            minimum=0,
        ),
    }
    if any(value <= 0 for value in peaks.values()):
        raise MutFastAccurateV2Error("all three measured peaks must be positive")
    slowdown = _number(
        protected_task_slowdown_fraction,
        label="protected_task_slowdown_fraction",
        minimum=0.0,
    )
    raw_events = dict(memory_event_deltas or {})
    events: dict[str, int] = {}
    for key in ("high", "max", "oom", "oom_kill"):
        events[key] = _integer(raw_events.get(key, 0), label=f"events.{key}")
    peak_bytes = max(peaks.values())
    peak_gib = peak_bytes / GIB
    blockers: list[str] = []
    if peak_bytes > 40 * GIB:
        blockers.append("UNEXPECTED_MUT_MEMORY_PEAK_GT_40_GIB")
    if any(events[key] > 0 for key in ("max", "oom", "oom_kill")):
        blockers.append("CGROUP_MEMORY_LIMIT_EVENT_INCREASED")
    if slowdown > 0.10:
        blockers.append("PROTECTED_TASK_SLOWDOWN_GT_10_PERCENT")
    if semantic_equivalence_pass is not True:
        blockers.append("SEMANTIC_EQUIVALENCE_NOT_PASS")
    if checkpoint_reload_pass is not True:
        blockers.append("CHECKPOINT_RELOAD_NOT_PASS")

    full_max_gib: int | None = None
    full_high_gib: int | None = None
    parent_headroom_gib: int | None = None
    if not blockers:
        unrounded = max(48 * GIB, min(128 * GIB, 3 * peak_bytes + 16 * GIB))
        full_max_gib = math.ceil(unrounded / GIB)
        full_high_gib = math.floor(0.75 * full_max_gib)
        parent_headroom_gib = full_max_gib + 16
    return {
        "schema_version": "mut_empirical_memory_admission_v2",
        "status": "PASS" if not blockers else "BLOCKED",
        "reason": "EMPIRICAL_MEMORY_ADMISSION_PASS" if not blockers else blockers[0],
        "blockers": blockers,
        "measured_peaks": peaks,
        "peak_bytes": peak_bytes,
        "peak_gib": peak_gib,
        "memory_event_deltas": events,
        "protected_task_slowdown_fraction": slowdown,
        "semantic_equivalence_pass": semantic_equivalence_pass is True,
        "checkpoint_reload_pass": checkpoint_reload_pass is True,
        "full_memory_max_gib": full_max_gib,
        "full_memory_high_gib": full_high_gib,
        "parent_required_headroom_gib": parent_headroom_gib,
        "full_memory_max_bytes": (
            None if full_max_gib is None else full_max_gib * GIB
        ),
        "full_memory_high_bytes": (
            None if full_high_gib is None else full_high_gib * GIB
        ),
        "parent_required_headroom_bytes": (
            None if parent_headroom_gib is None else parent_headroom_gib * GIB
        ),
        "static_440_gib_gate_used": False,
    }


_EQUIVALENCE_CHECKS = (
    "rng_state",
    "selected_head",
    "parent_id",
    "action",
    "candidate_graph_hash",
    "prediction",
    "source_probability",
    "strict_flip",
    "accept_reject",
    "duplicate_decision",
    "candidate_frequency",
    "lineage_predecessor",
    "lineage_downstream",
    "serialized_candidate_multiset",
    "checkpoint_reload",
)


def validate_500_step_semantic_equivalence(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete discrete 500-step legacy/compact comparison."""

    row = _mapping(receipt, label="equivalence receipt")
    if row.get("schema_version") == "mut_checkpoint_instrumentation_equivalence_v1":
        expected = {
            "status": "PASS",
            "paper_eligible": False,
            "dataset": "mutagenicity",
            "steps": EQUIVALENCE_STEPS,
            "seed": MUT_SEED,
            "step_action_trace_exact": True,
            "rng_state_exact": True,
            "checkpoint_mirror_verified": True,
            "checkpoint_resume_exercised": True,
            "calibration_loaded": False,
            "test_loaded": False,
        }
        failures = [
            key
            for key, expected_value in expected.items()
            if row.get(key) != expected_value
        ]
        payload = _mapping(row.get("payload_equivalence"), label="payload_equivalence")
        candidate_parity = _mapping(
            payload.get("candidate_parity"),
            label="payload_equivalence.candidate_parity",
        )
        source_audit = _mapping(row.get("source_audit"), label="source_audit")
        source_delta = _mapping(
            source_audit.get("delta_audit"), label="source_audit.delta_audit"
        )
        legacy = _mapping(source_audit.get("legacy"), label="source_audit.legacy")
        instrumented = _mapping(
            source_audit.get("instrumented"), label="source_audit.instrumented"
        )
        source_commits = (
            legacy.get("project_commit"),
            instrumented.get("project_commit"),
        )
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in source_commits
        ) or source_commits[0] == source_commits[1]:
            failures.append("source_audit.project_commits")
        for label, value in (
            ("legacy.inventory_sha256", legacy.get("inventory_sha256")),
            ("instrumented.inventory_sha256", instrumented.get("inventory_sha256")),
        ):
            try:
                _sha256(value, label=label)
            except MutFastAccurateV2Error:
                failures.append(label)
        if row.get("failures") != []:
            failures.append("failures")
        if payload.get("failures") != [] or candidate_parity.get(
            "trace_parity_passed"
        ) is not True:
            failures.append("payload_equivalence")
        if source_delta.get("status") != "PASS" or source_delta.get("failures") != []:
            failures.append("source_audit.delta_audit")
        summary_sha = row.get("summary_sha256")
        if summary_sha is not None and "path" not in row and "sha256" not in row:
            unsigned = {key: value for key, value in row.items() if key != "summary_sha256"}
            encoded = json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            if summary_sha != hashlib.sha256(encoded).hexdigest():
                failures.append("summary_sha256")
        if failures:
            raise MutFastAccurateV2Error(
                "500-step instrumentation equivalence is invalid: "
                + ",".join(sorted(set(failures)))
            )
        return {
            "schema_version": "mut_500_step_semantic_equivalence_validation_v2",
            "source_schema_version": row["schema_version"],
            "status": "PASS",
            "semantic_equivalence_pass": True,
            "equivalence_steps": EQUIVALENCE_STEPS,
            "step_action_trace_exact": True,
            "rng_state_exact": True,
            "candidate_payload_exact": True,
            "checkpoint_mirror_verified": True,
            "checkpoint_resume_exercised": True,
            "checkpoint_reload_pass": True,
            "calibration_loaded": False,
            "test_loaded": False,
        }

    if row.get("status") != "PASS" or row.get("semantic_equivalence_pass") is not True:
        raise MutFastAccurateV2Error("500-step semantic equivalence did not PASS")
    steps = _first(row, "equivalence_steps", "steps", "completed_steps")
    if steps != EQUIVALENCE_STEPS:
        raise MutFastAccurateV2Error("semantic equivalence must cover exactly 500 steps")
    _required_false(row, "calibration_loaded")
    _required_false(row, "test_loaded")
    checks = _mapping(row.get("checks"), label="equivalence checks")
    failed = [name for name in _EQUIVALENCE_CHECKS if checks.get(name) is not True]
    extras = sorted(str(name) for name, value in checks.items() if value is not True)
    if failed or extras:
        raise MutFastAccurateV2Error(
            "semantic equivalence checks are incomplete: "
            + ",".join(sorted(set(failed + extras)))
        )
    return {
        "schema_version": "mut_500_step_semantic_equivalence_validation_v2",
        "status": "PASS",
        "semantic_equivalence_pass": True,
        "equivalence_steps": EQUIVALENCE_STEPS,
        "checks": {name: True for name in _EQUIVALENCE_CHECKS},
        "checkpoint_reload_pass": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def _lineage_evidence(
    manifest: Mapping[str, Any], explicit: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    if explicit is not None:
        return _mapping(explicit, label="lineage audit")
    value = _first(manifest, "lineage_recovery_audit", "trace_summary.lineage_recovery_audit")
    return _mapping(value, label="lineage audit")


def _closure_evidence(
    manifest: Mapping[str, Any], explicit: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    if explicit is not None:
        return _mapping(explicit, label="closure audit")
    value = _first(
        manifest,
        "frozen_payload_closure",
        "trace_summary.frozen_payload_closure",
    )
    return _mapping(value, label="closure audit")


def verify_historical_50k_artifact(
    manifest: Mapping[str, Any],
    *,
    equivalence_receipt: Mapping[str, Any] | None,
    adoption_without_full_50k_parity_rerun_authorized: bool | str,
    lineage_audit: Mapping[str, Any] | None = None,
    closure_audit: Mapping[str, Any] | None = None,
    no_active_writer: bool,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the historical 50k result without relabelling trace-on as off.

    The authorized fast path is intentionally conditional: the complete
    historical artifact remains trace-on and a separately validated 500-step
    receipt proves that removing trace instrumentation changes no decision.
    It is *not* a completed 50k trace-on/off parity experiment.
    """

    row = _mapping(manifest, label="historical manifest")
    dataset = str(row.get("dataset") or "").lower()
    method = str(row.get("method") or "").lower()
    parameters = _mapping(row.get("parameters"), label="historical parameters")
    steps = _first(row, "generation_steps", "total_steps", "parameters.steps")
    capacity = _first(row, "candidate_capacity", "parameters.candidate_capacity")
    seed = _first(row, "seed", "parameters.seed")
    checks = {
        "dataset": dataset == "mutagenicity",
        "method": method == "comrecgc",
        "steps": steps == MUT_STEPS,
        "candidate_capacity": capacity == MUT_CANDIDATE_CAPACITY,
        "seed": seed == MUT_SEED,
        "generation_complete": _first(
            row, "generation_complete", "run_complete"
        )
        is True,
        "calibration_closed": row.get("calibration_loaded") is False,
        "test_closed": row.get("test_loaded") is False,
        "trace_truthful": row.get("trace_enabled") is True,
        "no_active_writer": no_active_writer is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise MutFastAccurateV2Error(
            "historical 50k contract failed: " + ",".join(failed)
        )

    candidate_count = _integer(
        _first(row, "counterfactual_candidate_count", "candidate_count"),
        label="historical candidate_count",
        minimum=1,
    )
    lineage = _lineage_evidence(row, lineage_audit)
    resolved = _integer(
        _first(
            lineage,
            "candidate_lineage_resolved_count",
            "resolved_candidate_count",
        ),
        label="lineage resolved count",
    )
    lineage_zero_fields = (
        "recorded_action_replay_mismatch_count",
        "predecessor_unverified_conflict_count",
        "predecessor_unresolved_legacy_conflict_count",
        "predecessor_selected_parent_mismatch_count",
        "selected_event_source_parent_mismatch_count",
    )
    bad_lineage = [name for name in lineage_zero_fields if lineage.get(name) != 0]
    if resolved != candidate_count or bad_lineage:
        raise MutFastAccurateV2Error(
            "historical lineage is incomplete: " + ",".join(bad_lineage)
        )
    target_mismatch = _integer(
        lineage.get("selected_event_target_parent_mismatch_count", 0),
        label="selected target-parent mismatch count",
    )
    cross_parent = _integer(
        lineage.get("predecessor_cross_parent_convergence_count", 0),
        label="cross-parent convergence count",
    )
    # ``target_mismatch`` counts selected events, whereas ``cross_parent``
    # counts unique converged predecessor graphs.  The historical authority is
    # therefore truthfully 14 versus 1.  Neither field participates in the
    # recorded source-parent/predecessor choice, whose zero-error fields above
    # remain the fail-closed lineage gate.

    closure = _closure_evidence(row, closure_audit)
    for field in (
        "closure_complete",
        "post_write_reload_verified",
    ):
        _required_true(closure, field)
    for field in (
        "candidate_order_changed",
        "candidate_payload_changed",
        "scientific_parameters_changed",
    ):
        _required_false(closure, field)
    if closure.get("sha_mismatch_count", 0) != 0 or closure.get(
        "unresolved_hash_count", 0
    ) != 0:
        raise MutFastAccurateV2Error("historical frozen payload closure is incomplete")

    authorized = (
        adoption_without_full_50k_parity_rerun_authorized is True
        or adoption_without_full_50k_parity_rerun_authorized
        == ADOPTION_WITHOUT_FULL_PARITY_AUTHORITY
    )
    if not authorized:
        raise MutFastAccurateV2Error(
            "adoption without a full 50k parity rerun is not authorized"
        )
    if equivalence_receipt is None:
        raise MutFastAccurateV2Error(
            "trace-on historical adoption requires a 500-step equivalence receipt"
        )
    equivalence = validate_500_step_semantic_equivalence(equivalence_receipt)

    identity = {
        "dataset_sha": _first(
            row,
            "dataset_sha",
            "source_dataset_fingerprint",
            "dataset_audit.dataset_fingerprint",
        ),
        "split_sha": _first(row, "split_sha", "split_manifest_sha256"),
        "source_cohort_sha": _first(
            row, "source_cohort_sha", "generation_parent_ids_sha256"
        ),
        "oracle_sha": _first(
            row,
            "RF_oracle_sha",
            "oracle_sha",
            "gnn.checkpoint_sha256",
        ),
        "feature_schema_sha": _first(row, "feature_schema_sha"),
        "config_sha": _first(row, "config_sha", "config_sha256"),
        "graph_identity_contract": _first(
            closure, "canonical_hash_algorithm", "graph_identity_contract"
        ),
        "candidate_predicate": _first(row, "candidate_predicate", "cf_mode"),
        "lineage_contract": _first(
            lineage, "schema_version", "lineage_contract"
        ),
        "checkpoint_schema": _first(row, "checkpoint_schema"),
        "candidate_serialization_schema": _first(
            closure, "graph_serialization_version", "candidate_serialization_schema"
        ),
    }
    if expected_identity is not None:
        expected = _mapping(expected_identity, label="expected identity")
        mismatches = [
            key
            for key, expected_value in expected.items()
            if identity.get(str(key)) != expected_value
        ]
        if mismatches:
            raise MutFastAccurateV2Error(
                "historical identity mismatch: " + ",".join(sorted(mismatches))
            )
    missing_identity = sorted(key for key, value in identity.items() if value is None)
    return {
        "schema_version": "mut_historical_50k_verification_v2",
        "status": "PASS",
        "route_a_mode": "CONDITIONAL_TRACE_ON_50K_ADOPTION",
        "dataset": "mutagenicity",
        "method": "comrecgc",
        "generation_steps": MUT_STEPS,
        "candidate_capacity": MUT_CANDIDATE_CAPACITY,
        "seed": MUT_SEED,
        "candidate_count": candidate_count,
        "trace_enabled": True,
        "trace_parity_passed": False,
        "traceoff_reference_rerun": False,
        "adoption_without_full_50k_parity_rerun_authorized": True,
        "semantic_equivalence": equivalence,
        "generation_complete": True,
        "lineage_pass": True,
        "candidate_freeze_pass": True,
        "no_test_leakage": True,
        "no_active_writer": True,
        "identity": identity,
        "missing_identity_fields": missing_identity,
        "target_parent_mismatch_count": target_mismatch,
        "cross_parent_convergence_count": cross_parent,
        "target_parent_mismatch_nonfatal": target_mismatch > 0,
        "target_parent_mismatch_semantics": (
            "canonical_representative_metadata_not_recorded_parent_selection"
            if target_mismatch > 0
            else "none"
        ),
    }


def verify_candidate_universe_binding(
    generation: Mapping[str, Any],
    pair_store: Mapping[str, Any],
    dbscan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the exact payload -> pair vectors -> DBSCAN binding.

    Older manifests do not natively contain three identically named candidate
    universe fields.  This verifier therefore records the real transitive
    chain and never claims that such native fields existed.
    """

    generation = _mapping(generation, label="generation evidence")
    pair_store = _mapping(pair_store, label="pair-store evidence")
    dbscan = _mapping(dbscan, label="DBSCAN evidence")
    if _first(generation, "generation_complete", "run_complete") is not True:
        raise MutFastAccurateV2Error("generation evidence is not complete")
    if _first(pair_store, "run_complete", "pair_store_complete") is not True:
        raise MutFastAccurateV2Error("pair-store evidence is not complete")
    if _first(dbscan, "run_complete", "dbscan_complete") is not True:
        raise MutFastAccurateV2Error("DBSCAN evidence is not complete")

    generation_payload = _sha256(
        _first(generation, "payload_sha256", "counterfactuals_sha256"),
        label="generation payload SHA-256",
    )
    pair_source_payload = _sha256(
        _first(
            pair_store,
            "source_generation_payload_sha256",
            "scientific_identity.counterfactuals_sha256",
            "counterfactuals_sha256",
        ),
        label="pair-store source generation payload SHA-256",
    )
    if generation_payload != pair_source_payload:
        raise MutFastAccurateV2Error(
            "pair store is not bound to the exact generation payload"
        )
    universe_sha = _sha256(
        _first(
            pair_store,
            "source_candidate_universe_sha256",
            "candidate_universe_sha256",
            "scientific_identity.candidate_graph_hashes_sha256",
        ),
        label="pair-store candidate universe SHA-256",
    )
    generation_candidate_count = _integer(
        _first(
            generation,
            "candidate_count",
            "counterfactual_candidate_count",
        ),
        label="generation payload candidate count",
        minimum=1,
    )
    filtered_candidate_count = _integer(
        _first(
            pair_store,
            "candidate_count",
            "scientific_identity.candidate_count",
        ),
        label="strict-flip filtered candidate count",
        minimum=1,
    )
    if filtered_candidate_count > generation_candidate_count:
        raise MutFastAccurateV2Error(
            "strict-flip filtered universe exceeds the generation payload universe"
        )
    independently_derived_universe = _first(
        generation,
        "verified_strict_flip_candidate_universe_sha256",
        "strict_flip_candidate_universe_sha256",
    )
    if independently_derived_universe is not None and _sha256(
        independently_derived_universe,
        label="generation-derived strict-flip candidate universe SHA-256",
    ) != universe_sha:
        raise MutFastAccurateV2Error(
            "generation-derived strict-flip universe differs from the pair store"
        )
    pair_vectors = _sha256(
        _first(
            pair_store,
            "recourse_vectors_sha256",
            "artifacts.recourse_vectors_sha256",
        ),
        label="pair-store recourse vectors SHA-256",
    )
    dbscan_vectors = _sha256(
        _first(
            dbscan,
            "source_recourse_vectors_sha256",
            "vectors_sha256",
            "scientific_identity.vectors_sha256",
        ),
        label="DBSCAN source vectors SHA-256",
    )
    if pair_vectors != dbscan_vectors:
        raise MutFastAccurateV2Error("DBSCAN is not bound to the pair-store vectors")
    dbscan_native_universe = _first(
        dbscan,
        "source_candidate_universe_sha256",
        "candidate_universe_sha256",
        "scientific_identity.source_candidate_universe_sha256",
    )
    if dbscan_native_universe is not None:
        native = _sha256(
            dbscan_native_universe,
            label="DBSCAN native candidate universe SHA-256",
        )
        if native != universe_sha:
            raise MutFastAccurateV2Error(
                "DBSCAN native candidate universe differs from the pair store"
            )
    else:
        native = None

    generation_manifest_sha = _first(
        generation, "manifest_sha256", "generation_manifest_sha256"
    )
    pair_generation_manifest_sha = _first(
        pair_store,
        "source_generation_manifest_sha256",
        "scientific_identity.generation_manifest_sha256",
    )
    if generation_manifest_sha is not None or pair_generation_manifest_sha is not None:
        generation_manifest_sha = _sha256(
            generation_manifest_sha, label="generation manifest SHA-256"
        )
        pair_generation_manifest_sha = _sha256(
            pair_generation_manifest_sha,
            label="pair-store source generation manifest SHA-256",
        )
        if generation_manifest_sha != pair_generation_manifest_sha:
            raise MutFastAccurateV2Error(
                "pair store is not bound to the exact generation manifest"
            )
    return {
        "schema_version": "mut_candidate_universe_transitive_binding_v2",
        "status": "PASS",
        "binding_mode": "EXACT_GENERATION_PAYLOAD_TO_PAIR_VECTORS_TO_DBSCAN",
        "generation_payload_sha256": generation_payload,
        "generation_payload_universe_sha256": generation_payload,
        "generation_payload_candidate_count": generation_candidate_count,
        "pair_store_source_generation_payload_sha256": pair_source_payload,
        "generation_manifest_sha256": generation_manifest_sha,
        "pair_store_source_generation_manifest_sha256": pair_generation_manifest_sha,
        "pair_store_verified_candidate_universe_sha256": universe_sha,
        "strict_flip_filtered_candidate_universe_sha256": universe_sha,
        "strict_flip_filtered_candidate_count": filtered_candidate_count,
        "dbscan_native_candidate_universe_sha256": native,
        "dbscan_native_candidate_universe_field_present": native is not None,
        "pair_store_recourse_vectors_sha256": pair_vectors,
        "dbscan_source_recourse_vectors_sha256": dbscan_vectors,
        "candidate_universe_binding_pass": True,
        "candidate_universe_binding_transitive": True,
        "same_universe_requirement_closed_by_exact_transitive_chain": True,
        "generation_payload_and_filtered_universe_are_distinct_tiers": True,
        "claims_legacy_native_three_way_universe_fields": False,
    }


def validate_common_adoption_receipt(
    receipt: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate read-only adoption of pair, DBSCAN, and common recourse."""

    row = _mapping(receipt, label="common adoption receipt")
    binding = _mapping(expected_binding, label="expected candidate binding")
    if binding.get("status") != "PASS" or binding.get(
        "candidate_universe_binding_pass"
    ) is not True:
        raise MutFastAccurateV2Error("expected candidate binding is not PASS")
    required_true = (
        "pair_store_adopted_read_only",
        "dbscan_adopted_read_only",
        "common_recourse_adopted_read_only",
        "common_recourse_complete",
        "no_active_writer",
    )
    required_false = (
        "pair_store_rerun",
        "dbscan_rerun",
        "common_recourse_rerun",
        "calibration_loaded",
        "test_loaded",
    )
    if row.get("status") != "PASS":
        raise MutFastAccurateV2Error("common adoption receipt is not PASS")
    for field in required_true:
        _required_true(row, field)
    for field in required_false:
        _required_false(row, field)
    if row.get("writable_fd_count") != 0:
        raise MutFastAccurateV2Error("common adoption source still has a writer")
    expected_values = {
        "generation_payload_sha256": binding.get("generation_payload_sha256"),
        "candidate_universe_sha256": binding.get(
            "pair_store_verified_candidate_universe_sha256"
        ),
        "pair_store_recourse_vectors_sha256": binding.get(
            "pair_store_recourse_vectors_sha256"
        ),
        "dbscan_source_recourse_vectors_sha256": binding.get(
            "dbscan_source_recourse_vectors_sha256"
        ),
    }
    mismatches: list[str] = []
    for field, expected in expected_values.items():
        observed = _sha256(row.get(field), label=field)
        if observed != expected:
            mismatches.append(field)
    if mismatches:
        raise MutFastAccurateV2Error(
            "common adoption binding mismatch: " + ",".join(mismatches)
        )
    selected_count = _integer(
        row.get("selected_common_recourse_count"),
        label="selected_common_recourse_count",
        minimum=1,
    )
    return {
        "schema_version": "mut_common_recourse_adoption_validation_v2",
        "status": "PASS",
        **expected_values,
        "selected_common_recourse_count": selected_count,
        "pair_store_rerun": False,
        "dbscan_rerun": False,
        "common_recourse_rerun": False,
        "no_active_writer": True,
        "no_test_leakage": True,
    }


@dataclass(frozen=True, slots=True)
class MutConvergencePolicy:
    m_max: int = 50_000
    m_min: int = 20_000
    check_interval: int = 2_500
    patience_checks: int = 2
    top100_candidate_jaccard_min: float = 0.99
    top20_provisional_rule_jaccard_min: float = 0.95
    candidate_frequency_rank_spearman_min: float = 0.99
    absolute_train_coverage_gain_max: float = 0.005
    minimum_valid_unique_candidate_count: int = 10


MUT_CONVERGENCE_POLICY = MutConvergencePolicy()


def evaluate_train_side_convergence(
    windows: Sequence[Mapping[str, Any]],
    *,
    policy: MutConvergencePolicy = MUT_CONVERGENCE_POLICY,
) -> dict[str, Any]:
    """Evaluate exactly two consecutive committed train-only windows."""

    if policy.patience_checks != 2:
        raise MutFastAccurateV2Error("Mut convergence patience must remain two")
    if len(windows) < policy.patience_checks:
        return {
            "schema_version": "mut_train_side_convergence_v2",
            "status": "CONTINUE",
            "reason": "INSUFFICIENT_COMMITTED_WINDOWS",
            "policy": asdict(policy),
            "early_stop_used": False,
            "m_effective": None,
            "test_loaded": False,
            "calibration_loaded": False,
        }
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(windows):
        row = _mapping(raw, label=f"convergence window {index}")
        _required_true(row, "checkpoint_committed")
        _required_false(row, "calibration_loaded")
        _required_false(row, "test_loaded")
        if row.get("evidence_split") != "train":
            raise MutFastAccurateV2Error("convergence evidence must be train-only")
        step = _integer(row.get("step"), label="convergence step", minimum=1)
        if step > policy.m_max or step % policy.check_interval != 0:
            raise MutFastAccurateV2Error("convergence step is off policy boundary")
        checks = {
            "top100_candidate_jaccard": _number(
                row.get("top100_candidate_jaccard"),
                label="top100_candidate_jaccard",
            )
            >= policy.top100_candidate_jaccard_min,
            "top20_provisional_rule_jaccard": _number(
                row.get("top20_provisional_rule_jaccard"),
                label="top20_provisional_rule_jaccard",
            )
            >= policy.top20_provisional_rule_jaccard_min,
            "candidate_frequency_rank_spearman": _number(
                row.get("candidate_frequency_rank_spearman"),
                label="candidate_frequency_rank_spearman",
            )
            >= policy.candidate_frequency_rank_spearman_min,
            "absolute_train_coverage_gain": _number(
                row.get("absolute_train_coverage_gain"),
                label="absolute_train_coverage_gain",
                minimum=0.0,
            )
            <= policy.absolute_train_coverage_gain_max,
            "lineage_error_count": row.get("lineage_error_count") == 0,
            "valid_unique_candidate_count": (
                _integer(
                    row.get("valid_unique_candidate_count"),
                    label="valid_unique_candidate_count",
                )
                >= policy.minimum_valid_unique_candidate_count
            ),
            "checkpoint_reload_pass": row.get("checkpoint_reload_pass") is True,
        }
        normalized.append(
            {
                "step": step,
                "checks": checks,
                "pass": all(checks.values()) and step >= policy.m_min,
            }
        )
    steps = [row["step"] for row in normalized]
    if steps != sorted(set(steps)):
        raise MutFastAccurateV2Error(
            "convergence windows must have strictly increasing unique steps"
        )
    selected = normalized[-policy.patience_checks :]
    consecutive = (
        selected[1]["step"] - selected[0]["step"] == policy.check_interval
    )
    converged = consecutive and all(row["pass"] for row in selected)
    return {
        "schema_version": "mut_train_side_convergence_v2",
        "status": "CONVERGED_EARLY_STOP" if converged else "CONTINUE",
        "reason": (
            "TRAIN_SIDE_CONVERGENCE"
            if converged
            else (
                "NON_CONSECUTIVE_WINDOWS"
                if not consecutive
                else "THRESHOLDS_NOT_MET_TWICE"
            )
        ),
        "policy": asdict(policy),
        "windows": normalized,
        "selected_window_steps": [row["step"] for row in selected],
        "early_stop_used": converged,
        "m_effective": selected[-1]["step"] if converged else None,
        "stop_reason": "TRAIN_SIDE_CONVERGENCE" if converged else None,
        "test_loaded": False,
        "calibration_loaded": False,
    }


def build_capacity_report(
    *,
    candidate_capacity: int,
    max_resident_candidate_count: int,
    capacity_eviction_count: int,
    candidate_count_at_stop: int,
    eviction_policy: str | None = None,
) -> dict[str, Any]:
    """Report capacity as a maximum, never as a completion target."""

    capacity = _integer(candidate_capacity, label="candidate_capacity", minimum=1)
    if capacity != MUT_CANDIDATE_CAPACITY:
        raise MutFastAccurateV2Error("Mut candidate capacity must remain 100000")
    resident = _integer(
        max_resident_candidate_count,
        label="max_resident_candidate_count",
    )
    evictions = _integer(
        capacity_eviction_count,
        label="capacity_eviction_count",
    )
    at_stop = _integer(candidate_count_at_stop, label="candidate_count_at_stop")
    if resident > capacity:
        raise MutFastAccurateV2Error("resident candidate count exceeded capacity")
    reached = resident == capacity
    if evictions > 0:
        if not reached:
            raise MutFastAccurateV2Error(
                "capacity evictions are impossible before capacity is reached"
            )
        if eviction_policy not in {
            "frequency_based_eviction",
            "frozen_protocol_frequency_based_eviction",
        }:
            raise MutFastAccurateV2Error(
                "capacity eviction policy is absent or not frozen"
            )
    elif eviction_policy not in {None, "none"}:
        raise MutFastAccurateV2Error(
            "an eviction policy was claimed although no eviction occurred"
        )
    return {
        "schema_version": "mut_candidate_capacity_report_v2",
        "candidate_capacity": capacity,
        "max_resident_candidate_count": resident,
        "capacity_reached": reached,
        "capacity_eviction_count": evictions,
        "candidate_count_at_stop": at_stop,
        "capacity_constraint_inactive": not reached and evictions == 0,
        "eviction_policy": eviction_policy,
        "capacity_is_maximum_not_completion_target": True,
    }


def _json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MutFastAccurateV2Error(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise MutFastAccurateV2Error(f"JSON input must be an object: {path}")
    return value


def _emit(value: Mapping[str, Any]) -> int:
    print(json.dumps(dict(value), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Read-only JSON CLI for diagnostics and focused controller wiring."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cgroup = commands.add_parser("cgroup-snapshot")
    cgroup.add_argument("--root", required=True)
    cgroup.add_argument("--version", choices=("auto", "v1", "v2"), default="auto")
    supersede = commands.add_parser("supersede-440")
    supersede.add_argument("--old-required-free-gib", type=int, default=440)
    supersede.add_argument(
        "--authority", default=EMPIRICAL_ADMISSION_AUTHORITY
    )
    memory = commands.add_parser("derive-memory")
    memory.add_argument("--cgroup-peak-bytes", type=int, required=True)
    memory.add_argument("--process-peak-rss-bytes", type=int, required=True)
    memory.add_argument("--checkpoint-peak-bytes", type=int, required=True)
    capacity = commands.add_parser("capacity-report")
    capacity.add_argument("--candidate-capacity", type=int, default=100_000)
    capacity.add_argument("--max-resident-candidate-count", type=int, required=True)
    capacity.add_argument("--capacity-eviction-count", type=int, required=True)
    capacity.add_argument("--candidate-count-at-stop", type=int, required=True)
    capacity.add_argument("--eviction-policy")
    convergence = commands.add_parser("convergence")
    convergence.add_argument("--windows", required=True)
    args = parser.parse_args(argv)
    if args.command == "cgroup-snapshot":
        return _emit(read_cgroup_snapshot(args.root, version=args.version))
    if args.command == "supersede-440":
        return _emit(
            supersede_static_440g_gate(
                authority=args.authority,
                old_required_free_gib=args.old_required_free_gib,
            )
        )
    if args.command == "derive-memory":
        return _emit(
            derive_empirical_memory_admission(
                cgroup_memory_peak_bytes=args.cgroup_peak_bytes,
                process_peak_rss_bytes=args.process_peak_rss_bytes,
                checkpoint_peak_bytes=args.checkpoint_peak_bytes,
            )
        )
    if args.command == "capacity-report":
        return _emit(
            build_capacity_report(
                candidate_capacity=args.candidate_capacity,
                max_resident_candidate_count=args.max_resident_candidate_count,
                capacity_eviction_count=args.capacity_eviction_count,
                candidate_count_at_stop=args.candidate_count_at_stop,
                eviction_policy=args.eviction_policy,
            )
        )
    if args.command == "convergence":
        payload = json.loads(Path(args.windows).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise MutFastAccurateV2Error("convergence JSON must contain one list")
        return _emit(evaluate_train_side_convergence(payload))
    raise AssertionError(args.command)


__all__ = [
    "ADOPTION_WITHOUT_FULL_PARITY_AUTHORITY",
    "EMPIRICAL_ADMISSION_AUTHORITY",
    "GIB",
    "MUT_CANDIDATE_CAPACITY",
    "MUT_CONVERGENCE_POLICY",
    "MUT_STEPS",
    "MutConvergencePolicy",
    "MutFastAccurateV2Error",
    "build_capacity_report",
    "build_static_440_supersession",
    "derive_empirical_memory_admission",
    "evaluate_train_side_convergence",
    "main",
    "parse_cgroup_snapshot",
    "read_cgroup_snapshot",
    "supersede_static_440g_gate",
    "validate_500_step_semantic_equivalence",
    "validate_common_adoption_receipt",
    "verify_candidate_universe_binding",
    "verify_historical_50k_artifact",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
