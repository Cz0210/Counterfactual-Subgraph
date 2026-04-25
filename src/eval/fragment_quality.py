"""Shared fragment-audit helpers for SFT data and inference analysis."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.chem import (
    delete_fragment_from_parent,
    is_connected_fragment,
    is_parent_substructure,
    is_valid_capped_subgraph,
    parse_smiles,
)
from src.rewards.reward_wrapper import normalize_fragment_with_dummy_atoms
from src.utils.io import read_jsonl


LEGACY_SFT_PROMPT_TEMPLATE = (
    "[System]\n"
    "Generate a valid, chemically capped subgraph for the following parent molecule. "
    "Output only the fragment SMILES.\n\n"
    "[Input]\n"
    "PARENT_SMILES: {parent_smiles}\n\n"
    "[Output]\n"
)


_PARENT_PATTERNS = (
    re.compile(
        r"PARENT_SMILES:\s*(?P<smiles>.+?)(?:\n\s*\n\[Output\]|\n\s*\[Output\]|\Z)",
        flags=re.DOTALL,
    ),
    re.compile(
        r"MOLECULE_SMILES:\s*(?P<smiles>.+?)(?:\nFRAGMENT_SMILES:|\Z)",
        flags=re.DOTALL,
    ),
    re.compile(
        r"parent_smiles\s*[:=]\s*(?P<smiles>[^\s]+)",
        flags=re.IGNORECASE,
    ),
)
_SAMPLE_BLOCK_PATTERN = re.compile(r"(?m)^\[Sample\s+(?P<index>\d+)\]\s*$")
_TXT_PROMPT_PATTERN = re.compile(
    r"Prompt:\s*(?P<prompt>.*?)(?=\n(?:Reference Fragment:|Raw Generation:|Generated Fragment:|Contains '\*':|RDKit Valid:|Error:)|\Z)",
    flags=re.DOTALL,
)
_TXT_SINGLE_LINE_FIELDS = {
    "parent_smiles": re.compile(r"(?m)^Parent SMILES:\s*(?P<value>.*)$"),
    "reference_fragment": re.compile(r"(?m)^Reference Fragment:\s*(?P<value>.*)$"),
    "raw_generation": re.compile(r"(?m)^Raw Generation:\s*(?P<value>.*)$"),
    "generated_fragment": re.compile(r"(?m)^Generated Fragment:\s*(?P<value>.*)$"),
    "error": re.compile(r"(?m)^Error:\s*(?P<value>.*)$"),
}
_REFERENCE_FIELDS = (
    "reference_fragment",
    "reference",
    "target_fragment",
    "weak_fragment",
    "weak_label_fragment",
    "response",
    "output",
    "fragment_smiles",
)
_GENERATED_FIELDS = (
    "generated_fragment",
    "prediction",
    "predicted_fragment",
    "generated_smiles",
    "fragment_prediction",
    "completion_smiles",
    "model_fragment",
)
_INSTRUCTION_FIELDS = ("instruction", "prompt", "input")
_RAW_GENERATION_FIELDS = ("raw_generation", "raw_text", "generated_text", "completion")
_ROLE_SUMMARY_TEMPLATE = OrderedDict(
    (
        ("0-0.05", 0),
        ("0.05-0.1", 0),
        ("0.1-0.2", 0),
        ("0.2-0.4", 0),
        ("0.4-0.6", 0),
        ("0.6-0.8", 0),
        ("0.8-1.0", 0),
        ("full-parent", 0),
    )
)


@dataclass(frozen=True, slots=True)
class FragmentSourceRecord:
    """One input row adapted from SFT data or inference outputs."""

    sample_id: str
    source_path: str
    source_kind: str
    parent_smiles: str | None
    record_index: int | None = None
    instruction: str | None = None
    prompt: str | None = None
    label: int | None = None
    reference_fragment: str | None = None
    generated_fragment: str | None = None
    raw_generation: str | None = None
    error: str | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AuditOptions:
    """Execution controls for large-scale fragment audits."""

    skip_deleteability_check: bool = False
    skip_substructure_check: bool = False
    fail_fast: bool = False
    slow_stage_threshold_sec: float = 1.0


@dataclass(frozen=True, slots=True)
class SlowAuditEvent:
    """One slow audit stage worth surfacing for manual inspection."""

    record_index: int
    sample_id: str
    role: str
    parent_smiles: str | None
    fragment_smiles: str | None
    elapsed_sec: float
    stage: str
    audit_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "sample_id": self.sample_id,
            "role": self.role,
            "parent_smiles": self.parent_smiles,
            "fragment_smiles": self.fragment_smiles,
            "elapsed_sec": _round_or_none(self.elapsed_sec),
            "stage": self.stage,
            "audit_status": self.audit_status,
        }


@dataclass(frozen=True, slots=True)
class FragmentInspection:
    """Normalized chemistry audit for one raw fragment string."""

    raw_fragment: str
    core_fragment: str | None
    raw_canonical_smiles: str | None
    parent_canonical_smiles: str | None
    parent_parse_ok: bool
    parent_chemically_valid: bool
    raw_parse_ok: bool
    chemically_valid: bool
    connected: bool
    substructure_ok: bool
    substructure_check_skipped: bool
    deletion_supported: bool
    deleteability_check_skipped: bool
    deletion_fallback_used: bool
    residual_smiles: str | None
    residual_nonempty: bool
    empty_residual: bool
    has_dummy_atoms: bool
    dummy_count: int
    parent_atom_count: int | None
    core_atom_count: int
    atom_ratio: float | None
    is_full_parent: bool
    audit_status: str
    error_type: str | None
    error_message: str | None
    failure_reasons: tuple[str, ...]

    def to_flat_dict(self, prefix: str) -> dict[str, Any]:
        """Return one CSV-friendly flat dictionary."""

        return {
            f"{prefix}_raw_fragment": self.raw_fragment,
            f"{prefix}_core_fragment": self.core_fragment,
            f"{prefix}_raw_canonical_smiles": self.raw_canonical_smiles,
            f"{prefix}_parent_canonical_smiles": self.parent_canonical_smiles,
            f"{prefix}_parent_parse_ok": self.parent_parse_ok,
            f"{prefix}_parent_chemically_valid": self.parent_chemically_valid,
            f"{prefix}_raw_parse_ok": self.raw_parse_ok,
            f"{prefix}_chemically_valid": self.chemically_valid,
            f"{prefix}_connected": self.connected,
            f"{prefix}_substructure_ok": self.substructure_ok,
            f"{prefix}_substructure_check_skipped": self.substructure_check_skipped,
            f"{prefix}_deletion_supported": self.deletion_supported,
            f"{prefix}_deleteability_check_skipped": self.deleteability_check_skipped,
            f"{prefix}_deletion_fallback_used": self.deletion_fallback_used,
            f"{prefix}_residual_smiles": self.residual_smiles,
            f"{prefix}_residual_nonempty": self.residual_nonempty,
            f"{prefix}_empty_residual": self.empty_residual,
            f"{prefix}_has_dummy_atoms": self.has_dummy_atoms,
            f"{prefix}_dummy_count": self.dummy_count,
            f"{prefix}_parent_atom_count": self.parent_atom_count,
            f"{prefix}_core_atom_count": self.core_atom_count,
            f"{prefix}_atom_ratio": _round_or_none(self.atom_ratio),
            f"{prefix}_is_full_parent": self.is_full_parent,
            f"{prefix}_audit_status": self.audit_status,
            f"{prefix}_error_type": self.error_type,
            f"{prefix}_error_message": self.error_message,
            f"{prefix}_failure_reasons": " | ".join(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class AuditedFragmentRecord:
    """One adapted sample plus audited reference/generated fragments."""

    record: FragmentSourceRecord
    reference: FragmentInspection | None = None
    generated: FragmentInspection | None = None

    def to_detail_row(self) -> dict[str, Any]:
        """Return one CSV/detail row."""

        row: dict[str, Any] = {
            "record_index": self.record.record_index,
            "sample_id": self.record.sample_id,
            "source_path": self.record.source_path,
            "source_kind": self.record.source_kind,
            "label": self.record.label,
            "parent_smiles": self.record.parent_smiles,
            "instruction": self.record.instruction,
            "prompt": self.record.prompt,
            "raw_generation": self.record.raw_generation,
            "error": self.record.error,
        }
        if self.reference is not None:
            row.update(self.reference.to_flat_dict("reference"))
        else:
            row.update(_empty_detail_columns("reference"))
        if self.generated is not None:
            row.update(self.generated.to_flat_dict("generated"))
        else:
            row.update(_empty_detail_columns("generated"))
        return row


def load_fragment_source_records(paths: Sequence[str | Path]) -> list[FragmentSourceRecord]:
    """Load and adapt JSONL/TXT inputs into a common fragment record shape."""

    records: list[FragmentSourceRecord] = []
    for path_like in paths:
        path = Path(path_like).expanduser().resolve()
        if path.suffix.lower() == ".txt":
            records.extend(_load_records_from_txt(path))
        elif path.suffix.lower() == ".jsonl":
            records.extend(_load_records_from_jsonl(path))
        else:
            raise ValueError(
                f"Unsupported input format for {path}. Expected .jsonl or .txt."
            )
    return [replace(record, record_index=index) for index, record in enumerate(records)]


def audit_fragment_records(
    records: Sequence[FragmentSourceRecord],
    *,
    options: AuditOptions | None = None,
    progress_every: int = 0,
    slow_events: list[SlowAuditEvent] | None = None,
) -> list[AuditedFragmentRecord]:
    """Run chemistry audits for the reference/generated fragment columns."""

    audit_options = options or AuditOptions()
    audited: list[AuditedFragmentRecord] = []
    total_records = len(records)
    for list_index, record in enumerate(records):
        record_index = record.record_index if record.record_index is not None else list_index
        if record.record_index is None:
            record = replace(record, record_index=record_index)
        parent_smiles = str(record.parent_smiles or "").strip()
        reference = inspect_fragment(
            parent_smiles,
            record.reference_fragment,
            options=audit_options,
            record_index=record_index,
            sample_id=record.sample_id,
            role="reference",
            slow_events=slow_events,
        )
        generated = inspect_fragment(
            parent_smiles,
            record.generated_fragment,
            options=audit_options,
            record_index=record_index,
            sample_id=record.sample_id,
            role="generated",
            slow_events=slow_events,
        )
        audited.append(AuditedFragmentRecord(record=record, reference=reference, generated=generated))
        if progress_every > 0 and (list_index + 1) % progress_every == 0:
            print(
                f"[audit-progress] processed={list_index + 1}/{total_records} "
                f"last_record_index={record_index} sample_id={record.sample_id}"
            )
    return audited


def inspect_fragment(
    parent_smiles: str,
    fragment_smiles: str | None,
    *,
    options: AuditOptions | None = None,
    record_index: int = -1,
    sample_id: str = "",
    role: str = "fragment",
    slow_events: list[SlowAuditEvent] | None = None,
) -> FragmentInspection | None:
    """Normalize one fragment and compute structure/deletion statistics."""

    audit_options = options or AuditOptions()
    raw_fragment = str(fragment_smiles or "").strip()
    normalized_parent = str(parent_smiles or "").strip()
    if not raw_fragment:
        return None

    started = time.perf_counter()
    try:
        return _inspect_fragment_impl(
            normalized_parent,
            raw_fragment,
            options=audit_options,
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            slow_events=slow_events,
        )
    except Exception as exc:
        _maybe_record_slow_event(
            slow_events,
            threshold_sec=0.0,
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            parent_smiles=normalized_parent,
            fragment_smiles=raw_fragment,
            elapsed_sec=time.perf_counter() - started,
            stage="audit_exception",
            audit_status="error",
        )
        if audit_options.fail_fast:
            raise
        return _error_inspection(parent_smiles, raw_fragment, exc)


def _inspect_fragment_impl(
    parent_smiles: str,
    raw_fragment: str,
    *,
    options: AuditOptions,
    record_index: int,
    sample_id: str,
    role: str,
    slow_events: list[SlowAuditEvent] | None,
) -> FragmentInspection:
    failure_reasons: list[str] = []

    parent, parent_elapsed = _timed_call(
        lambda: parse_smiles(
            parent_smiles,
            sanitize=True,
            canonicalize=True,
            allow_capped_fragments=False,
        )
    )
    _maybe_record_slow_event(
        slow_events,
        threshold_sec=options.slow_stage_threshold_sec,
        record_index=record_index,
        sample_id=sample_id,
        role=role,
        parent_smiles=parent_smiles,
        fragment_smiles=raw_fragment,
        elapsed_sec=parent_elapsed,
        stage="parse_parent",
    )
    parent_parse_ok = bool(parent.parseable)
    parent_chemically_valid = bool(parent.sanitized)
    parent_atom_count = int(parent.atom_count or 0) if parent.sanitized else None
    parent_canonical = parent.canonical_smiles if parent.sanitized else None
    if not parent.parseable:
        failure_reasons.append(f"Parent parse failed: {parent.failure_reason}")
    elif not parent.sanitized:
        failure_reasons.append(f"Parent sanitization failed: {parent.failure_reason}")

    fragment_info, normalize_elapsed = _timed_call(
        lambda: normalize_fragment_with_dummy_atoms(raw_fragment)
    )
    _maybe_record_slow_event(
        slow_events,
        threshold_sec=options.slow_stage_threshold_sec,
        record_index=record_index,
        sample_id=sample_id,
        role=role,
        parent_smiles=parent_smiles,
        fragment_smiles=raw_fragment,
        elapsed_sec=normalize_elapsed,
        stage="normalize_fragment",
    )
    core_smiles = _clean_text(fragment_info.get("core_smiles"))
    raw_canonical = _clean_text(fragment_info.get("raw_canonical_smiles"))
    raw_parse_ok = bool(fragment_info.get("raw_parse_ok"))
    chemically_valid = bool(fragment_info.get("raw_sanitized"))
    has_dummy_atoms = bool(fragment_info.get("has_dummy"))
    core_atom_count = int(fragment_info.get("core_atom_count") or 0)
    dummy_count = max(int(fragment_info.get("dummy_count") or 0), raw_fragment.count("*"))
    if not raw_parse_ok:
        failure_reasons.append("Fragment parse failed after dummy-aware normalization.")
    elif not chemically_valid:
        failure_reasons.append("Fragment sanitization failed after dummy-aware normalization.")

    atom_ratio = None
    if parent_atom_count and core_atom_count >= 0:
        atom_ratio = core_atom_count / parent_atom_count
    is_full_parent = bool(core_smiles and parent_canonical and core_smiles == parent_canonical)

    connected = False
    if chemically_valid:
        connected, connected_elapsed = _timed_call(lambda: is_connected_fragment(raw_fragment))
        _maybe_record_slow_event(
            slow_events,
            threshold_sec=options.slow_stage_threshold_sec,
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            parent_smiles=parent_smiles,
            fragment_smiles=raw_fragment,
            elapsed_sec=connected_elapsed,
            stage="connected_check",
        )
        if not connected:
            failure_reasons.append("Fragment contains more than one connected component.")

    if (
        parent_atom_count is not None
        and core_atom_count > 0
        and core_atom_count > parent_atom_count
    ):
        failure_reasons.append("Fragment core atom count exceeds parent atom count.")

    substructure_ok = False
    substructure_check_skipped = bool(options.skip_substructure_check)
    if is_full_parent and parent_chemically_valid and chemically_valid:
        substructure_ok = True
        substructure_check_skipped = False
    elif (
        not substructure_check_skipped
        and parent_chemically_valid
        and chemically_valid
        and connected
        and core_atom_count > 0
        and not (
            parent_atom_count is not None
            and core_atom_count > parent_atom_count
        )
    ):
        if has_dummy_atoms:
            substructure_ok, substructure_elapsed = _timed_call(
                lambda: is_valid_capped_subgraph(parent_smiles, raw_fragment)
            )
        else:
            substructure_ok, substructure_elapsed = _timed_call(
                lambda: is_parent_substructure(parent_smiles, raw_fragment)
            )
        _maybe_record_slow_event(
            slow_events,
            threshold_sec=options.slow_stage_threshold_sec,
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            parent_smiles=parent_smiles,
            fragment_smiles=raw_fragment,
            elapsed_sec=substructure_elapsed,
            stage="substructure_check",
        )
        if not substructure_ok:
            failure_reasons.append(
                "Capped fragment does not correspond to a fully capped parent subgraph."
                if has_dummy_atoms
                else "Fragment is not a substructure of the parent molecule."
            )
    elif substructure_check_skipped:
        failure_reasons.append("Substructure check skipped by CLI option.")

    deleteability_check_skipped = bool(
        options.skip_deleteability_check or substructure_check_skipped
    )
    deletion_supported = False
    deletion_fallback_used = False
    residual_smiles: str | None = None
    if is_full_parent and parent_chemically_valid and chemically_valid and connected:
        deleteability_check_skipped = False
        deletion_supported = True
        residual_smiles = ""
    elif (
        not deleteability_check_skipped
        and parent_chemically_valid
        and chemically_valid
        and connected
        and substructure_ok
    ):
        deletion_result, deletion_elapsed = _timed_call(
            lambda: delete_fragment_from_parent(
                parent_smiles,
                raw_fragment,
                max_matches=1,
            )
        )
        _maybe_record_slow_event(
            slow_events,
            threshold_sec=options.slow_stage_threshold_sec,
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            parent_smiles=parent_smiles,
            fragment_smiles=raw_fragment,
            elapsed_sec=deletion_elapsed,
            stage="deleteability_check",
        )
        deletion_supported = bool(deletion_result.success)
        residual_smiles = deletion_result.residual_smiles
        if not deletion_result.success:
            failure_reasons.append(
                f"Deletion failed: {deletion_result.failure_reason or 'unknown deletion failure'}"
            )
        elif not residual_smiles:
            residual_smiles = ""

        if (
            not deletion_supported
            and core_smiles
            and core_smiles != raw_fragment
        ):
            fallback_result, fallback_elapsed = _timed_call(
                lambda: delete_fragment_from_parent(
                    parent_smiles,
                    core_smiles,
                    max_matches=1,
                )
            )
            _maybe_record_slow_event(
                slow_events,
                threshold_sec=options.slow_stage_threshold_sec,
                record_index=record_index,
                sample_id=sample_id,
                role=role,
                parent_smiles=parent_smiles,
                fragment_smiles=core_smiles,
                elapsed_sec=fallback_elapsed,
                stage="deleteability_check_core_fallback",
            )
            if fallback_result.success:
                deletion_supported = True
                deletion_fallback_used = True
                residual_smiles = fallback_result.residual_smiles
                failure_reasons = [
                    reason
                    for reason in failure_reasons
                    if not reason.startswith("Deletion failed:")
                ]
                failure_reasons.append("Deletion fallback used core fragment.")
    elif deleteability_check_skipped:
        if options.skip_deleteability_check:
            failure_reasons.append("Deleteability check skipped by CLI option.")
        elif substructure_check_skipped:
            failure_reasons.append("Deleteability check skipped because substructure check was skipped.")

    residual_nonempty = deletion_supported and residual_smiles not in (None, "")
    empty_residual = deletion_supported and residual_smiles == ""
    return FragmentInspection(
        raw_fragment=raw_fragment,
        core_fragment=core_smiles,
        raw_canonical_smiles=raw_canonical,
        parent_canonical_smiles=parent_canonical,
        parent_parse_ok=parent_parse_ok,
        parent_chemically_valid=parent_chemically_valid,
        raw_parse_ok=raw_parse_ok,
        chemically_valid=chemically_valid,
        connected=connected,
        substructure_ok=substructure_ok,
        substructure_check_skipped=substructure_check_skipped,
        deletion_supported=deletion_supported,
        deleteability_check_skipped=deleteability_check_skipped,
        deletion_fallback_used=deletion_fallback_used,
        residual_smiles=residual_smiles,
        residual_nonempty=residual_nonempty,
        empty_residual=empty_residual,
        has_dummy_atoms=has_dummy_atoms,
        dummy_count=dummy_count,
        parent_atom_count=parent_atom_count,
        core_atom_count=core_atom_count,
        atom_ratio=atom_ratio,
        is_full_parent=is_full_parent,
        audit_status="ok",
        error_type=None,
        error_message=None,
        failure_reasons=tuple(failure_reasons),
    )


def _error_inspection(
    parent_smiles: str,
    raw_fragment: str,
    exc: Exception,
) -> FragmentInspection:
    return FragmentInspection(
        raw_fragment=raw_fragment,
        core_fragment=None,
        raw_canonical_smiles=None,
        parent_canonical_smiles=None,
        parent_parse_ok=False,
        parent_chemically_valid=False,
        raw_parse_ok=False,
        chemically_valid=False,
        connected=False,
        substructure_ok=False,
        substructure_check_skipped=False,
        deletion_supported=False,
        deleteability_check_skipped=False,
        deletion_fallback_used=False,
        residual_smiles=None,
        residual_nonempty=False,
        empty_residual=False,
        has_dummy_atoms="*" in raw_fragment,
        dummy_count=raw_fragment.count("*"),
        parent_atom_count=None,
        core_atom_count=0,
        atom_ratio=None,
        is_full_parent=False,
        audit_status="error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        failure_reasons=(f"Audit exception: {type(exc).__name__}: {exc}",),
    )


def _timed_call(func):
    started = time.perf_counter()
    value = func()
    return value, time.perf_counter() - started


def _maybe_record_slow_event(
    slow_events: list[SlowAuditEvent] | None,
    *,
    threshold_sec: float,
    record_index: int,
    sample_id: str,
    role: str,
    parent_smiles: str | None,
    fragment_smiles: str | None,
    elapsed_sec: float,
    stage: str,
    audit_status: str = "ok",
) -> None:
    if slow_events is None or elapsed_sec < threshold_sec:
        return
    slow_events.append(
        SlowAuditEvent(
            record_index=record_index,
            sample_id=sample_id,
            role=role,
            parent_smiles=parent_smiles,
            fragment_smiles=fragment_smiles,
            elapsed_sec=elapsed_sec,
            stage=stage,
            audit_status=audit_status,
        )
    )


def summarize_role_metrics(
    audited_records: Sequence[AuditedFragmentRecord],
    *,
    role: str,
    near_parent_threshold: float = 0.8,
    tiny_fragment_threshold: float = 0.08,
    mid_size_min: float = 0.1,
    mid_size_max: float = 0.6,
) -> dict[str, Any]:
    """Aggregate one role's fragment-quality metrics over audited records."""

    inspections = [
        _inspection_for_role(record, role)
        for record in audited_records
        if _inspection_for_role(record, role) is not None
    ]
    available_count = len(inspections)
    total_count = len(audited_records)
    ratios = [inspection.atom_ratio for inspection in inspections if inspection.atom_ratio is not None]
    histogram = OrderedDict((label, 0) for label in _ROLE_SUMMARY_TEMPLATE)
    dummy_distribution = Counter()
    substructure_evaluable_count = sum(
        not inspection.substructure_check_skipped for inspection in inspections
    )
    deleteability_evaluable_count = sum(
        not inspection.deleteability_check_skipped for inspection in inspections
    )
    audit_error_count = sum(inspection.audit_status == "error" for inspection in inspections)

    for inspection in inspections:
        dummy_distribution[str(inspection.dummy_count)] += 1
        if inspection.atom_ratio is None:
            continue
        if inspection.is_full_parent:
            histogram["full-parent"] += 1
            continue
        ratio = inspection.atom_ratio
        if 0.0 <= ratio < 0.05:
            histogram["0-0.05"] += 1
        elif ratio < 0.1:
            histogram["0.05-0.1"] += 1
        elif ratio < 0.2:
            histogram["0.1-0.2"] += 1
        elif ratio < 0.4:
            histogram["0.2-0.4"] += 1
        elif ratio < 0.6:
            histogram["0.4-0.6"] += 1
        elif ratio < 0.8:
            histogram["0.6-0.8"] += 1
        else:
            histogram["0.8-1.0"] += 1

    return {
        "role": role,
        "total_records": total_count,
        "available_fragments": available_count,
        "missing_fragments": total_count - available_count,
        "audit_ok_count": available_count - audit_error_count,
        "audit_error_count": audit_error_count,
        "valid_rate": _safe_rate(sum(inspection.chemically_valid for inspection in inspections), available_count),
        "parse_fail_rate": _safe_rate(sum(not inspection.raw_parse_ok for inspection in inspections), available_count),
        "substructure_evaluable_fragments": substructure_evaluable_count,
        "substructure_skipped_fragments": available_count - substructure_evaluable_count,
        "substructure_rate": _safe_rate(
            sum(inspection.substructure_ok for inspection in inspections if not inspection.substructure_check_skipped),
            substructure_evaluable_count,
        ),
        "atom_ratio_mean": _round_or_none(_safe_mean(ratios)),
        "atom_ratio_median": _round_or_none(_median(ratios)),
        "atom_ratio_p25": _round_or_none(_percentile(ratios, 0.25)),
        "atom_ratio_p75": _round_or_none(_percentile(ratios, 0.75)),
        "near_parent_rate": _safe_rate(
            sum(
                inspection.atom_ratio is not None and inspection.atom_ratio >= near_parent_threshold
                for inspection in inspections
            ),
            available_count,
        ),
        "full_parent_rate": _safe_rate(sum(inspection.is_full_parent for inspection in inspections), available_count),
        "tiny_fragment_rate": _safe_rate(
            sum(
                inspection.atom_ratio is not None and inspection.atom_ratio <= tiny_fragment_threshold
                for inspection in inspections
            ),
            available_count,
        ),
        "mid_size_rate": _safe_rate(
            sum(
                inspection.atom_ratio is not None
                and mid_size_min <= inspection.atom_ratio <= mid_size_max
                for inspection in inspections
            ),
            available_count,
        ),
        "deleteability_evaluable_fragments": deleteability_evaluable_count,
        "deleteability_skipped_fragments": available_count - deleteability_evaluable_count,
        "deleteable_rate": _safe_rate(
            sum(
                inspection.residual_nonempty
                for inspection in inspections
                if not inspection.deleteability_check_skipped
            ),
            deleteability_evaluable_count,
        ),
        "empty_residual_rate": _safe_rate(
            sum(
                inspection.empty_residual
                for inspection in inspections
                if not inspection.deleteability_check_skipped
            ),
            deleteability_evaluable_count,
        ),
        "dummy_count_distribution": dict(sorted(dummy_distribution.items(), key=_counter_sort_key)),
        "atom_ratio_histogram": dict(histogram),
    }


def build_summary_payload(
    audited_records: Sequence[AuditedFragmentRecord],
    *,
    input_paths: Sequence[str | Path],
    near_parent_threshold: float = 0.8,
    tiny_fragment_threshold: float = 0.08,
    mid_size_min: float = 0.1,
    mid_size_max: float = 0.6,
) -> dict[str, Any]:
    """Build a JSON-friendly summary over both reference and generated roles."""

    return {
        "input_paths": [str(Path(path).expanduser().resolve()) for path in input_paths],
        "total_records": len(audited_records),
        "source_kinds": dict(Counter(record.record.source_kind for record in audited_records)),
        "reference": summarize_role_metrics(
            audited_records,
            role="reference",
            near_parent_threshold=near_parent_threshold,
            tiny_fragment_threshold=tiny_fragment_threshold,
            mid_size_min=mid_size_min,
            mid_size_max=mid_size_max,
        ),
        "generated": summarize_role_metrics(
            audited_records,
            role="generated",
            near_parent_threshold=near_parent_threshold,
            tiny_fragment_threshold=tiny_fragment_threshold,
            mid_size_min=mid_size_min,
            mid_size_max=mid_size_max,
        ),
    }


def select_top_k_by_atom_ratio(
    audited_records: Sequence[AuditedFragmentRecord],
    *,
    role: str,
    k: int,
    largest: bool,
) -> list[dict[str, Any]]:
    """Return top-k largest/smallest atom-ratio samples for manual inspection."""

    candidates: list[tuple[float, AuditedFragmentRecord, FragmentInspection]] = []
    for audited in audited_records:
        inspection = _inspection_for_role(audited, role)
        if inspection is None or inspection.atom_ratio is None:
            continue
        candidates.append((inspection.atom_ratio, audited, inspection))

    candidates.sort(key=lambda item: item[0], reverse=largest)
    selected = candidates[: max(k, 0)]
    output: list[dict[str, Any]] = []
    for ratio, audited, inspection in selected:
        output.append(
            {
                "record_index": audited.record.record_index,
                "sample_id": audited.record.sample_id,
                "source_path": audited.record.source_path,
                "source_kind": audited.record.source_kind,
                "parent_smiles": audited.record.parent_smiles,
                "reference_fragment": audited.record.reference_fragment,
                "generated_fragment": audited.record.generated_fragment,
                "raw_generation": audited.record.raw_generation,
                "core_fragment": inspection.core_fragment,
                "atom_ratio": _round_or_none(ratio),
                "dummy_count": inspection.dummy_count,
                "is_full_parent": inspection.is_full_parent,
                "residual_nonempty": inspection.residual_nonempty,
                "audit_status": inspection.audit_status,
                "error_type": inspection.error_type,
                "error": audited.record.error,
            }
        )
    return output


def build_detail_rows(audited_records: Sequence[AuditedFragmentRecord]) -> list[dict[str, Any]]:
    """Return CSV/detail rows for every audited sample."""

    return [record.to_detail_row() for record in audited_records]


def write_detail_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one per-sample CSV file."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else _default_detail_fieldnames()
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_rows(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write JSONL rows for slow-event or debug artifacts."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def format_role_summary_lines(summary: dict[str, Any]) -> list[str]:
    """Render one role summary for console/text output."""

    if summary.get("available_fragments", 0) == 0:
        return [
            f"{summary.get('role', 'unknown')}: no usable fragments",
        ]

    return [
        (
            f"{summary['role']}: available={summary['available_fragments']} "
            f"valid={_format_rate(summary['valid_rate'])} "
            f"substructure={_format_rate(summary['substructure_rate'])} "
            f"deleteable={_format_rate(summary['deleteable_rate'])}"
        ),
        (
            f"  audit_errors={summary.get('audit_error_count', 0)} "
            f"substructure_skipped={summary.get('substructure_skipped_fragments', 0)} "
            f"deleteability_skipped={summary.get('deleteability_skipped_fragments', 0)}"
        ),
        (
            f"  atom_ratio mean/median/p25/p75="
            f"{_format_number(summary['atom_ratio_mean'])}/"
            f"{_format_number(summary['atom_ratio_median'])}/"
            f"{_format_number(summary['atom_ratio_p25'])}/"
            f"{_format_number(summary['atom_ratio_p75'])}"
        ),
        (
            f"  near_parent={_format_rate(summary['near_parent_rate'])} "
            f"full_parent={_format_rate(summary['full_parent_rate'])} "
            f"tiny={_format_rate(summary['tiny_fragment_rate'])} "
            f"mid_size={_format_rate(summary['mid_size_rate'])} "
            f"empty_residual={_format_rate(summary['empty_residual_rate'])}"
        ),
        f"  histogram={summary['atom_ratio_histogram']}",
        f"  dummy_count_distribution={summary['dummy_count_distribution']}",
    ]


def build_training_row_from_record(
    record: FragmentSourceRecord,
    *,
    output_fragment: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Convert one adapted record back into a train_sft.py-compatible JSON row."""

    payload = dict(record.raw_payload or {})
    instruction = (
        _clean_text(payload.get("instruction"))
        or _clean_text(payload.get("prompt"))
        or _clean_text(record.instruction)
        or _clean_text(record.prompt)
    )
    if not instruction:
        parent_smiles = _clean_text(record.parent_smiles)
        if not parent_smiles:
            raise ValueError("Cannot build an SFT row without instruction/prompt or parent_smiles.")
        instruction = LEGACY_SFT_PROMPT_TEMPLATE.format(parent_smiles=parent_smiles)

    payload["instruction"] = instruction
    payload["output"] = str(output_fragment).strip()
    if record.parent_smiles:
        payload["parent_smiles"] = record.parent_smiles
    if record.label is not None:
        payload["label"] = record.label
    existing_meta = payload.get("meta")
    merged_meta: dict[str, Any] = {}
    if isinstance(existing_meta, dict):
        merged_meta.update(existing_meta)
    merged_meta.update(meta)
    payload["meta"] = merged_meta
    return payload


def bucket_name_for_atom_ratio(atom_ratio: float | None) -> str | None:
    """Map one atom ratio to the small/medium/large rebalance buckets."""

    if atom_ratio is None:
        return None
    if 0.08 <= atom_ratio < 0.2:
        return "small"
    if 0.2 <= atom_ratio < 0.4:
        return "medium"
    if 0.4 <= atom_ratio <= 0.6:
        return "large"
    return None


def _load_records_from_jsonl(path: Path) -> list[FragmentSourceRecord]:
    rows = read_jsonl(path)
    source_kind = _detect_jsonl_source_kind(rows)
    records: list[FragmentSourceRecord] = []
    for index, row in enumerate(rows):
        parent_smiles = _extract_parent_smiles_from_row(row)
        label = _coerce_label(
            row.get("label", row.get("HIV_active", _extract_from_meta(row, "label")))
        )
        instruction = _first_text(row, _INSTRUCTION_FIELDS)
        prompt = _clean_text(row.get("prompt")) or instruction

        reference_fragment = _extract_reference_fragment(row, source_kind=source_kind)
        generated_fragment = _extract_generated_fragment(row, source_kind=source_kind)
        sample_id = _coerce_sample_id(row, index=index, source_path=path)
        records.append(
            FragmentSourceRecord(
                sample_id=sample_id,
                source_path=str(path),
                source_kind=source_kind,
                parent_smiles=parent_smiles,
                instruction=instruction,
                prompt=prompt,
                label=label,
                reference_fragment=reference_fragment,
                generated_fragment=generated_fragment,
                raw_generation=_first_text(row, _RAW_GENERATION_FIELDS),
                error=_clean_text(row.get("error")),
                raw_payload=row,
            )
        )
    return records


def _load_records_from_txt(path: Path) -> list[FragmentSourceRecord]:
    text = path.read_text(encoding="utf-8")
    matches = list(_SAMPLE_BLOCK_PATTERN.finditer(text))
    if not matches:
        raise ValueError(
            f"Could not find '[Sample N]' blocks in {path}. "
            "Expected the current scripts/run_infer_sft.py log format."
        )

    records: list[FragmentSourceRecord] = []
    for block_index, match in enumerate(matches):
        start = match.end()
        end = matches[block_index + 1].start() if block_index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        prompt_match = _TXT_PROMPT_PATTERN.search(block)
        prompt = _clean_text(prompt_match.group("prompt")) if prompt_match else None
        parent_smiles = _extract_text_field(block, "parent_smiles") or _extract_parent_smiles_from_text(prompt)
        sample_id = match.group("index")
        records.append(
            FragmentSourceRecord(
                sample_id=sample_id,
                source_path=str(path),
                source_kind="sft_inference_txt",
                parent_smiles=parent_smiles,
                prompt=prompt,
                reference_fragment=_extract_text_field(block, "reference_fragment"),
                generated_fragment=_extract_text_field(block, "generated_fragment"),
                raw_generation=_extract_text_field(block, "raw_generation"),
                error=_extract_text_field(block, "error"),
            )
        )
    return records


def _detect_jsonl_source_kind(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "empty_jsonl"
    if any(
        any(field in row for field in _GENERATED_FIELDS) or "raw_generation" in row
        for row in rows
    ):
        return "sft_inference_jsonl"
    if any("instruction" in row and "output" in row for row in rows):
        return "sft_dataset_jsonl"
    return "generic_jsonl"


def _extract_reference_fragment(row: dict[str, Any], *, source_kind: str) -> str | None:
    direct = _first_text(row, _REFERENCE_FIELDS)
    if direct:
        return direct
    if source_kind == "generic_jsonl":
        return _clean_text(_extract_from_meta(row, "reference_fragment"))
    return None


def _extract_generated_fragment(row: dict[str, Any], *, source_kind: str) -> str | None:
    direct = _first_text(row, _GENERATED_FIELDS)
    if direct:
        return direct
    if source_kind == "sft_inference_jsonl":
        fallback = _clean_text(row.get("output"))
        if fallback:
            return fallback
    return _clean_text(_extract_from_meta(row, "generated_fragment"))


def _extract_parent_smiles_from_row(row: dict[str, Any]) -> str | None:
    direct_candidates = (
        row.get("parent_smiles"),
        row.get("smiles"),
        row.get("molecule_smiles"),
        row.get("parent"),
        row.get("input_smiles"),
        _extract_from_meta(row, "parent_smiles"),
        _extract_from_meta(row, "smiles"),
    )
    for candidate in direct_candidates:
        text = _clean_text(candidate)
        if text:
            return text
    for field in _INSTRUCTION_FIELDS + ("text",):
        text = _clean_text(row.get(field))
        extracted = _extract_parent_smiles_from_text(text)
        if extracted:
            return extracted
    return None


def _extract_parent_smiles_from_text(text: str | None) -> str | None:
    normalized = _clean_text(text)
    if not normalized:
        return None
    for pattern in _PARENT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return _clean_text(match.group("smiles"))
    return None


def _extract_text_field(block: str, field_name: str) -> str | None:
    pattern = _TXT_SINGLE_LINE_FIELDS[field_name]
    match = pattern.search(block)
    if not match:
        return None
    return _clean_text(match.group("value"))


def _coerce_sample_id(row: dict[str, Any], *, index: int, source_path: Path) -> str:
    for key in ("id", "record_id", "sample_id", "index"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return f"{source_path.stem}:{index}"


def _extract_from_meta(row: dict[str, Any], key: str) -> Any:
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _first_text(row: dict[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in row:
            text = _clean_text(row.get(key))
            if text:
                return text
    for key in keys:
        text = _clean_text(_extract_from_meta(row, key))
        if text:
            return text
    return None


def _inspection_for_role(
    record: AuditedFragmentRecord,
    role: str,
) -> FragmentInspection | None:
    if role == "reference":
        return record.reference
    if role == "generated":
        return record.generated
    raise ValueError(f"Unsupported role: {role}")


def _empty_detail_columns(prefix: str) -> dict[str, Any]:
    placeholder = FragmentInspection(
        raw_fragment="",
        core_fragment=None,
        raw_canonical_smiles=None,
        parent_canonical_smiles=None,
        parent_parse_ok=False,
        parent_chemically_valid=False,
        raw_parse_ok=False,
        chemically_valid=False,
        connected=False,
        substructure_ok=False,
        substructure_check_skipped=False,
        deletion_supported=False,
        deleteability_check_skipped=False,
        deletion_fallback_used=False,
        residual_smiles=None,
        residual_nonempty=False,
        empty_residual=False,
        has_dummy_atoms=False,
        dummy_count=0,
        parent_atom_count=None,
        core_atom_count=0,
        atom_ratio=None,
        is_full_parent=False,
        audit_status="missing",
        error_type=None,
        error_message=None,
        failure_reasons=(),
    )
    return placeholder.to_flat_dict(prefix)


def _default_detail_fieldnames() -> list[str]:
    return list(
        AuditedFragmentRecord(
            record=FragmentSourceRecord(
                sample_id="",
                source_path="",
                source_kind="",
                parent_smiles=None,
            )
        ).to_detail_row().keys()
    )


def _coerce_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _safe_mean(values: Sequence[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _median(values: Sequence[float | None]) -> float | None:
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return None
    size = len(usable)
    middle = size // 2
    if size % 2 == 1:
        return usable[middle]
    return (usable[middle - 1] + usable[middle]) / 2.0


def _percentile(values: Sequence[float | None], quantile: float) -> float | None:
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]
    position = (len(usable) - 1) * min(max(quantile, 0.0), 1.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return usable[lower_index]
    lower_value = usable[lower_index]
    upper_value = usable[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _round_or_none(value: float | None, *, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _counter_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    key, _ = item
    try:
        return int(key), key
    except ValueError:
        return 10**9, key


def _format_rate(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100.0:.2f}%"


def _format_number(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"
