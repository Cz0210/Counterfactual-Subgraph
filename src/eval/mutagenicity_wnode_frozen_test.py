"""One-shot frozen-selector WNode evaluation on the Mutagenicity test cohort."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from src.eval.close_counterfactual_coverage import (
    hard_delete_substructure_any_match,
    predict_with_teacher,
)
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.mutagenicity_wnode_matrix import (
    CalibrationParent,
    evaluate_parent_candidate_pair,
    run_wnode_self_test,
)
from src.eval.mutagenicity_wnode_selector import (
    MatrixData,
    ThresholdBundle,
    ThresholdLevel,
    build_candidate_chemistry,
    build_coverage_redundancy_matrix,
    fixed_denominator_capped_cost,
    single_threshold_coverage,
    weighted_multi_threshold_utility,
)


SOURCE_LABEL = 1
TARGET_LABEL = 0
EXPECTED_SELECTED_VARIANT = "A2_MultiThreshold"
EXPECTED_TEST_PARENT_COUNT = 217
EXPECTED_CANDIDATE_COUNT = 20
EXPECTED_PAIR_COUNT = 4340
EXPECTED_TOP_K = 20
EXPECTED_TABLE_K = 10
METHOD_NAME = "Ours-ChemLLM-PPO-WNode-A2"
DISTANCE_TYPE = "node_wasserstein"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
FLOAT_TOLERANCE = 1e-12

REQUIRED_FROZEN_FILES = (
    "_FROZEN.json",
    "thresholds.json",
    "calibration_decision.json",
    "selected_variant/selected_sequence.jsonl",
    "selected_variant/selected_top10.json",
    "selected_variant/selected_top20.json",
)

REQUIRED_OUTPUT_FILES = (
    "pair_matrix.jsonl",
    "match_instances.jsonl",
    "selected_sequence.jsonl",
    "thresholds.json",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_ours_k10.csv",
    "table2_ours_k20.csv",
    "summary.json",
    "run_manifest.json",
    "resume_checkpoint.json",
    "_RUN_COMPLETE.json",
)


class TeacherProtocol(Protocol):
    def score_smiles(
        self,
        smiles: str,
        label: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...


class DistanceProtocol(Protocol):
    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        ...

    def stats_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class FrozenSelectorPackage:
    root: Path
    manifest: dict[str, Any]
    frozen_marker: dict[str, Any]
    calibration_decision: dict[str, Any]
    threshold_payload: dict[str, Any]
    thresholds: ThresholdBundle
    selected_sequence: tuple[dict[str, Any], ...]
    selected_variant: str
    top_k: int
    table_k: int
    frozen_selector_hash: str
    verified_file_sha256: dict[str, str]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(str(row["candidate_id"]) for row in self.selected_sequence)


@dataclass(frozen=True, slots=True)
class FrozenTestConfig:
    id_col: str = "molecule_id"
    smiles_col: str = "smiles"
    label_col: str = "label"
    cohort_name: str = "test"
    expected_parent_count: int = EXPECTED_TEST_PARENT_COUNT
    wnode_size_penalty_beta: float = 0.0
    flush_every: int = 100
    resume: bool = True
    local_files_only: bool = True


class FrozenTestInterrupted(RuntimeError):
    """Private test hook used to exercise true partial-run resume."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, tuple, dict))
                            else ("" if value is None else value)
                        )
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(
    path: Path,
    *,
    allow_truncated_last_line: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty = nonempty[-1] if nonempty else -1
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if allow_truncated_last_line and index == final_nonempty:
                break
            raise
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{index + 1}")
        rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if path.is_file() and stat.st_size <= 64 * 1024 * 1024:
        payload["sha256"] = _sha256_file(path)
    return payload


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _pair_key(parent_id: str, candidate_id: str) -> str:
    return f"{parent_id}\t{candidate_id}"


def _extract_hash_entries(manifest: dict[str, Any]) -> dict[str, str]:
    entries: dict[str, str] = {}

    def add(path_value: Any, hash_value: Any) -> None:
        relative = str(path_value or "").strip()
        digest = str(hash_value or "").strip().lower()
        if relative and len(digest) == 64:
            entries[relative] = digest

    for field in ("file_sha256", "files", "artifacts"):
        payload = manifest.get(field)
        if isinstance(payload, dict):
            for path_value, value in payload.items():
                if isinstance(value, str):
                    add(path_value, value)
                elif isinstance(value, dict):
                    add(
                        value.get("path") or value.get("file") or path_value,
                        value.get("sha256")
                        or value.get("file_sha256")
                        or value.get("hash"),
                    )
        elif isinstance(payload, list):
            for value in payload:
                if not isinstance(value, dict):
                    continue
                add(
                    value.get("path") or value.get("file") or value.get("name"),
                    value.get("sha256")
                    or value.get("file_sha256")
                    or value.get("hash"),
                )
    if not entries:
        raise ValueError(
            "frozen_selector_manifest.json has no supported file_sha256 entries."
        )
    return entries


def _safe_frozen_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Frozen manifest path escapes package root: {relative}") from exc
    return path


def _candidate_ids_from_json(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in (
            "candidate_ids",
            "selected_candidate_ids",
            "candidates",
            "selected",
            "selected_sequence",
        ):
            if key in payload:
                return _candidate_ids_from_json(payload[key])
        if "candidate_id" in payload:
            return [str(payload["candidate_id"])]
        raise ValueError(
            "Selected-candidate JSON has no candidate list or candidate_id."
        )
    if not isinstance(payload, list):
        raise ValueError("Selected-candidate JSON must contain a list.")
    result: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            candidate_id = str(item.get("candidate_id") or "").strip()
        else:
            candidate_id = str(item or "").strip()
        if not candidate_id:
            raise ValueError("Selected-candidate JSON contains an empty candidate_id.")
        result.append(candidate_id)
    return result


def _load_threshold_bundle(payload: dict[str, Any]) -> ThresholdBundle:
    raw_rows = payload.get("raw_quantile_thresholds")
    merged_rows = payload.get("merged_thresholds")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Frozen thresholds lack raw_quantile_thresholds.")
    if not isinstance(merged_rows, list) or not merged_rows:
        raise ValueError("Frozen thresholds lack merged_thresholds.")
    requested_quantiles: list[float] = []
    requested_weights: list[float] = []
    raw_thresholds: list[float] = []
    labels: list[str] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("raw_quantile_thresholds rows must be JSON objects.")
        quantile = _finite_float(row.get("quantile"))
        weight = _finite_float(row.get("weight"))
        threshold = _finite_float(row.get("threshold"))
        label = str(row.get("quantile_label") or "").strip()
        if (
            quantile is None
            or not 0.0 <= quantile <= 1.0
            or weight is None
            or weight < 0.0
            or threshold is None
            or threshold < 0.0
            or not label
        ):
            raise ValueError(f"Invalid frozen raw threshold row: {row}")
        requested_quantiles.append(quantile)
        requested_weights.append(weight)
        raw_thresholds.append(threshold)
        labels.append(label)
    levels: list[ThresholdLevel] = []
    for row in merged_rows:
        if not isinstance(row, dict):
            raise ValueError("merged_thresholds rows must be JSON objects.")
        threshold = _finite_float(row.get("threshold"))
        weight = _finite_float(row.get("weight"))
        quantiles = tuple(float(value) for value in row.get("quantiles") or ())
        quantile_labels = tuple(str(value) for value in row.get("quantile_labels") or ())
        threshold_id = str(row.get("threshold_id") or "").strip()
        if (
            threshold is None
            or threshold < 0.0
            or weight is None
            or weight < 0.0
            or not quantiles
            or len(quantiles) != len(quantile_labels)
            or not threshold_id
        ):
            raise ValueError(f"Invalid frozen merged threshold row: {row}")
        levels.append(
            ThresholdLevel(
                threshold_id=threshold_id,
                threshold=threshold,
                weight=weight,
                quantiles=quantiles,
                quantile_labels=quantile_labels,
            )
        )
    theta_star_quantile = _finite_float(payload.get("theta_star_quantile"))
    theta_star = _finite_float(payload.get("theta_star"))
    cost_cap_quantile = _finite_float(payload.get("cost_cap_quantile"))
    cost_cap = _finite_float(payload.get("cost_cap"))
    if (
        theta_star_quantile is None
        or theta_star is None
        or cost_cap_quantile is None
        or cost_cap is None
        or cost_cap < 0.0
    ):
        raise ValueError("Frozen thresholds lack finite theta_star/cost_cap values.")
    if not math.isclose(theta_star_quantile, 0.30, abs_tol=FLOAT_TOLERANCE):
        raise ValueError(
            f"Frozen theta_star_quantile must be 0.30, found {theta_star_quantile}."
        )
    required_quantiles = {0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90}
    if {
        round(value, 12) for value in requested_quantiles
    } != {round(value, 12) for value in required_quantiles}:
        raise ValueError(
            "Frozen threshold grid must contain q05,q10,q20,q30,q50,q70,q90."
        )
    quantile_to_threshold = dict(zip(requested_quantiles, raw_thresholds))
    if not math.isclose(
        theta_star,
        quantile_to_threshold[0.30],
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("Frozen theta_star is not the frozen q30 threshold.")
    if not math.isclose(
        cost_cap,
        quantile_to_threshold[0.90],
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("Frozen cost_cap is not the frozen q90 threshold.")
    if sum(requested_weights) <= 0.0 or sum(level.weight for level in levels) <= 0.0:
        raise ValueError("Frozen threshold weights must have positive total weight.")
    return ThresholdBundle(
        finite_distance_count=int(
            payload.get("finite_strict_flip_distance_count") or 0
        ),
        requested_quantiles=tuple(requested_quantiles),
        requested_weights=tuple(requested_weights),
        raw_thresholds=tuple(raw_thresholds),
        quantile_labels=tuple(labels),
        levels=tuple(levels),
        theta_star_quantile=theta_star_quantile,
        theta_star=theta_star,
        cost_cap_quantile=cost_cap_quantile,
        cost_cap=cost_cap,
    )


def _agreement_value(
    payloads: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> list[Any]:
    values: list[Any] = []
    for payload in payloads:
        for key in keys:
            if key in payload and payload[key] is not None:
                values.append(payload[key])
                break
    return values


def load_and_verify_frozen_selector(
    frozen_selector_root: str | Path,
) -> FrozenSelectorPackage:
    root = Path(frozen_selector_root).expanduser().resolve()
    manifest_path = root / "frozen_selector_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen selector manifest: {manifest_path}")
    for relative in REQUIRED_FROZEN_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing frozen selector artifact: {path}")

    manifest = _read_json(manifest_path)
    marker = _read_json(root / "_FROZEN.json")
    decision = _read_json(root / "calibration_decision.json")
    threshold_payload = _read_json(root / "thresholds.json")
    hash_entries = _extract_hash_entries(manifest)
    verified: dict[str, str] = {}
    for relative, expected in sorted(hash_entries.items()):
        path = _safe_frozen_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen hash entry does not exist: {path}")
        actual = _sha256_file(path)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Frozen SHA256 mismatch for {relative}: "
                f"expected={expected}, actual={actual}"
            )
        verified[relative] = actual
    normalized_hash_paths = {Path(value).as_posix().lstrip("./") for value in verified}
    for relative in REQUIRED_FROZEN_FILES:
        if Path(relative).as_posix() not in normalized_hash_paths:
            raise ValueError(
                f"Frozen manifest does not hash required artifact: {relative}"
            )

    metadata_payloads = (manifest, marker, decision)
    frozen_values = _agreement_value(
        metadata_payloads,
        ("frozen", "selector_frozen", "is_frozen"),
    )
    if not frozen_values or not all(_bool_value(value) for value in frozen_values):
        raise ValueError("Frozen selector metadata does not consistently assert frozen=true.")
    variants = [
        str(value)
        for value in _agreement_value(
            metadata_payloads,
            ("selected_variant", "variant_name", "winner"),
        )
    ]
    if not variants or any(value != EXPECTED_SELECTED_VARIANT for value in variants):
        raise ValueError(
            f"Frozen selected_variant must be {EXPECTED_SELECTED_VARIANT}; "
            f"found={variants}."
        )
    top_values = [
        int(value)
        for value in _agreement_value(metadata_payloads, ("top_k", "selected_top_k"))
    ]
    table_values = [
        int(value)
        for value in _agreement_value(metadata_payloads, ("table_k",))
    ]
    if not top_values or any(value != EXPECTED_TOP_K for value in top_values):
        raise ValueError(f"Frozen top_k must be {EXPECTED_TOP_K}; found={top_values}.")
    if not table_values or any(value != EXPECTED_TABLE_K for value in table_values):
        raise ValueError(
            f"Frozen table_k must be {EXPECTED_TABLE_K}; found={table_values}."
        )
    selection_test_values = _agreement_value(
        metadata_payloads,
        ("test_used_for_selection",),
    )
    if not selection_test_values or any(_bool_value(value) for value in selection_test_values):
        raise ValueError("Frozen selector must prove test_used_for_selection=false.")

    sequence = _read_jsonl(root / "selected_variant/selected_sequence.jsonl")
    if len(sequence) != EXPECTED_TOP_K:
        raise ValueError(
            f"Frozen selected_sequence must contain {EXPECTED_TOP_K} rows, "
            f"found {len(sequence)}."
        )
    candidate_ids: list[str] = []
    for expected_rank, row in enumerate(sequence, start=1):
        rank = int(row.get("rank") or 0)
        candidate_id = str(row.get("candidate_id") or "").strip()
        fragment = str(row.get("canonical_fragment") or "").strip()
        if rank != expected_rank:
            raise ValueError(
                f"Frozen selected_sequence rank mismatch: expected {expected_rank}, "
                f"found {rank}."
            )
        if not candidate_id or not fragment:
            raise ValueError("Frozen candidate requires candidate_id and canonical_fragment.")
        if canonicalize_smiles(fragment) is None:
            raise ValueError(f"Frozen candidate is not valid SMILES: {fragment!r}")
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Frozen selected_sequence contains duplicate candidate IDs.")

    top10 = _candidate_ids_from_json(
        json.loads((root / "selected_variant/selected_top10.json").read_text())
    )
    top20 = _candidate_ids_from_json(
        json.loads((root / "selected_variant/selected_top20.json").read_text())
    )
    if top10 != candidate_ids[:EXPECTED_TABLE_K]:
        raise ValueError("Frozen selected_top10 is not selected_sequence[:10].")
    if top20 != candidate_ids:
        raise ValueError("Frozen selected_top20 is not the complete selected_sequence.")

    thresholds = _load_threshold_bundle(threshold_payload)
    selector_hash_payload = {
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_files": verified,
        "selected_candidate_ids": candidate_ids,
    }
    selector_hash = hashlib.sha256(
        _json_dumps(selector_hash_payload).encode("utf-8")
    ).hexdigest()
    return FrozenSelectorPackage(
        root=root,
        manifest=manifest,
        frozen_marker=marker,
        calibration_decision=decision,
        threshold_payload=threshold_payload,
        thresholds=thresholds,
        selected_sequence=tuple(dict(row) for row in sequence),
        selected_variant=EXPECTED_SELECTED_VARIANT,
        top_k=EXPECTED_TOP_K,
        table_k=EXPECTED_TABLE_K,
        frozen_selector_hash=selector_hash,
        verified_file_sha256=verified,
    )


def frozen_threshold_output(package: FrozenSelectorPackage) -> dict[str, Any]:
    payload = copy.deepcopy(package.threshold_payload)
    payload.update(
        {
            "threshold_source": "frozen_calibration_selector",
            "test_threshold_fitting": False,
            "test_candidate_selection": False,
            "test_variant_selection": False,
            "frozen_selector_hash": package.frozen_selector_hash,
        }
    )
    return payload


def load_test_parents(
    test_csv: str | Path,
    *,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
    label_col: str = "label",
    cohort_name: str = "test",
    expected_parent_count: int = EXPECTED_TEST_PARENT_COUNT,
) -> list[CalibrationParent]:
    path = Path(test_csv).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Test CSV is missing or empty: {path}")
    if "calibration" in path.name.lower():
        raise ValueError(f"Calibration CSV is forbidden for final test evaluation: {path}")
    if str(cohort_name).strip().lower() != "test":
        raise ValueError("cohort_name must be exactly 'test'.")
    rows = _read_csv(path)
    if len(rows) != int(expected_parent_count):
        raise ValueError(
            f"Test parent count mismatch: expected {expected_parent_count}, "
            f"found {len(rows)}."
        )
    required = {id_col, smiles_col, label_col}
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Test CSV is missing columns: {missing}")
    parents: list[CalibrationParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        parent_id = str(row.get(id_col) or "").strip()
        smiles = str(row.get(smiles_col) or "").strip()
        try:
            label = int(float(str(row.get(label_col) or "")))
        except ValueError as exc:
            raise ValueError(f"Invalid label at {path}:{row_number}") from exc
        split = str(row.get("split") or cohort_name).strip().lower()
        if not parent_id or parent_id in seen:
            raise ValueError(f"Missing/duplicate parent ID at {path}:{row_number}")
        if label != SOURCE_LABEL:
            raise ValueError(
                f"Test source label must be {SOURCE_LABEL} at {path}:{row_number}."
            )
        if split != "test":
            raise ValueError(f"Test CSV row has non-test split={split!r}.")
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            raise ValueError(f"Invalid test parent SMILES at {path}:{row_number}.")
        seen.add(parent_id)
        parents.append(
            CalibrationParent(
                parent_id=parent_id,
                smiles=canonical,
                label=label,
                split="test",
            )
        )
    parents.sort(key=lambda item: item.parent_id)
    return parents


def test_cohort_hash(parents: Sequence[CalibrationParent]) -> str:
    payload = [
        {
            "parent_id": parent.parent_id,
            "canonical_smiles": canonicalize_smiles(parent.smiles),
            "label": parent.label,
            "split": parent.split,
        }
        for parent in parents
    ]
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _pair_rows_to_matrix(
    pair_rows: Sequence[dict[str, Any]],
    parents: Sequence[CalibrationParent],
    candidates: Sequence[dict[str, Any]],
) -> MatrixData:
    parent_ids = tuple(parent.parent_id for parent in parents)
    candidate_ids = tuple(str(row["candidate_id"]) for row in candidates)
    parent_index = {value: index for index, value in enumerate(parent_ids)}
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    distances = np.full((len(parents), len(candidates)), np.inf, dtype=np.float64)
    cf_drops = np.full_like(distances, np.nan)
    applicable = np.zeros_like(distances, dtype=bool)
    seen: set[tuple[str, str]] = set()
    for row in pair_rows:
        parent_id = str(row.get("parent_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        key = (parent_id, candidate_id)
        if key in seen:
            raise ValueError(f"Duplicate test pair row: {key}")
        if parent_id not in parent_index or candidate_id not in candidate_index:
            raise ValueError(f"Unexpected test pair row: {key}")
        seen.add(key)
        left = parent_index[parent_id]
        right = candidate_index[candidate_id]
        applicable[left, right] = _bool_value(row.get("applicable"))
        if not _bool_value(row.get("pair_strict_flip")):
            if _finite_float(row.get("wnode_distance")) is not None:
                raise ValueError(f"Non-flip pair has WNode distance: {key}")
            continue
        distance = _finite_float(row.get("wnode_distance"))
        cf_drop = _finite_float(row.get("cf_drop"))
        if distance is None or distance < 0.0 or cf_drop is None:
            raise ValueError(f"Strict-flip pair lacks finite distance/CFDrop: {key}")
        distances[left, right] = distance
        cf_drops[left, right] = cf_drop
    expected = len(parents) * len(candidates)
    if len(seen) != expected:
        raise ValueError(
            f"Test matrix is incomplete: rows={len(seen)}, expected={expected}."
        )
    return MatrixData(
        matrix_run_dir=Path("."),
        parent_ids=parent_ids,
        candidate_rows=tuple(dict(row) for row in candidates),
        distances=distances,
        cf_drops=cf_drops,
        applicable=applicable,
        full_finite_distances=distances[np.isfinite(distances)],
        full_parent_count=len(parents),
        full_candidate_count=len(candidates),
        full_pair_count=expected,
        full_strict_flip_pair_count=int(np.count_nonzero(np.isfinite(distances))),
        summary={},
        manifest={},
        full_candidate_rows=tuple(dict(row) for row in candidates),
    )


def _pairwise_mean(matrix: np.ndarray, count: int) -> float:
    if count < 2:
        return 0.0
    selected = matrix[:count, :count]
    upper = selected[np.triu_indices(count, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def compute_frozen_prefix_metrics(
    pair_rows: Sequence[dict[str, Any]],
    parents: Sequence[CalibrationParent],
    candidates: Sequence[dict[str, Any]],
    thresholds: ThresholdBundle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = _pair_rows_to_matrix(pair_rows, parents, candidates)
    chemistry = build_candidate_chemistry(
        candidates,
        size_normalization_rows=candidates,
    )
    coverage_redundancy = build_coverage_redundancy_matrix(
        matrix.distances,
        thresholds.levels,
    )
    best = np.full(len(parents), np.inf, dtype=np.float64)
    best_cf_drop = np.full(len(parents), np.nan, dtype=np.float64)
    best_candidate = np.full(len(parents), -1, dtype=np.int64)
    applicable = np.zeros(len(parents), dtype=bool)
    metrics: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    previous_coverage = -math.inf
    previous_capped_mean = math.inf
    for candidate_index in range(len(candidates)):
        candidate_distances = matrix.distances[:, candidate_index]
        candidate_drops = matrix.cf_drops[:, candidate_index]
        improved = candidate_distances < best
        tied_better_drop = (
            np.isfinite(candidate_distances)
            & np.isclose(candidate_distances, best, atol=0.0, rtol=0.0)
            & (
                np.nan_to_num(candidate_drops, nan=-np.inf)
                > np.nan_to_num(best_cf_drop, nan=-np.inf)
            )
        )
        update = improved | tied_better_drop
        best[update] = candidate_distances[update]
        best_cf_drop[update] = candidate_drops[update]
        best_candidate[update] = candidate_index
        applicable |= matrix.applicable[:, candidate_index]

        finite = np.isfinite(best)
        theta_covered = best <= thresholds.theta_star
        capped_mean, capped_median, capped = fixed_denominator_capped_cost(
            best,
            thresholds.cost_cap,
        )
        conditional = best[finite]
        theta_costs = best[theta_covered]
        applicable_count = int(np.count_nonzero(applicable))
        strict_count = int(np.count_nonzero(finite))
        coverage = float(np.mean(theta_covered))
        if coverage + FLOAT_TOLERANCE < previous_coverage:
            raise AssertionError("Frozen prefix coverage decreased with K.")
        if capped_mean > previous_capped_mean + FLOAT_TOLERANCE:
            raise AssertionError("Frozen fixed capped mean cost increased with K.")
        previous_coverage = coverage
        previous_capped_mean = capped_mean
        row: dict[str, Any] = {
            "k": candidate_index + 1,
            "ccrcov_theta_star": coverage,
            "weighted_multi_threshold_utility": weighted_multi_threshold_utility(
                best,
                thresholds.levels,
            ),
            "applicable_rate": float(applicable_count / len(parents)),
            "strict_flip_parent_count": strict_count,
            "flip_rate_among_applicable": (
                float(strict_count / applicable_count) if applicable_count else 0.0
            ),
            "num_theta_star_covered": int(np.count_nonzero(theta_covered)),
            "fixed_capped_mean_cost": capped_mean,
            "fixed_capped_median_cost": capped_median,
            "conditional_mean_cost": (
                float(np.mean(conditional)) if conditional.size else None
            ),
            "conditional_median_cost": (
                float(np.median(conditional)) if conditional.size else None
            ),
            "theta_star_conditional_mean_cost": (
                float(np.mean(theta_costs)) if theta_costs.size else None
            ),
            "theta_star_conditional_median_cost": (
                float(np.median(theta_costs)) if theta_costs.size else None
            ),
            "mean_cf_drop": (
                float(np.mean(best_cf_drop[finite])) if np.any(finite) else None
            ),
            "theta_star_mean_cf_drop": (
                float(np.mean(best_cf_drop[theta_covered]))
                if np.any(theta_covered)
                else None
            ),
            "coverage_redundancy": _pairwise_mean(
                coverage_redundancy,
                candidate_index + 1,
            ),
            "structural_redundancy": _pairwise_mean(
                chemistry.structural_similarity,
                candidate_index + 1,
            ),
        }
        for label, threshold in zip(
            thresholds.quantile_labels,
            thresholds.raw_thresholds,
        ):
            row[f"ccrcov_theta_{label}"] = single_threshold_coverage(best, threshold)
        metrics.append(row)
        for parent_index, parent in enumerate(parents):
            selected_index = int(best_candidate[parent_index])
            parent_rows.append(
                {
                    "k": candidate_index + 1,
                    "parent_id": parent.parent_id,
                    "best_distance": (
                        float(best[parent_index])
                        if math.isfinite(float(best[parent_index]))
                        else None
                    ),
                    "capped_distance": float(capped[parent_index]),
                    "best_candidate_id": (
                        str(candidates[selected_index]["candidate_id"])
                        if selected_index >= 0
                        else None
                    ),
                    "cf_drop": (
                        float(best_cf_drop[parent_index])
                        if math.isfinite(float(best_cf_drop[parent_index]))
                        else None
                    ),
                    "strict_recourse_available": bool(finite[parent_index]),
                    "theta_star_covered": bool(theta_covered[parent_index]),
                    "applicable": bool(applicable[parent_index]),
                }
            )
    return metrics, parent_rows


def build_figure4_rows(
    metrics: Sequence[dict[str, Any]],
    thresholds: ThresholdBundle,
    *,
    table_k: int,
    top_k: int,
    num_parents: int,
) -> list[dict[str, Any]]:
    by_k = {int(row["k"]): row for row in metrics}
    rows: list[dict[str, Any]] = []
    for k in (int(table_k), int(top_k)):
        metric = by_k[k]
        for quantile, label, threshold in zip(
            thresholds.requested_quantiles,
            thresholds.quantile_labels,
            thresholds.raw_thresholds,
        ):
            coverage = float(metric[f"ccrcov_theta_{label}"])
            rows.append(
                {
                    "k": k,
                    "quantile": quantile,
                    "quantile_label": label,
                    "threshold": threshold,
                    "coverage": coverage,
                    "num_covered": int(round(coverage * int(num_parents))),
                    "threshold_source": "frozen_calibration_selector",
                }
            )
    return rows


def build_table_row(
    metric: dict[str, Any],
    *,
    package: FrozenSelectorPackage,
    num_test_parents: int,
) -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "dataset": "Mutagenicity",
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "k": int(metric["k"]),
        "theta": package.thresholds.theta_star,
        "coverage": metric["ccrcov_theta_star"],
        "applicable_rate": metric["applicable_rate"],
        "flip_rate_among_applicable": metric["flip_rate_among_applicable"],
        "mean_cf_drop": metric["theta_star_mean_cf_drop"],
        "conditional_mean_cost": metric["conditional_mean_cost"],
        "conditional_median_cost": metric["conditional_median_cost"],
        "fixed_capped_mean_cost": metric["fixed_capped_mean_cost"],
        "fixed_capped_median_cost": metric["fixed_capped_median_cost"],
        "coverage_redundancy": metric["coverage_redundancy"],
        "structural_redundancy": metric["structural_redundancy"],
        "num_test_parents": num_test_parents,
        "num_candidates": int(metric["k"]),
        "selected_variant": package.selected_variant,
    }


def _checkpoint_payload(
    fingerprint: str,
    completed_keys: set[str],
    *,
    run_complete: bool,
) -> dict[str, Any]:
    return {
        "config_fingerprint": fingerprint,
        "completed_pair_keys": sorted(completed_keys),
        "completed_pair_count": len(completed_keys),
        "run_complete": bool(run_complete),
        "updated_at": _utc_now(),
    }


def _provider_stats(provider: DistanceProtocol) -> dict[str, Any]:
    try:
        return dict(provider.stats_dict())
    except Exception:
        return {}


def build_frozen_test_run(
    *,
    frozen_selector_root: str | Path,
    test_csv: str | Path,
    teacher_path: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    output_dir: str | Path,
    wnode_cache_db: str | Path,
    teacher: TeacherProtocol,
    distance_provider: DistanceProtocol,
    config: FrozenTestConfig | None = None,
    deletion_fn: Callable[[str, str], list[dict[str, Any]]] = (
        hard_delete_substructure_any_match
    ),
    _interrupt_after_pairs: int | None = None,
) -> dict[str, Any]:
    resolved = config or FrozenTestConfig()
    if resolved.flush_every <= 0:
        raise ValueError("flush_every must be positive.")
    if float(resolved.wnode_size_penalty_beta) < 0.0:
        raise ValueError("wnode_size_penalty_beta must be non-negative.")
    package = load_and_verify_frozen_selector(frozen_selector_root)
    parents = load_test_parents(
        test_csv,
        id_col=resolved.id_col,
        smiles_col=resolved.smiles_col,
        label_col=resolved.label_col,
        cohort_name=resolved.cohort_name,
        expected_parent_count=resolved.expected_parent_count,
    )
    if len(package.selected_sequence) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Frozen selector candidate count is not 20.")
    cohort_hash = test_cohort_hash(parents)
    self_test = run_wnode_self_test(distance_provider)
    before_cache: dict[str, dict[str, Any]] = {}
    for parent in parents:
        result = predict_with_teacher(teacher, parent.smiles, SOURCE_LABEL)
        if not result.get("ok") or result.get("pred_label") != SOURCE_LABEL:
            raise ValueError(
                "Final test cohort is not teacher-confirmed source-label=1: "
                f"parent_id={parent.parent_id}, result={result}"
            )
        before_cache[parent.parent_id] = result

    destination = Path(output_dir).expanduser().resolve()
    complete_path = destination / "_RUN_COMPLETE.json"
    if complete_path.is_file():
        raise FileExistsError(
            f"Completed frozen test run cannot be rerun: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint_payload = {
        "frozen_selector_hash": package.frozen_selector_hash,
        "test_csv": _file_identity(test_csv),
        "test_cohort_hash": cohort_hash,
        "teacher_path": _file_identity(teacher_path),
        "molclr_root": _file_identity(molclr_root),
        "molclr_checkpoint": _file_identity(molclr_checkpoint),
        "wnode_cache_db": str(Path(wnode_cache_db).expanduser().resolve()),
        "wnode_size_penalty_beta": float(resolved.wnode_size_penalty_beta),
        "candidate_ids_in_order": list(package.candidate_ids),
        "thresholds_sha256": _sha256_file(package.root / "thresholds.json"),
        "id_col": resolved.id_col,
        "smiles_col": resolved.smiles_col,
        "label_col": resolved.label_col,
        "cohort_name": resolved.cohort_name,
        "expected_parent_count": resolved.expected_parent_count,
        "local_files_only": bool(resolved.local_files_only),
    }
    fingerprint = hashlib.sha256(
        _json_dumps(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    manifest_path = destination / "run_manifest.json"
    checkpoint_path = destination / "resume_checkpoint.json"
    pair_path = destination / "pair_matrix.jsonl"
    match_path = destination / "match_instances.jsonl"
    existing_entries = list(destination.iterdir())
    if existing_entries and not resolved.resume:
        raise FileExistsError(
            f"Output directory is non-empty and resume is disabled: {destination}"
        )
    if existing_entries:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise ValueError(
                "Resume requires run_manifest.json and resume_checkpoint.json."
            )
        manifest = _read_json(manifest_path)
        checkpoint = _read_json(checkpoint_path)
        if manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume manifest is incompatible with this frozen test run.")
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint is incompatible with this frozen test run.")
    else:
        manifest = {
            "created_at": _utc_now(),
            "config_fingerprint": fingerprint,
            "inputs": fingerprint_payload,
            "dataset": "Mutagenicity",
            "cohort_name": "test",
            "source_label": SOURCE_LABEL,
            "target_label": TARGET_LABEL,
            "strict_flip_definition": "pred_before == 1 and pred_after == 0",
            "cf_drop_definition": "p1_before - p1_after",
            "distance_type": DISTANCE_TYPE,
            "distance_line": DISTANCE_LINE,
            "selector_frozen": True,
            "selector_frozen_before_test": True,
            "selected_variant": package.selected_variant,
            "test_used": True,
            "test_used_for_selection": False,
            "test_threshold_fitting": False,
            "test_candidate_selection": False,
            "test_variant_selection": False,
            "run_complete": False,
        }
        _write_json(manifest_path, manifest)
        _write_json(
            checkpoint_path,
            _checkpoint_payload(fingerprint, set(), run_complete=False),
        )
    _write_jsonl(destination / "selected_sequence.jsonl", package.selected_sequence)
    _write_json(destination / "thresholds.json", frozen_threshold_output(package))

    existing_pairs = _read_jsonl(pair_path, allow_truncated_last_line=True)
    pair_rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    for row in existing_pairs:
        key = _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        if key not in completed:
            completed.add(key)
            pair_rows.append(row)
    match_rows = [
        row
        for row in _read_jsonl(match_path, allow_truncated_last_line=True)
        if _pair_key(str(row["parent_id"]), str(row["candidate_id"])) in completed
    ]
    _write_jsonl(pair_path, pair_rows)
    _write_jsonl(match_path, match_rows)
    _write_json(
        checkpoint_path,
        _checkpoint_payload(fingerprint, completed, run_complete=False),
    )

    pending_pairs: list[dict[str, Any]] = []
    pending_matches: list[dict[str, Any]] = []
    new_pair_count = 0

    def flush() -> None:
        if not pending_pairs:
            return
        _append_jsonl(match_path, pending_matches)
        _append_jsonl(pair_path, pending_pairs)
        for row in pending_pairs:
            completed.add(_pair_key(str(row["parent_id"]), str(row["candidate_id"])))
        _write_json(
            checkpoint_path,
            _checkpoint_payload(fingerprint, completed, run_complete=False),
        )
        pending_pairs.clear()
        pending_matches.clear()

    for parent in parents:
        for candidate in package.selected_sequence:
            key = _pair_key(parent.parent_id, str(candidate["candidate_id"]))
            if key in completed:
                continue
            pair, matches = evaluate_parent_candidate_pair(
                parent,
                candidate,
                teacher=teacher,
                distance_provider=distance_provider,
                before_prediction=before_cache[parent.parent_id],
                deletion_fn=deletion_fn,
            )
            pending_pairs.append(pair)
            pending_matches.extend(matches)
            new_pair_count += 1
            if len(pending_pairs) >= resolved.flush_every:
                flush()
            if (
                _interrupt_after_pairs is not None
                and new_pair_count >= int(_interrupt_after_pairs)
            ):
                flush()
                raise FrozenTestInterrupted("Intentional frozen-test resume interruption.")
    flush()

    pair_rows = _read_jsonl(pair_path)
    match_rows = _read_jsonl(match_path)
    expected_keys = {
        _pair_key(parent.parent_id, candidate_id)
        for parent in parents
        for candidate_id in package.candidate_ids
    }
    actual_keys = {
        _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        for row in pair_rows
    }
    if len(pair_rows) != len(actual_keys):
        raise RuntimeError("Frozen test pair matrix contains duplicate pair rows.")
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Frozen test Cartesian matrix is incomplete: "
            f"missing={len(expected_keys - actual_keys)}, "
            f"unexpected={len(actual_keys - expected_keys)}"
        )
    if len(pair_rows) != len(parents) * EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("Frozen test matrix does not contain parent_count x 20 rows.")

    metrics, parent_best = compute_frozen_prefix_metrics(
        pair_rows,
        parents,
        package.selected_sequence,
        package.thresholds,
    )
    _write_csv(destination / "prefix_metrics.csv", metrics)
    _write_json(destination / "prefix_metrics.json", metrics)
    _write_csv(destination / "parent_best_distances.csv", parent_best)
    _write_csv(destination / "figure3_coverage_vs_k.csv", metrics)
    figure4_rows = build_figure4_rows(
        metrics,
        package.thresholds,
        table_k=package.table_k,
        top_k=package.top_k,
        num_parents=len(parents),
    )
    _write_csv(destination / "figure4_coverage_vs_threshold.csv", figure4_rows)
    by_k = {int(row["k"]): row for row in metrics}
    table10 = build_table_row(
        by_k[package.table_k],
        package=package,
        num_test_parents=len(parents),
    )
    table20 = build_table_row(
        by_k[package.top_k],
        package=package,
        num_test_parents=len(parents),
    )
    _write_csv(destination / "table2_ours_k10.csv", [table10])
    _write_csv(destination / "table2_ours_k20.csv", [table20])

    provider_stats = _provider_stats(distance_provider)
    applicable_parents = {
        str(row["parent_id"]) for row in pair_rows if _bool_value(row.get("applicable"))
    }
    strict_parents = {
        str(row["parent_id"])
        for row in pair_rows
        if _bool_value(row.get("pair_strict_flip"))
    }
    summary = {
        "selector_frozen": True,
        "selector_frozen_before_test": True,
        "selected_variant": package.selected_variant,
        "test_used": True,
        "test_used_for_selection": False,
        "test_threshold_fitting": False,
        "test_candidate_selection": False,
        "test_variant_selection": False,
        "test_parent_count": len(parents),
        "candidate_count": len(package.selected_sequence),
        "expected_pair_rows": len(expected_keys),
        "actual_pair_rows": len(pair_rows),
        "complete_cartesian": actual_keys == expected_keys,
        "applicable_pair_count": sum(
            _bool_value(row.get("applicable")) for row in pair_rows
        ),
        "strict_flip_pair_count": sum(
            _bool_value(row.get("pair_strict_flip")) for row in pair_rows
        ),
        "finite_wnode_count": sum(
            _finite_float(row.get("wnode_distance")) is not None for row in pair_rows
        ),
        "valid_match_instance_count": sum(
            _bool_value(row.get("delete_valid")) for row in match_rows
        ),
        "strict_flip_match_instance_count": sum(
            _bool_value(row.get("teacher_strict_flip")) for row in match_rows
        ),
        "parent_coverage_any_applicable": len(applicable_parents) / len(parents),
        "parent_coverage_any_strict_flip": len(strict_parents) / len(parents),
        "k10_ccrcov_theta_star": table10["coverage"],
        "k20_ccrcov_theta_star": table20["coverage"],
        "k10_weighted_multi_threshold_utility": by_k[10][
            "weighted_multi_threshold_utility"
        ],
        "k20_weighted_multi_threshold_utility": by_k[20][
            "weighted_multi_threshold_utility"
        ],
        "k10_fixed_capped_mean_cost": table10["fixed_capped_mean_cost"],
        "k20_fixed_capped_mean_cost": table20["fixed_capped_mean_cost"],
        "theta_star": package.thresholds.theta_star,
        "cost_cap": package.thresholds.cost_cap,
        "wnode_size_penalty_beta": float(resolved.wnode_size_penalty_beta),
        "frozen_selector_hash": package.frozen_selector_hash,
        "test_cohort_hash": cohort_hash,
        "wnode_self_test_passed": bool(self_test["passed"]),
        "wnode_self_test": self_test,
        "cache_hit_rate": provider_stats.get(
            "pair_distance_cache_hit_rate",
            provider_stats.get("cache_hit_rate", 0.0),
        ),
        "node_embedding_cache_hit_rate": provider_stats.get(
            "node_embedding_cache_hit_rate",
            0.0,
        ),
        "run_complete": True,
    }
    _write_json(destination / "summary.json", summary)
    manifest.update(
        {
            "run_complete": True,
            "completed_at": _utc_now(),
            "summary": str((destination / "summary.json").resolve()),
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        checkpoint_path,
        _checkpoint_payload(fingerprint, actual_keys, run_complete=True),
    )
    _write_json(
        complete_path,
        {
            "run_complete": True,
            "config_fingerprint": fingerprint,
            "frozen_selector_hash": package.frozen_selector_hash,
            "test_cohort_hash": cohort_hash,
            "actual_pair_rows": len(pair_rows),
            "completed_at": _utc_now(),
        },
    )
    return summary


def _values_equal(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    if left is None or left == "":
        return right is None or right == ""
    if right is None or right == "":
        return False
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    if left_number is not None and right_number is not None:
        return math.isclose(
            left_number,
            right_number,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return _bool_value(left) == _bool_value(right)
    return str(left) == str(right)


def _audit_match_aggregation(
    pair_rows: Sequence[dict[str, Any]],
    match_rows: Sequence[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in match_rows:
        key = _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        grouped.setdefault(key, []).append(row)
    for pair in pair_rows:
        key = _pair_key(str(pair["parent_id"]), str(pair["candidate_id"]))
        matches = grouped.get(key, [])
        if int(pair.get("num_matches") or 0) != len(matches):
            raise AssertionError(f"num_matches mismatch for {key}")
        strict_finite = []
        for match in matches:
            expected_strict = bool(
                _bool_value(match.get("delete_valid"))
                and int(match.get("pred_before")) == SOURCE_LABEL
                and int(match.get("pred_after")) == TARGET_LABEL
            ) if (
                match.get("pred_before") is not None
                and match.get("pred_after") is not None
            ) else False
            if _bool_value(match.get("teacher_strict_flip")) != expected_strict:
                raise AssertionError(f"Match strict-flip mismatch for {key}")
            distance = _finite_float(match.get("wnode_distance"))
            if expected_strict and distance is not None and distance >= 0.0:
                strict_finite.append(match)
            elif not expected_strict and distance is not None:
                raise AssertionError(f"Non-flip match has WNode distance for {key}")
        strict_finite.sort(
            key=lambda row: (
                float(row["wnode_distance"]),
                -float(
                    row["cf_drop"]
                    if _finite_float(row.get("cf_drop")) is not None
                    else float("-inf")
                ),
                int(row["match_index"]),
            )
        )
        best = strict_finite[0] if strict_finite else None
        if _bool_value(pair.get("pair_strict_flip")) != (best is not None):
            raise AssertionError(f"Pair strict-flip aggregate mismatch for {key}")
        if best is None:
            if _finite_float(pair.get("wnode_distance")) is not None:
                raise AssertionError(f"Non-flip pair has WNode distance for {key}")
            continue
        for field in ("wnode_distance", "cf_drop", "residual_smiles", "best_match_index"):
            match_field = "match_index" if field == "best_match_index" else field
            if not _values_equal(pair.get(field), best.get(match_field)):
                raise AssertionError(f"Best-match {field} mismatch for {key}")
        if list(pair.get("best_match_atom_indices") or []) != list(
            best.get("match_atom_indices") or []
        ):
            raise AssertionError(f"Best-match atom indices mismatch for {key}")


def audit_frozen_test_run(
    run_dir: str | Path,
    *,
    frozen_selector_root: str | Path,
    test_csv: str | Path,
    expected_parent_count: int = EXPECTED_TEST_PARENT_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_pair_count: int = EXPECTED_PAIR_COUNT,
    expected_top_k: int = EXPECTED_TOP_K,
    expected_table_k: int = EXPECTED_TABLE_K,
    require_complete_cartesian: bool = True,
    require_frozen_thresholds: bool = True,
    require_frozen_candidate_order: bool = True,
    require_monotonic_coverage: bool = True,
    require_nonincreasing_capped_cost: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    for relative in REQUIRED_OUTPUT_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing frozen test output: {path}")
    package = load_and_verify_frozen_selector(frozen_selector_root)
    parents = load_test_parents(
        test_csv,
        expected_parent_count=expected_parent_count,
    )
    selected = _read_jsonl(root / "selected_sequence.jsonl")
    selected_ids = [str(row.get("candidate_id") or "") for row in selected]
    if require_frozen_candidate_order and selected_ids != list(package.candidate_ids):
        raise AssertionError("Run candidate order differs from frozen selected_sequence.")
    if len(selected) != expected_candidate_count or len(set(selected_ids)) != len(selected):
        raise AssertionError("Run selected sequence is not 20 unique candidates.")
    if expected_top_k != package.top_k or expected_table_k != package.table_k:
        raise AssertionError("Expected K settings differ from frozen package.")

    actual_thresholds = _read_json(root / "thresholds.json")
    expected_thresholds = frozen_threshold_output(package)
    if require_frozen_thresholds and actual_thresholds != expected_thresholds:
        raise AssertionError("Run thresholds differ from frozen calibration thresholds.")
    if actual_thresholds.get("test_threshold_fitting") is not False:
        raise AssertionError("Run does not prove test_threshold_fitting=false.")

    pair_rows = _read_jsonl(root / "pair_matrix.jsonl")
    match_rows = _read_jsonl(root / "match_instances.jsonl")
    parent_ids = {parent.parent_id for parent in parents}
    expected_keys = {
        _pair_key(parent_id, candidate_id)
        for parent_id in parent_ids
        for candidate_id in selected_ids
    }
    actual_keys = [
        _pair_key(str(row.get("parent_id")), str(row.get("candidate_id")))
        for row in pair_rows
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise AssertionError("Frozen test matrix contains duplicate pair keys.")
    if len(parent_ids) != expected_parent_count:
        raise AssertionError("Unexpected number of unique test parents.")
    if len(pair_rows) != expected_pair_count:
        raise AssertionError(
            f"Pair count mismatch: expected={expected_pair_count}, found={len(pair_rows)}."
        )
    if require_complete_cartesian and set(actual_keys) != expected_keys:
        raise AssertionError("Frozen test matrix is not the complete Cartesian product.")
    for row in pair_rows:
        is_flip = _bool_value(row.get("pair_strict_flip"))
        if is_flip:
            if not _bool_value(row.get("applicable")):
                raise AssertionError("Strict-flip pair is not applicable.")
            if int(row.get("pred_before")) != 1 or int(row.get("pred_after")) != 0:
                raise AssertionError("Pair strict-flip definition is not 1 -> 0.")
            distance = _finite_float(row.get("wnode_distance"))
            if distance is None or distance < 0.0:
                raise AssertionError("Strict-flip pair lacks finite non-negative WNode.")
        elif _finite_float(row.get("wnode_distance")) is not None:
            raise AssertionError("Non-flip pair must have null WNode distance.")
    _audit_match_aggregation(pair_rows, match_rows)

    recomputed_metrics, _ = compute_frozen_prefix_metrics(
        pair_rows,
        parents,
        selected,
        package.thresholds,
    )
    csv_metrics = _read_csv(root / "prefix_metrics.csv")
    if len(csv_metrics) != expected_top_k or len(recomputed_metrics) != expected_top_k:
        raise AssertionError("Prefix metrics do not contain K=1..20.")
    previous_coverage = -math.inf
    previous_cost = math.inf
    for expected_k, (actual, expected) in enumerate(
        zip(csv_metrics, recomputed_metrics),
        start=1,
    ):
        if int(actual.get("k") or 0) != expected_k:
            raise AssertionError("Prefix K is incomplete or reordered.")
        for field, expected_value in expected.items():
            if not _values_equal(actual.get(field), expected_value):
                raise AssertionError(
                    f"Prefix metric mismatch at K={expected_k}, field={field}: "
                    f"actual={actual.get(field)!r}, expected={expected_value!r}"
                )
        coverage = float(expected["ccrcov_theta_star"])
        cost = float(expected["fixed_capped_mean_cost"])
        if require_monotonic_coverage and coverage + FLOAT_TOLERANCE < previous_coverage:
            raise AssertionError("Prefix coverage is not monotonic nondecreasing.")
        if (
            require_nonincreasing_capped_cost
            and cost > previous_cost + FLOAT_TOLERANCE
        ):
            raise AssertionError("Fixed capped cost is not monotonic nonincreasing.")
        previous_coverage = coverage
        previous_cost = cost

    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "run_manifest.json")
    complete = _read_json(root / "_RUN_COMPLETE.json")
    for payload, name in ((summary, "summary"), (manifest, "manifest")):
        if payload.get("test_used_for_selection") is not False:
            raise AssertionError(f"{name} does not prove no test selection.")
        if payload.get("test_threshold_fitting") is not False:
            raise AssertionError(f"{name} does not prove no test threshold fitting.")
        if payload.get("test_candidate_selection") is not False:
            raise AssertionError(f"{name} does not prove no test candidate selection.")
        if payload.get("test_variant_selection") is not False:
            raise AssertionError(f"{name} does not prove no test variant selection.")
    if summary.get("frozen_selector_hash") != package.frozen_selector_hash:
        raise AssertionError("Summary frozen selector hash mismatch.")
    if summary.get("test_cohort_hash") != test_cohort_hash(parents):
        raise AssertionError("Summary test cohort hash mismatch.")
    if summary.get("selected_variant") != EXPECTED_SELECTED_VARIANT:
        raise AssertionError("Summary selected_variant differs from frozen package.")
    if int(summary.get("test_parent_count") or 0) != expected_parent_count:
        raise AssertionError("Summary test_parent_count mismatch.")
    if int(summary.get("candidate_count") or 0) != expected_candidate_count:
        raise AssertionError("Summary candidate_count mismatch.")
    if int(summary.get("actual_pair_rows") or 0) != expected_pair_count:
        raise AssertionError("Summary actual_pair_rows mismatch.")
    if not _values_equal(summary.get("theta_star"), package.thresholds.theta_star):
        raise AssertionError("Summary theta_star differs from frozen package.")
    if not _values_equal(summary.get("cost_cap"), package.thresholds.cost_cap):
        raise AssertionError("Summary cost_cap differs from frozen package.")
    if not _bool_value(summary.get("run_complete")):
        raise AssertionError("Summary run_complete is false.")
    if not _bool_value(manifest.get("run_complete")):
        raise AssertionError("Manifest run_complete is false.")
    if not _bool_value(complete.get("run_complete")):
        raise AssertionError("_RUN_COMPLETE.json is false.")
    if not _bool_value(summary.get("wnode_self_test_passed")):
        raise AssertionError("WNode self-test did not pass.")

    return {
        "audit_passed": True,
        "selector_frozen": True,
        "frozen_selector_sha256_verified": True,
        "frozen_selector_hash": package.frozen_selector_hash,
        "test_parent_count": len(parents),
        "candidate_count": len(selected),
        "pair_count": len(pair_rows),
        "complete_cartesian": set(actual_keys) == expected_keys,
        "candidate_order_matches_frozen": selected_ids == list(package.candidate_ids),
        "thresholds_match_frozen": actual_thresholds == expected_thresholds,
        "test_threshold_fitting": False,
        "test_candidate_selection": False,
        "test_variant_selection": False,
        "run_complete": True,
    }


__all__ = [
    "EXPECTED_CANDIDATE_COUNT",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_SELECTED_VARIANT",
    "EXPECTED_TABLE_K",
    "EXPECTED_TEST_PARENT_COUNT",
    "EXPECTED_TOP_K",
    "FrozenSelectorPackage",
    "FrozenTestConfig",
    "FrozenTestInterrupted",
    "audit_frozen_test_run",
    "build_figure4_rows",
    "build_frozen_test_run",
    "build_table_row",
    "compute_frozen_prefix_metrics",
    "frozen_threshold_output",
    "load_and_verify_frozen_selector",
    "load_test_parents",
    "test_cohort_hash",
]
