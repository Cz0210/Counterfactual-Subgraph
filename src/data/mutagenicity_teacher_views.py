"""Build metadata-complete Mutagenicity views aligned with RF predictions."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLITS = ("train", "val", "calibration", "test")
SOURCE_LABEL = 1
TARGET_LABEL = 0
DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS = {
    "train": 1448,
    "val": 260,
    "calibration": 235,
    "test": 217,
}
PROCESSED_REQUIRED_FIELDS = (
    "molecule_id",
    "smiles",
    "label",
    "semantic_label",
    "scaffold_smiles",
    "split",
)
TEACHER_FIELDS = (
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
)
PREDICTION_REQUIRED_FIELDS = (
    "molecule_id",
    "smiles",
    "label",
    *TEACHER_FIELDS,
)


@dataclass(frozen=True, slots=True)
class TeacherViewConfig:
    source_label: int = SOURCE_LABEL
    target_label: int = TARGET_LABEL
    expected_source_correct_counts: Mapping[str, int] | None = None

    def expected_count(self, split: str) -> int | None:
        counts = self.expected_source_correct_counts
        if counts is None:
            counts = DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS
        value = counts.get(split)
        return None if value is None or int(value) <= 0 else int(value)


def _read_csv(
    path: str | Path,
    *,
    required_fields: Sequence[str],
) -> tuple[list[dict[str, str]], list[str]]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.stat().st_size <= 0:
        raise FileNotFoundError(f"Required Mutagenicity CSV is missing or empty: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(required_fields) - set(fieldnames))
        if missing:
            raise ValueError(f"{target} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Mutagenicity CSV has no data rows: {target}")
    return rows, fieldnames


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: str | Path,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    missing_rows: list[int] = []
    for row_index, row in enumerate(rows, 1):
        molecule_id = str(row.get("molecule_id") or "").strip()
        if not molecule_id:
            missing_rows.append(row_index)
            continue
        if molecule_id in indexed:
            duplicates.append(molecule_id)
        else:
            indexed[molecule_id] = row
    if missing_rows:
        raise ValueError(
            f"{Path(path).resolve()} contains missing molecule_id values at rows "
            f"{missing_rows[:10]}"
        )
    if duplicates:
        raise ValueError(
            f"{Path(path).resolve()} contains duplicate molecule_id values: "
            f"count={len(duplicates)} sample={sorted(set(duplicates))[:10]}"
        )
    return indexed


def _binary_int(value: Any, *, field: str, molecule_id: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"molecule_id={molecule_id} has invalid {field}={value!r}"
        ) from exc
    if parsed not in (0, 1):
        raise ValueError(f"molecule_id={molecule_id} has non-binary {field}={parsed}")
    return parsed


def _boolean(value: Any, *, field: str, molecule_id: str) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"molecule_id={molecule_id} has invalid {field}={value!r}")


def _probability(value: Any, *, field: str, molecule_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"molecule_id={molecule_id} has invalid {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"molecule_id={molecule_id} has out-of-range {field}={value!r}"
        )
    return parsed


def _atomic_write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def merge_processed_and_predictions(
    processed_path: str | Path,
    prediction_path: str | Path,
    *,
    split: str,
    source_label: int = SOURCE_LABEL,
    target_label: int = TARGET_LABEL,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Strictly merge one processed split with its teacher predictions."""

    if split not in SPLITS:
        raise ValueError(f"Unsupported Mutagenicity split: {split!r}")
    processed_rows, processed_fields = _read_csv(
        processed_path, required_fields=PROCESSED_REQUIRED_FIELDS
    )
    prediction_rows, _prediction_fields = _read_csv(
        prediction_path, required_fields=PREDICTION_REQUIRED_FIELDS
    )
    processed_by_id = _index_unique(processed_rows, path=processed_path)
    predictions_by_id = _index_unique(prediction_rows, path=prediction_path)

    processed_ids = set(processed_by_id)
    prediction_ids = set(predictions_by_id)
    missing_predictions = sorted(processed_ids - prediction_ids)
    extra_predictions = sorted(prediction_ids - processed_ids)
    missing_merge_count = len(missing_predictions) + len(extra_predictions)
    if len(processed_rows) != len(prediction_rows) or missing_merge_count:
        raise ValueError(
            "Processed/prediction molecule_id sets are not one-to-one for "
            f"split={split}: processed_rows={len(processed_rows)} "
            f"prediction_rows={len(prediction_rows)} "
            f"missing_predictions={missing_predictions[:10]} "
            f"extra_predictions={extra_predictions[:10]}"
        )

    merged: list[dict[str, Any]] = []
    smiles_mismatches: list[str] = []
    label_mismatches: list[str] = []
    split_mismatches: list[str] = []
    label_counts = {"0": 0, "1": 0}
    for processed in processed_rows:
        molecule_id = str(processed["molecule_id"]).strip()
        prediction = predictions_by_id[molecule_id]
        processed_smiles = str(processed.get("smiles") or "").strip()
        prediction_smiles = str(prediction.get("smiles") or "").strip()
        if processed_smiles != prediction_smiles:
            smiles_mismatches.append(molecule_id)
        processed_label = _binary_int(
            processed.get("label"), field="processed label", molecule_id=molecule_id
        )
        prediction_label = _binary_int(
            prediction.get("label"), field="prediction label", molecule_id=molecule_id
        )
        if processed_label != prediction_label:
            label_mismatches.append(molecule_id)
        processed_split = str(processed.get("split") or "").strip().lower()
        if processed_split != split:
            split_mismatches.append(molecule_id)

        teacher_pred = _binary_int(
            prediction.get("teacher_pred"), field="teacher_pred", molecule_id=molecule_id
        )
        teacher_prob_0 = _probability(
            prediction.get("teacher_prob_0"),
            field="teacher_prob_0",
            molecule_id=molecule_id,
        )
        teacher_prob_1 = _probability(
            prediction.get("teacher_prob_1"),
            field="teacher_prob_1",
            molecule_id=molecule_id,
        )
        if not math.isclose(teacher_prob_0 + teacher_prob_1, 1.0, abs_tol=1e-8):
            raise ValueError(
                f"molecule_id={molecule_id} teacher probabilities do not sum to one"
            )
        teacher_correct = _boolean(
            prediction.get("teacher_correct"),
            field="teacher_correct",
            molecule_id=molecule_id,
        )
        if teacher_correct != (teacher_pred == processed_label):
            raise ValueError(
                f"molecule_id={molecule_id} has inconsistent teacher_correct="
                f"{teacher_correct} for label={processed_label} teacher_pred={teacher_pred}"
            )

        row = dict(processed)
        row.update(
            {
                "molecule_id": molecule_id,
                "smiles": processed_smiles,
                "label": processed_label,
                "teacher_pred": teacher_pred,
                "teacher_prob_0": teacher_prob_0,
                "teacher_prob_1": teacher_prob_1,
                "teacher_correct": teacher_correct,
                "source_label": int(source_label),
                "target_label": int(target_label),
            }
        )
        merged.append(row)
        label_counts[str(processed_label)] += 1

    if smiles_mismatches:
        raise ValueError(
            f"SMILES mismatch between processed and predictions for split={split}: "
            f"count={len(smiles_mismatches)} sample={smiles_mismatches[:10]}"
        )
    if label_mismatches:
        raise ValueError(
            f"Label mismatch between processed and predictions for split={split}: "
            f"count={len(label_mismatches)} sample={label_mismatches[:10]}"
        )
    if split_mismatches:
        raise ValueError(
            f"Processed split column does not equal {split!r}: "
            f"count={len(split_mismatches)} sample={split_mismatches[:10]}"
        )

    output_fields = list(processed_fields)
    for field in (*TEACHER_FIELDS, "source_label", "target_label"):
        if field not in output_fields:
            output_fields.append(field)
    audit = {
        "split": split,
        "processed_path": str(Path(processed_path).expanduser().resolve()),
        "prediction_path": str(Path(prediction_path).expanduser().resolve()),
        "source_rows": len(processed_rows),
        "prediction_rows": len(prediction_rows),
        "merged_rows": len(merged),
        "label_counts": label_counts,
        "split_counts": {split: len(merged)},
        "missing_merge_count": 0,
        "duplicate_molecule_count": 0,
        "smiles_mismatch_count": 0,
        "label_mismatch_count": 0,
    }
    return merged, output_fields, audit


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Mutagenicity Teacher-Consistent Views",
        "",
        "Processed split metadata is the authoritative parent table. RF prediction files are",
        "joined one-to-one by `molecule_id`; no row is silently dropped.",
        "",
        "## Direction",
        "",
        "- Source label: 1 (mutagenic)",
        "- Target label: 0 (non-mutagenic)",
        "",
        "## Split audit",
        "",
        "| Split | Processed | Merged | Source label 1 | Source teacher-correct | Target teacher-correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = summary["splits"][split]
        lines.append(
            f"| {split} | {values['source_rows']} | {values['merged_rows']} | "
            f"{values['source_label_rows']} | "
            f"{values['source_label_teacher_correct_rows']} | "
            f"{values['target_label_teacher_correct_rows']} |"
        )
    lines.extend(
        (
            "",
            "## Validation",
            "",
            "- one-to-one molecule ID merge: passed",
            "- SMILES equality: passed",
            "- label equality: passed",
            "- split identity: passed",
            "- expected source-label teacher-correct counts: passed",
            "",
        )
    )
    return "\n".join(lines)


def build_teacher_consistent_views(
    *,
    processed_root: str | Path,
    teacher_root: str | Path,
    output_dir: str | Path,
    config: TeacherViewConfig | None = None,
) -> dict[str, Any]:
    """Build all four metadata-complete teacher-consistent split views."""

    active = config or TeacherViewConfig()
    processed_root = Path(processed_root).expanduser().resolve()
    teacher_root = Path(teacher_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, Any]] = {}
    totals = {
        "source_rows": 0,
        "merged_rows": 0,
        "source_label_rows": 0,
        "source_label_teacher_correct_rows": 0,
        "target_label_teacher_correct_rows": 0,
    }
    for split in SPLITS:
        rows, fields, audit = merge_processed_and_predictions(
            processed_root / f"{split}.csv",
            teacher_root / f"predictions_{split}.csv",
            split=split,
            source_label=active.source_label,
            target_label=active.target_label,
        )
        source_rows = [row for row in rows if int(row["label"]) == active.source_label]
        source_correct = [
            row
            for row in source_rows
            if int(row["teacher_pred"]) == active.source_label
            and bool(row["teacher_correct"])
        ]
        target_correct = [
            row
            for row in rows
            if int(row["label"]) == active.target_label
            and int(row["teacher_pred"]) == active.target_label
            and bool(row["teacher_correct"])
        ]
        expected = active.expected_count(split)
        if expected is not None and len(source_correct) != expected:
            raise ValueError(
                f"{split} source-label teacher-correct count mismatch: "
                f"expected={expected} actual={len(source_correct)}"
            )

        outputs = {
            "source_label1_all": output_dir / f"{split}_source_label1_all.csv",
            "source_label1_teacher_correct": (
                output_dir / f"{split}_source_label1_teacher_correct.csv"
            ),
            "target_label0_teacher_correct": (
                output_dir / f"{split}_target_label0_teacher_correct.csv"
            ),
        }
        _atomic_write_csv(outputs["source_label1_all"], source_rows, fields)
        _atomic_write_csv(
            outputs["source_label1_teacher_correct"], source_correct, fields
        )
        _atomic_write_csv(outputs["target_label0_teacher_correct"], target_correct, fields)

        audit.update(
            {
                "source_label_rows": len(source_rows),
                "source_label_teacher_correct_rows": len(source_correct),
                "target_label_teacher_correct_rows": len(target_correct),
                "expected_source_label_teacher_correct_rows": expected,
                "outputs": {key: str(path) for key, path in outputs.items()},
            }
        )
        split_summaries[split] = audit
        for field in totals:
            totals[field] += int(audit[field])

    summary: dict[str, Any] = {
        "dataset": "Mutagenicity",
        "dataset_version": "v1",
        "processed_root": str(processed_root),
        "teacher_root": str(teacher_root),
        "output_dir": str(output_dir),
        "source_label": int(active.source_label),
        "target_label": int(active.target_label),
        "merge_key": "molecule_id",
        "merge_validation": "one_to_one_strict",
        "splits": split_summaries,
        **totals,
        "label_counts": {
            label: sum(values["label_counts"][label] for values in split_summaries.values())
            for label in ("0", "1")
        },
        "split_counts": {
            split: values["merged_rows"] for split, values in split_summaries.items()
        },
        "missing_merge_count": 0,
        "duplicate_molecule_count": 0,
        "smiles_mismatch_count": 0,
        "label_mismatch_count": 0,
        "build_passed": True,
    }
    _atomic_write_text(
        output_dir / "teacher_consistent_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_dir / "teacher_consistent_report.md", _report(summary))
    return summary


__all__ = [
    "DEFAULT_EXPECTED_SOURCE_CORRECT_COUNTS",
    "PREDICTION_REQUIRED_FIELDS",
    "PROCESSED_REQUIRED_FIELDS",
    "SOURCE_LABEL",
    "SPLITS",
    "TARGET_LABEL",
    "TEACHER_FIELDS",
    "TeacherViewConfig",
    "build_teacher_consistent_views",
    "merge_processed_and_predictions",
]
