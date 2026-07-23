"""Data and audit utilities for Mutagenicity continued SFT."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.hiv_dataset_utils import parent_atom_count_bin, sample_records_by_strata


SOURCE_LABEL = 1
TARGET_LABEL = 0
EXPECTED_TRAIN_ROWS = 1317
EXPECTED_VAL_ROWS = 250
DEFAULT_MAX_SEQUENCE_LENGTH = 1024
REQUIRED_FIELDS = (
    "molecule_id",
    "parent_smiles",
    "label",
    "source_label",
    "target_label",
    "semantic_label",
    "split",
    "scaffold_smiles",
    "teacher_pred",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_correct",
    "prompt",
    "completion",
)
GENERATION_FIELDS = (
    "molecule_id",
    "parent_smiles",
    "prompt",
    "reference_completion",
    "generated_text",
    "decoded_fragment",
    "parse_ok",
    "contains_dummy",
    "empty_output",
    "exact_match",
    "generation_length",
)


@dataclass(frozen=True, slots=True)
class ContinuedSFTRecord:
    molecule_id: str
    parent_smiles: str
    canonical_parent_smiles: str
    label: int
    source_label: int
    target_label: int
    semantic_label: str
    split: str
    scaffold_smiles: str
    teacher_pred: int
    teacher_prob_0: float
    teacher_prob_1: float
    teacher_correct: bool
    prompt: str
    completion: str
    parent_atom_count: int
    source_row_index: int

    @property
    def stratum_key(self) -> str:
        scaffold = self.scaffold_smiles or "ACYCLIC"
        return f"{scaffold}::{parent_atom_count_bin(self.parent_atom_count)}"


@dataclass(frozen=True, slots=True)
class TokenizedSFTExample:
    molecule_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    num_prompt_tokens: int
    num_completion_tokens: int
    num_total_tokens: int
    num_supervised_tokens: int
    truncated: bool
    completion_fully_truncated: bool

    def as_dataset_row(self) -> dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
        }


class SupervisedTokenDataset:
    """Minimal torch-compatible dataset without a torch import at module load."""

    def __init__(
        self,
        examples: Sequence[TokenizedSFTExample],
        *,
        track_coverage: bool = False,
    ) -> None:
        self.examples = tuple(examples)
        self.track_coverage = bool(track_coverage)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.examples[index].as_dataset_row()
        row["_track_coverage"] = self.track_coverage
        return row


class ParentCoverageTracker:
    """Track parent examples materialized by the training collator."""

    def __init__(self, all_parent_ids: Sequence[str]) -> None:
        self.all_parent_ids = frozenset(str(value) for value in all_parent_ids)
        self.num_examples_seen = 0
        self.unique_parent_ids_seen: set[str] = set()

    def record(self, molecule_ids: Sequence[str]) -> None:
        normalized = [str(value) for value in molecule_ids]
        unknown = sorted(set(normalized) - self.all_parent_ids)
        if unknown:
            raise ValueError(f"Coverage tracker received unknown molecule IDs: {unknown[:10]}")
        self.num_examples_seen += len(normalized)
        self.unique_parent_ids_seen.update(normalized)

    def summary(
        self,
        *,
        global_step: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        world_size: int = 1,
        current_epoch: float | None = None,
    ) -> dict[str, Any]:
        num_rows = len(self.all_parent_ids)
        effective_batch_size = (
            int(per_device_batch_size)
            * int(gradient_accumulation_steps)
            * int(world_size)
        )
        return {
            "num_dataset_rows": num_rows,
            "num_unique_parents": num_rows,
            "num_train_examples_seen": int(self.num_examples_seen),
            "num_unique_train_parents_seen": len(self.unique_parent_ids_seen),
            "unique_train_parent_coverage": (
                len(self.unique_parent_ids_seen) / num_rows if num_rows else 0.0
            ),
            "epochs_equivalent": self.num_examples_seen / num_rows if num_rows else 0.0,
            "global_step": int(global_step),
            "samples_per_step": effective_batch_size,
            "effective_batch_size": effective_batch_size,
            "current_epoch": current_epoch,
        }


class CompletionOnlyDataCollator:
    """Right-pad tokenized rows while preserving completion-only labels."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        coverage_tracker: ParentCoverageTracker | None = None,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.coverage_tracker = coverage_tracker

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - HPC runtime dependency
            raise RuntimeError("CompletionOnlyDataCollator requires PyTorch") from exc

        if not features:
            raise ValueError("Cannot collate an empty SFT batch")
        molecule_ids = [
            str(feature["molecule_id"])
            for feature in features
            if bool(feature.get("_track_coverage"))
        ]
        if self.coverage_tracker is not None:
            self.coverage_tracker.record(molecule_ids)
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            ids = [int(value) for value in feature["input_ids"]]
            mask = [int(value) for value in feature["attention_mask"]]
            target = [int(value) for value in feature["labels"]]
            if not (len(ids) == len(mask) == len(target)):
                raise ValueError("Tokenized SFT row has inconsistent sequence lengths")
            pad_count = max_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_count)
            attention_masks.append(mask + [0] * pad_count)
            labels.append(target + [-100] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _binary_int(value: Any, *, field: str, row_index: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}") from exc
    if parsed not in (0, 1):
        raise ValueError(f"row={row_index} has non-binary {field}={parsed}")
    return parsed


def _boolean(value: Any, *, field: str, row_index: int) -> bool:
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"row={row_index} has invalid {field}={value!r}")


def _probability(value: Any, *, field: str, row_index: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row={row_index} has invalid {field}={value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"row={row_index} has out-of-range {field}={value!r}")
    return parsed


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.stat().st_size <= 0:
        raise FileNotFoundError(f"Mutagenicity SFT CSV is missing or empty: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(set(REQUIRED_FIELDS) - set(fieldnames))
        if missing:
            raise ValueError(f"{target} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Mutagenicity SFT CSV has no rows: {target}")
    return rows, fieldnames


def load_continued_sft_records(
    path: str | Path,
    *,
    expected_split: str,
    expected_count: int | None,
) -> list[ContinuedSFTRecord]:
    """Load one fixed train/val CSV and enforce the source-parent contract."""

    if expected_split not in {"train", "val"}:
        raise ValueError(
            "Continued SFT accepts only train or val data; "
            f"received expected_split={expected_split!r}"
        )
    rows, _fieldnames = _read_csv(path)
    if expected_count is not None and int(expected_count) > 0:
        if len(rows) != int(expected_count):
            raise ValueError(
                f"{expected_split} row count mismatch: "
                f"expected={expected_count} actual={len(rows)}"
            )

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - HPC/local environment contract
        raise RuntimeError("Mutagenicity continued SFT validation requires RDKit") from exc

    records: list[ContinuedSFTRecord] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    for row_index, row in enumerate(rows):
        molecule_id = str(row.get("molecule_id") or "").strip()
        if not molecule_id:
            raise ValueError(f"row={row_index} has an empty molecule_id")
        if molecule_id in seen_ids:
            raise ValueError(f"duplicate molecule_id in {path}: {molecule_id}")
        parent_smiles = str(row.get("parent_smiles") or "").strip()
        molecule = Chem.MolFromSmiles(parent_smiles)
        if molecule is None or molecule.GetNumAtoms() <= 0:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has invalid parent_smiles"
            )
        canonical_smiles = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        if canonical_smiles in seen_smiles:
            raise ValueError(f"duplicate canonical parent SMILES in {path}: {canonical_smiles}")

        split = str(row.get("split") or "").strip().lower()
        if split != expected_split:
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has split={split!r}; "
                f"expected {expected_split!r}. Calibration/test inputs are forbidden."
            )
        label = _binary_int(row.get("label"), field="label", row_index=row_index)
        source_label = _binary_int(
            row.get("source_label"), field="source_label", row_index=row_index
        )
        target_label = _binary_int(
            row.get("target_label"), field="target_label", row_index=row_index
        )
        teacher_pred = _binary_int(
            row.get("teacher_pred"), field="teacher_pred", row_index=row_index
        )
        teacher_correct = _boolean(
            row.get("teacher_correct"), field="teacher_correct", row_index=row_index
        )
        if (
            label != SOURCE_LABEL
            or source_label != SOURCE_LABEL
            or target_label != TARGET_LABEL
            or teacher_pred != SOURCE_LABEL
            or not teacher_correct
        ):
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} violates the "
                "label-1 teacher-correct source contract"
            )
        semantic_label = str(row.get("semantic_label") or "").strip().lower()
        if semantic_label != "mutagenic":
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has "
                f"semantic_label={semantic_label!r}"
            )
        probability_0 = _probability(
            row.get("teacher_prob_0"), field="teacher_prob_0", row_index=row_index
        )
        probability_1 = _probability(
            row.get("teacher_prob_1"), field="teacher_prob_1", row_index=row_index
        )
        if not math.isclose(probability_0 + probability_1, 1.0, abs_tol=1e-8):
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} teacher probabilities "
                "do not sum to one"
            )
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if not prompt.strip():
            raise ValueError(f"row={row_index} molecule_id={molecule_id} has empty prompt")
        if not completion.strip():
            raise ValueError(
                f"row={row_index} molecule_id={molecule_id} has empty completion"
            )
        scaffold = str(row.get("scaffold_smiles") or "").strip()
        records.append(
            ContinuedSFTRecord(
                molecule_id=molecule_id,
                parent_smiles=parent_smiles,
                canonical_parent_smiles=canonical_smiles,
                label=label,
                source_label=source_label,
                target_label=target_label,
                semantic_label=semantic_label,
                split=split,
                scaffold_smiles=scaffold,
                teacher_pred=teacher_pred,
                teacher_prob_0=probability_0,
                teacher_prob_1=probability_1,
                teacher_correct=teacher_correct,
                prompt=prompt,
                completion=completion,
                parent_atom_count=int(molecule.GetNumAtoms()),
                source_row_index=row_index,
            )
        )
        seen_ids.add(molecule_id)
        seen_smiles.add(canonical_smiles)
    return records


def validate_train_val_isolation(
    train: Sequence[ContinuedSFTRecord],
    val: Sequence[ContinuedSFTRecord],
) -> dict[str, Any]:
    """Reject parent, molecule, or scaffold leakage across train and val."""

    train_ids = {row.molecule_id for row in train}
    val_ids = {row.molecule_id for row in val}
    train_smiles = {row.canonical_parent_smiles for row in train}
    val_smiles = {row.canonical_parent_smiles for row in val}
    train_scaffolds = {row.scaffold_smiles for row in train}
    val_scaffolds = {row.scaffold_smiles for row in val}
    audit = {
        "molecule_id_overlap": sorted(train_ids & val_ids),
        "canonical_parent_smiles_overlap": sorted(train_smiles & val_smiles),
        "scaffold_overlap": sorted(train_scaffolds & val_scaffolds),
    }
    if any(audit.values()):
        raise ValueError(f"Mutagenicity continued-SFT train/val leakage: {audit}")
    audit["passed"] = True
    return audit


def deterministic_smoke_sample(
    records: Sequence[ContinuedSFTRecord],
    *,
    max_rows: int,
    seed: int,
) -> list[ContinuedSFTRecord]:
    """Use the existing scaffold/size round-robin sampler, never file head."""

    if max_rows <= 0 or max_rows >= len(records):
        return list(records)
    sample_size = int(max_rows)
    return list(
        sample_records_by_strata(
            list(records),
            sample_size=sample_size,
            seed=int(seed),
            key_fn=lambda record: record.stratum_key,
        )
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]


def tokenize_completion_only(
    tokenizer: Any,
    record: ContinuedSFTRecord,
    *,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> TokenizedSFTExample:
    """Tokenize prompt+completion and supervise completion/EOS tokens only."""

    if int(max_sequence_length) <= 1:
        raise ValueError("max_sequence_length must be greater than one")
    prompt_ids = _token_ids(tokenizer, record.prompt)
    completion_content_ids = _token_ids(tokenizer, record.completion)
    if not completion_content_ids:
        raise ValueError(
            f"molecule_id={record.molecule_id} completion produced zero tokens"
        )
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for continued SFT")
    prefix_ids = ([int(bos_token_id)] if bos_token_id is not None else []) + prompt_ids
    completion_ids = completion_content_ids + [int(eos_token_id)]
    untruncated_length = len(prefix_ids) + len(completion_ids)
    available_completion = int(max_sequence_length) - len(prefix_ids)
    if available_completion <= 0:
        raise ValueError(
            f"molecule_id={record.molecule_id} completion is fully truncated: "
            f"prompt_tokens={len(prefix_ids)} max_sequence_length={max_sequence_length}"
        )
    retained_completion = completion_ids[:available_completion]
    retained_content_count = min(len(completion_content_ids), available_completion)
    if retained_content_count <= 0:
        raise ValueError(
            f"molecule_id={record.molecule_id} completion is fully truncated"
        )
    input_ids = prefix_ids + retained_completion
    labels = [-100] * len(prefix_ids) + retained_completion
    num_supervised = sum(value != -100 for value in labels)
    if num_supervised <= 0:
        raise ValueError(
            f"molecule_id={record.molecule_id} has no supervised completion tokens"
        )
    return TokenizedSFTExample(
        molecule_id=record.molecule_id,
        input_ids=tuple(input_ids),
        attention_mask=tuple([1] * len(input_ids)),
        labels=tuple(labels),
        num_prompt_tokens=len(prefix_ids),
        num_completion_tokens=len(retained_completion),
        num_total_tokens=len(input_ids),
        num_supervised_tokens=num_supervised,
        truncated=untruncated_length > int(max_sequence_length),
        completion_fully_truncated=False,
    )


def tokenize_records(
    tokenizer: Any,
    records: Sequence[ContinuedSFTRecord],
    *,
    max_sequence_length: int,
) -> tuple[list[TokenizedSFTExample], dict[str, Any]]:
    examples = [
        tokenize_completion_only(
            tokenizer, record, max_sequence_length=max_sequence_length
        )
        for record in records
    ]
    if not examples:
        raise ValueError("No records were available for tokenization")
    if any(example.num_supervised_tokens <= 0 for example in examples):
        raise ValueError("At least one SFT row has all labels masked to -100")
    audit = {
        "num_examples": len(examples),
        "max_sequence_length": int(max_sequence_length),
        "num_truncated": sum(example.truncated for example in examples),
        "num_completion_fully_truncated": sum(
            example.completion_fully_truncated for example in examples
        ),
        "min_prompt_tokens": min(example.num_prompt_tokens for example in examples),
        "max_prompt_tokens": max(example.num_prompt_tokens for example in examples),
        "min_completion_tokens": min(
            example.num_completion_tokens for example in examples
        ),
        "max_completion_tokens": max(
            example.num_completion_tokens for example in examples
        ),
        "min_total_tokens": min(example.num_total_tokens for example in examples),
        "max_total_tokens": max(example.num_total_tokens for example in examples),
        "total_supervised_tokens": sum(
            example.num_supervised_tokens for example in examples
        ),
        "all_rows_have_supervised_completion": True,
        "masking_semantics": "prompt=-100; completion_and_retained_eos=token_id",
        "sample_rows": [
            {
                "molecule_id": example.molecule_id,
                "num_prompt_tokens": example.num_prompt_tokens,
                "num_completion_tokens": example.num_completion_tokens,
                "num_total_tokens": example.num_total_tokens,
                "num_supervised_tokens": example.num_supervised_tokens,
                "truncated": example.truncated,
                "completion_fully_truncated": example.completion_fully_truncated,
            }
            for example in examples[:20]
        ],
    }
    return examples, audit


def validate_peft_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"BASE checkpoint directory does not exist: {checkpoint}")
    config_path = checkpoint / "adapter_config.json"
    weight_candidates = (
        checkpoint / "adapter_model.safetensors",
        checkpoint / "adapter_model.bin",
    )
    if not config_path.is_file():
        raise ValueError(
            f"BASE checkpoint is not a PEFT adapter; missing {config_path.name}: {checkpoint}"
        )
    weight_path = next((candidate for candidate in weight_candidates if candidate.is_file()), None)
    if weight_path is None:
        raise ValueError(
            "BASE checkpoint is missing adapter_model.safetensors/adapter_model.bin: "
            f"{checkpoint}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    peft_type = str(payload.get("peft_type") or "").upper()
    task_type = str(payload.get("task_type") or "").upper()
    if peft_type and peft_type != "LORA":
        raise ValueError(f"Unsupported PEFT checkpoint type: {peft_type}")
    if task_type and task_type != "CAUSAL_LM":
        raise ValueError(f"Unsupported PEFT task type: {task_type}")
    return {
        "checkpoint": str(checkpoint),
        "adapter_config": str(config_path),
        "adapter_weights": str(weight_path),
        "adapter_weight_bytes": weight_path.stat().st_size,
        "peft_type": peft_type or None,
        "task_type": task_type or None,
        "base_model_name_or_path": payload.get("base_model_name_or_path"),
        "inference_mode": payload.get("inference_mode"),
    }


def ensure_new_output_root(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output root is non-empty and will not be overwritten: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.split("checkpoint-", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def build_checkpoint_manifest(
    output_root: str | Path,
    *,
    initialization_checkpoint: str | Path,
    best_checkpoint: str | Path | None,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    initialization = validate_peft_checkpoint(initialization_checkpoint)
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(
        (candidate for candidate in root.glob("checkpoint-*") if candidate.is_dir()),
        key=lambda candidate: (_checkpoint_step(candidate) is None, _checkpoint_step(candidate) or 0),
    ):
        checkpoint = validate_peft_checkpoint(path)
        checkpoint["step"] = _checkpoint_step(path)
        checkpoints.append(checkpoint)
    return {
        "output_root": str(root),
        "initialization_checkpoint": initialization,
        "num_training_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
        "best_checkpoint": (
            str(Path(best_checkpoint).expanduser().resolve())
            if best_checkpoint is not None
            else None
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(
    *,
    train_path: str | Path,
    val_path: str | Path,
    train_all: Sequence[ContinuedSFTRecord],
    val_all: Sequence[ContinuedSFTRecord],
    train_selected: Sequence[ContinuedSFTRecord],
    val_selected: Sequence[ContinuedSFTRecord],
    isolation_audit: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset": "Mutagenicity",
        "dataset_version": "v1",
        "source_label": SOURCE_LABEL,
        "target_label": TARGET_LABEL,
        "training_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "forbidden_splits": ["calibration", "test"],
        "train_csv": str(Path(train_path).expanduser().resolve()),
        "val_csv": str(Path(val_path).expanduser().resolve()),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "num_train_rows_full": len(train_all),
        "num_val_rows_full": len(val_all),
        "num_train_rows_selected": len(train_selected),
        "num_val_rows_selected": len(val_selected),
        "num_unique_train_parents": len({row.molecule_id for row in train_selected}),
        "num_unique_val_parents": len({row.molecule_id for row in val_selected}),
        "selection_seed": int(seed),
        "sampling": (
            "full"
            if len(train_selected) == len(train_all) and len(val_selected) == len(val_all)
            else "deterministic_scaffold_size_stratified"
        ),
        "isolation_audit": dict(isolation_audit),
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv_atomic(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def score_generated_fragment(
    record: ContinuedSFTRecord,
    *,
    generated_text: str,
    generation_length: int,
) -> dict[str, Any]:
    """Normalize one generated fragment and compute lightweight sanity fields."""

    from src.models.llm_generator import clean_generated_smiles

    decoded = clean_generated_smiles(str(generated_text or "")).strip()
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - HPC/local environment contract
        raise RuntimeError("Generation scoring requires RDKit") from exc
    molecule = Chem.MolFromSmiles(decoded) if decoded else None
    reference = record.completion.strip()
    return {
        "molecule_id": record.molecule_id,
        "parent_smiles": record.parent_smiles,
        "prompt": record.prompt,
        "reference_completion": reference,
        "generated_text": str(generated_text or "").strip(),
        "decoded_fragment": decoded,
        "parse_ok": molecule is not None,
        "contains_dummy": "*" in decoded,
        "empty_output": not bool(decoded),
        "exact_match": decoded == reference,
        "generation_length": int(generation_length),
    }


__all__ = [
    "CompletionOnlyDataCollator",
    "ContinuedSFTRecord",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "EXPECTED_TRAIN_ROWS",
    "EXPECTED_VAL_ROWS",
    "GENERATION_FIELDS",
    "ParentCoverageTracker",
    "SOURCE_LABEL",
    "SupervisedTokenDataset",
    "TARGET_LABEL",
    "TokenizedSFTExample",
    "build_checkpoint_manifest",
    "dataset_manifest",
    "deterministic_smoke_sample",
    "ensure_new_output_root",
    "load_continued_sft_records",
    "score_generated_fragment",
    "sha256_file",
    "tokenize_completion_only",
    "tokenize_records",
    "validate_peft_checkpoint",
    "validate_train_val_isolation",
    "write_csv_atomic",
    "write_json_atomic",
]
