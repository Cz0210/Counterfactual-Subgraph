"""Executable TasteMolNet T13 native GlobalGCE full experiment.

This is deliberately a dataset-specific runner.  It reuses the audited
official GlobalGCE training bridge and native LHS-to-RHS action engine, but it
owns the full-experiment boundary that the bounded T8 smoke intentionally does
not provide:

* both Sweet->Bitter and Sweet->Tasteless branches are trained on train only;
* official epoch checkpoints and a small stage checkpoint make restart real;
* calibration orders at most twenty canonical native rules;
* the held-out test file is not opened until that order is durably frozen;
* the frozen three-class GINE and MolCLR-Node-Wasserstein evaluate test pairs;
* a separate verifier invocation publishes the terminal PASS.

The module does not introduce a reusable execution framework.  Its state,
schemas, selection policy, and exports are specific to TasteMolNet T13.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    apply_rule_to_parent,
    validate_official_globalgce_root,
)
from src.baselines.globalgce_mutagenicity_adapter import (
    FROZEN_GINE_IN_MEMORY_FILES,
    NativeGenerationResult,
    OfficialGlobalGCEMutagenicityGenerator,
    TrainParent,
)
from src.baselines.tastemolnet_globalgce_smoke import FrozenTasteGINEScorer
from src.data.tastemolnet_ppo import LABEL_MAP, TASTEMOLNET_PREPARED_FIELDS
from src.eval.four_by_four_registry import (
    PASS_STATUSES,
    audit_explicit_candidate,
    stable_json_sha256,
)
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)


STAGE = "T13_GLOBALGCE_FULL"
DATASET = "TasteMolNet"
DATASET_ID = "tastemolnet"
METHOD = "GlobalGCE"
SOURCE_LABEL = 1
TARGET_BRANCHES = (0, 2)
DESTINATION_LABELS = (0, 2)
NUM_CLASSES = 3
SEED = 7
K_MAX = 20
MIN_RULES = 10
TABLE2_K = 10
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
CF_MODE = "strict_flip"
DISTANCE_NAMESPACE = "tastemolnet_globalgce_full_wnode_v1"
CHECKPOINT_SCHEMA = "tastemolnet_t13_checkpoint_v1"
BRANCH_MANIFEST_SCHEMA = "tastemolnet_t13_branch_manifest_v1"
SELECTION_SCHEMA = "tastemolnet_t13_calibration_selection_v1"
RUN_MANIFEST_SCHEMA = "tastemolnet_t13_run_manifest_v1"
VERIFY_SCHEMA = "tastemolnet_t13_terminal_verification_v1"
PASS_MARKER = "[TASTE_T13_GLOBALGCE_PASS]"
GINE_FILES = tuple(sorted(FROZEN_GINE_IN_MEMORY_FILES))


class TasteGlobalGCEFullError(RuntimeError):
    """T13 failed a scientific, split, resume, or output-completeness gate."""


@dataclass(frozen=True, slots=True)
class TasteGlobalGCEFullConfig:
    """The fixed main-experiment knobs, all persisted in the checkpoint."""

    epochs: int = 100
    top_k_native: int = K_MAX
    min_freq: int = 2
    learning_rate: float = 0.1
    dropout: float = 0.5
    generation_chunk_size: int = 32
    oracle_batch_size: int = 256
    gspan_flush_every: int = 256
    gspan_max_in_memory_candidates: int = 256
    seed: int = SEED

    def validate(self) -> None:
        for name, value in {
            "epochs": self.epochs,
            "top_k_native": self.top_k_native,
            "min_freq": self.min_freq,
            "generation_chunk_size": self.generation_chunk_size,
            "oracle_batch_size": self.oracle_batch_size,
            "gspan_flush_every": self.gspan_flush_every,
            "gspan_max_in_memory_candidates": self.gspan_max_in_memory_candidates,
        }.items():
            if type(value) is not int or value <= 0:
                raise TasteGlobalGCEFullError(f"T13 {name} must be a positive integer")
        if self.epochs < 25:
            raise TasteGlobalGCEFullError("T13 full training requires at least 25 epochs")
        if self.top_k_native != K_MAX:
            raise TasteGlobalGCEFullError("T13 native rule surface is fixed to K_MAX=20")
        if self.min_freq < 2:
            raise TasteGlobalGCEFullError("T13 min_freq must be at least two")
        for name, value in {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
        }.items():
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise TasteGlobalGCEFullError(f"T13 {name} must be finite and positive")
        if not 0.0 < self.dropout < 1.0:
            raise TasteGlobalGCEFullError("T13 dropout must be inside (0,1)")
        if type(self.seed) is not int or self.seed != SEED:
            raise TasteGlobalGCEFullError("T13 is frozen to seed 7")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "epochs": self.epochs,
            "top_k_native": self.top_k_native,
            "min_freq": self.min_freq,
            "learning_rate_hex": self.learning_rate.hex(),
            "dropout_hex": self.dropout.hex(),
            "generation_chunk_size": self.generation_chunk_size,
            "oracle_batch_size": self.oracle_batch_size,
            "gspan_flush_every": self.gspan_flush_every,
            "gspan_max_in_memory_candidates": self.gspan_max_in_memory_candidates,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ThresholdContract:
    values: tuple[float, ...]
    theta_star: float
    cost_cap: float
    config_hash: str
    source: str
    source_split: str
    file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": list(self.values),
            "theta_star": self.theta_star,
            "cost_cap": self.cost_cap,
            "threshold_config_hash": self.config_hash,
            "threshold_source": self.source,
            "threshold_source_split": self.source_split,
            "threshold_contract_file_sha256": self.file_sha256,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        }


@dataclass(frozen=True, slots=True)
class InputAuthority:
    train_path: Path
    calibration_path: Path
    test_path: Path
    checkpoint_path: Path
    official_root: Path
    molclr_root: Path
    molclr_checkpoint: Path
    t8_pass_root: Path
    threshold_path: Path
    train_sha256: str
    calibration_sha256: str
    declared_test_sha256: str
    checkpoint_id: str
    dataset_hash: str
    split_manifest_sha256: str
    molclr_checkpoint_sha256: str
    t8_pass_sha256: str
    threshold: ThresholdContract
    train_count: int
    train_label_counts: dict[str, int]

    def resume_identity(self, config: TasteGlobalGCEFullConfig) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t13_resume_identity_v1",
            "dataset": DATASET,
            "method": METHOD,
            "stage": STAGE,
            "config": config.to_dict(),
            "train_sha256": self.train_sha256,
            "calibration_sha256": self.calibration_sha256,
            # The test file is deliberately not opened here.  This value is the
            # T3 split-manifest declaration and is checked only after freeze.
            "declared_test_sha256": self.declared_test_sha256,
            "checkpoint_id": self.checkpoint_id,
            "dataset_hash": self.dataset_hash,
            "split_manifest_sha256": self.split_manifest_sha256,
            "molclr_checkpoint_sha256": self.molclr_checkpoint_sha256,
            "t8_pass_sha256": self.t8_pass_sha256,
            "threshold_config_hash": self.threshold.config_hash,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ]
    _atomic_bytes(path, (("\n".join(lines) + "\n") if lines else "").encode("utf-8"))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise TasteGlobalGCEFullError(f"cannot write empty CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCEFullError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise TasteGlobalGCEFullError(f"JSON document must be an object: {path}")
    return value


def read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCEFullError(f"invalid JSON: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TasteGlobalGCEFullError(f"cannot read JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TasteGlobalGCEFullError(
                f"invalid JSONL row {line_number}: {path}"
            ) from exc
        if type(value) is not dict:
            raise TasteGlobalGCEFullError(f"JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_threshold_contract(path_like: str | Path) -> ThresholdContract:
    path = Path(path_like).expanduser().resolve(strict=True)
    payload = read_json(path)
    if str(payload.get("dataset") or "").strip().lower() not in {
        "taste",
        "tastemolnet",
    }:
        raise TasteGlobalGCEFullError("threshold contract is not for TasteMolNet")
    raw_values = payload.get("thresholds")
    if not isinstance(raw_values, list) or not raw_values:
        raise TasteGlobalGCEFullError("threshold contract lacks a nonempty grid")
    try:
        values = tuple(float(value) for value in raw_values)
        theta_star = float(payload["theta_star"])
        cost_cap = float(payload["cost_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TasteGlobalGCEFullError("threshold contract contains invalid numbers") from exc
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or any(right <= left for left, right in zip(values, values[1:]))
        or not math.isfinite(theta_star)
        or theta_star < 0.0
        or theta_star not in values
        or not math.isfinite(cost_cap)
        or cost_cap < theta_star
    ):
        raise TasteGlobalGCEFullError("threshold contract numeric gate failed")
    source_split = str(
        payload.get("threshold_source_split") or payload.get("selection_split") or ""
    ).strip().lower()
    if source_split not in {
        "calibration",
        "frozen_calibration",
        "frozen_protocol",
        "existing_frozen_protocol",
        "legacy_frozen_protocol",
    }:
        raise TasteGlobalGCEFullError("threshold source must be calibration/frozen protocol")
    if payload.get("test_used_for_selection") is not False:
        raise TasteGlobalGCEFullError("threshold contract does not exclude test selection")
    expected_hash = stable_json_sha256(list(values))
    claimed_hash = str(payload.get("threshold_config_hash") or "").lower()
    if claimed_hash != expected_hash:
        raise TasteGlobalGCEFullError("threshold_config_hash differs from its numeric grid")
    source = str(payload.get("threshold_source") or "").strip()
    if not source:
        raise TasteGlobalGCEFullError("threshold contract lacks its frozen source")
    return ThresholdContract(
        values=values,
        theta_star=theta_star,
        cost_cap=cost_cap,
        config_hash=expected_hash,
        source=source,
        source_split=source_split,
        file_sha256=sha256_file(path),
    )


def _find_t8_pass_document(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        "gate.json",
        "verification.json",
        "managed_verification.json",
        "run_manifest.json",
        "manifest.json",
    )
    observed: list[tuple[Path, dict[str, Any]]] = []
    for name in candidates:
        path = root / name
        if path.is_file():
            observed.append((path, read_json(path)))
    for path, payload in observed:
        stage_values = {
            str(payload.get(key) or "").strip().upper()
            for key in ("stage", "task_id", "managed_task_id")
        }
        pass_values = {
            str(payload.get(key) or "").strip().upper()
            for key in ("status", "state", "gate_state")
        }
        explicit_pass = any(
            payload.get(key) is True
            for key in ("passed", "gate_passed", "verification_passed")
        )
        if (
            any("T8" in value or "GLOBALGCE_SMOKE" in value for value in stage_values)
            and (explicit_pass or bool(pass_values & {"PASS", "PASSED"}))
        ):
            return path, payload
    raise TasteGlobalGCEFullError("T13 prerequisite has no typed T8 PASS document")


def validate_t8_pass(path_like: str | Path) -> tuple[Path, str]:
    requested = Path(path_like).expanduser().resolve(strict=True)
    if requested.is_file():
        payload = read_json(requested)
        root = requested.parent
        pass_path = requested
        stage = " ".join(
            str(payload.get(key) or "") for key in ("stage", "task_id", "managed_task_id")
        ).upper()
        state = str(payload.get("status") or payload.get("state") or "").upper()
        if not (
            ("T8" in stage or "GLOBALGCE_SMOKE" in stage)
            and (
                state in {"PASS", "PASSED"}
                or payload.get("passed") is True
                or payload.get("gate_passed") is True
            )
        ):
            raise TasteGlobalGCEFullError("T13 prerequisite receipt is not T8 PASS")
    elif requested.is_dir():
        root = requested
        marker = root / "PASS"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "PASS":
            raise TasteGlobalGCEFullError("T13 prerequisite root lacks exact PASS marker")
        pass_path, _payload = _find_t8_pass_document(root)
    else:  # pragma: no cover - resolve(strict=True) already excludes this.
        raise TasteGlobalGCEFullError("T8 prerequisite is not a file/directory")
    return root, sha256_file(pass_path)


def _checkpoint_payloads(checkpoint: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in GINE_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise TasteGlobalGCEFullError(f"frozen GINE is missing {name}")
        payloads[name] = path.read_bytes()
    return payloads


def load_input_authority(
    *,
    train_csv: str | Path,
    calibration_csv: str | Path,
    test_csv: str | Path,
    gnn_checkpoint: str | Path,
    official_root: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    t8_pass_root: str | Path,
    threshold_contract: str | Path,
) -> InputAuthority:
    """Validate all pre-test identities without opening the held-out test file."""

    train = Path(train_csv).expanduser().resolve(strict=True)
    calibration = Path(calibration_csv).expanduser().resolve(strict=True)
    # Lexical absolute resolution does not open test bytes.  Existence/hash is
    # checked only by ``authorize_and_load_test_after_freeze``.
    test = Path(test_csv).expanduser().absolute()
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    official = Path(official_root).expanduser().resolve(strict=True)
    molclr_source = Path(molclr_root).expanduser().resolve(strict=True)
    molclr_ckpt = Path(molclr_checkpoint).expanduser().resolve(strict=True)
    threshold_path = Path(threshold_contract).expanduser().resolve(strict=True)
    threshold = load_threshold_contract(threshold_path)
    t8_root, t8_sha = validate_t8_pass(t8_pass_root)
    validate_official_globalgce_root(official)
    payloads = _checkpoint_payloads(checkpoint)
    card = json.loads(payloads["model_card.json"].decode("utf-8"))
    if (
        card.get("dataset") not in {DATASET, DATASET_ID}
        or card.get("oracle_backend") != "gnn"
        or card.get("rf_oracle_used") is not False
        or str(card.get("backbone") or "").lower() != "gine"
        or card.get("num_classes") != NUM_CLASSES
        or card.get("source_label") != SOURCE_LABEL
    ):
        raise TasteGlobalGCEFullError("frozen GINE model card is not TasteMolNet GINE")
    checkpoint_id = sha256_bytes(payloads["model.pt"])
    if card.get("checkpoint_id") != checkpoint_id:
        raise TasteGlobalGCEFullError("frozen GINE checkpoint_id differs from model.pt")
    split = json.loads(payloads["split_manifest.json"].decode("utf-8"))
    files = split.get("files")
    roles = split.get("roles")
    if (
        split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") not in {DATASET, DATASET_ID}
        or not isinstance(files, dict)
        or set(files) != {"train", "validation", "calibration", "test"}
        or not isinstance(roles, dict)
        or roles.get("calibration") != "reserved_for_threshold_and_selector_only"
        or roles.get("test") != "frozen_model_final_quality_evaluation"
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGlobalGCEFullError("frozen GINE split-role contract changed")
    declared: dict[str, str] = {}
    for role in ("train", "calibration", "test"):
        row = files.get(role)
        if not isinstance(row, dict) or not _is_sha256(row.get("sha256")):
            raise TasteGlobalGCEFullError(f"split manifest lacks {role} SHA-256")
        declared[role] = str(row["sha256"]).lower()
    train_sha = sha256_file(train)
    calibration_sha = sha256_file(calibration)
    if train_sha != declared["train"] or calibration_sha != declared["calibration"]:
        raise TasteGlobalGCEFullError("train/calibration bytes differ from split manifest")
    manifest = split.get("train_manifest")
    if not isinstance(manifest, dict):
        raise TasteGlobalGCEFullError("split manifest lacks train manifest")
    train_count = manifest.get("num_records")
    train_counts = manifest.get("label_counts")
    dataset_hash = str(manifest.get("dataset_fingerprint") or "").lower()
    if (
        type(train_count) is not int
        or train_count <= 0
        or not isinstance(train_counts, dict)
        or set(train_counts) != {"0", "1", "2"}
        or any(type(value) is not int or value <= 0 for value in train_counts.values())
        or sum(train_counts.values()) != train_count
        or not _is_sha256(dataset_hash)
    ):
        raise TasteGlobalGCEFullError("frozen train manifest count/fingerprint changed")
    return InputAuthority(
        train_path=train,
        calibration_path=calibration,
        test_path=test,
        checkpoint_path=checkpoint,
        official_root=official,
        molclr_root=molclr_source,
        molclr_checkpoint=molclr_ckpt,
        t8_pass_root=t8_root,
        threshold_path=threshold_path,
        train_sha256=train_sha,
        calibration_sha256=calibration_sha,
        declared_test_sha256=declared["test"],
        checkpoint_id=checkpoint_id,
        dataset_hash=dataset_hash,
        split_manifest_sha256=sha256_bytes(payloads["split_manifest.json"]),
        molclr_checkpoint_sha256=sha256_file(molclr_ckpt),
        t8_pass_sha256=t8_sha,
        threshold=threshold,
        train_count=train_count,
        train_label_counts={str(key): int(value) for key, value in train_counts.items()},
    )


def load_prepared_split(
    path: Path,
    *,
    expected_split: str,
    expected_sha256: str,
) -> list[TrainParent]:
    """Load exactly one prepared split and retain source-label parents only."""

    if expected_split not in {"train", "calibration", "test"}:
        raise TasteGlobalGCEFullError("unsupported TasteMolNet split role")
    resolved = path.expanduser().resolve(strict=True)
    if sha256_file(resolved) != expected_sha256:
        raise TasteGlobalGCEFullError(f"{expected_split} split SHA-256 changed")
    try:
        reader = csv.DictReader(
            io.StringIO(resolved.read_text(encoding="utf-8-sig"), newline=""),
            strict=True,
        )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TasteGlobalGCEFullError(f"cannot parse {expected_split} split") from exc
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise TasteGlobalGCEFullError(f"{expected_split} split schema changed")
    result: list[TrainParent] = []
    seen: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        if None in row or set(row) != set(TASTEMOLNET_PREPARED_FIELDS):
            raise TasteGlobalGCEFullError(f"{expected_split} row width changed")
        parent_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get("model_smiles") or "").strip()
        label_text = str(row.get("label") or "").strip()
        if (
            not parent_id
            or parent_id in seen
            or not smiles
            or label_text not in {"0", "1", "2"}
            or str(row.get("label_name") or "").strip() != LABEL_MAP[int(label_text)]
            or str(row.get("split") or "").strip() != expected_split
            or str(row.get("exclusion_reason") or "").strip()
        ):
            raise TasteGlobalGCEFullError(
                f"{expected_split} row authority changed at row {row_number}"
            )
        seen.add(parent_id)
        if int(label_text) == SOURCE_LABEL:
            result.append(TrainParent(parent_id, smiles, SOURCE_LABEL, expected_split))
    if not result:
        raise TasteGlobalGCEFullError(f"{expected_split} has no Sweet source cohort")
    result.sort(key=lambda row: row.parent_id)
    return result


def load_full_train_split(authority: InputAuthority) -> list[TrainParent]:
    """Load every train row (all classes) for native codec/training authority."""

    path = authority.train_path
    reader = csv.DictReader(
        io.StringIO(path.read_text(encoding="utf-8-sig"), newline=""), strict=True
    )
    if tuple(reader.fieldnames or ()) != TASTEMOLNET_PREPARED_FIELDS:
        raise TasteGlobalGCEFullError("train split schema changed")
    rows: list[TrainParent] = []
    counts = {"0": 0, "1": 0, "2": 0}
    seen: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        parent_id = str(row.get("molecule_id") or "").strip()
        smiles = str(row.get("model_smiles") or "").strip()
        label_text = str(row.get("label") or "").strip()
        if (
            None in row
            or set(row) != set(TASTEMOLNET_PREPARED_FIELDS)
            or not parent_id
            or parent_id in seen
            or not smiles
            or label_text not in counts
            or str(row.get("split") or "") != "train"
            or str(row.get("exclusion_reason") or "").strip()
        ):
            raise TasteGlobalGCEFullError(f"train row authority changed at {row_number}")
        seen.add(parent_id)
        counts[label_text] += 1
        rows.append(TrainParent(parent_id, smiles, int(label_text), "train"))
    if len(rows) != authority.train_count or counts != authority.train_label_counts:
        raise TasteGlobalGCEFullError("train count/labels differ from frozen manifest")
    return rows


def select_full_sweet_train_cohort(
    rows: Sequence[TrainParent],
    *,
    scorer: FrozenTasteGINEScorer,
    batch_size: int,
) -> tuple[list[TrainParent], dict[str, Any]]:
    sweet = sorted(
        (row for row in rows if row.label == SOURCE_LABEL),
        key=lambda row: hashlib.sha256(
            f"{SEED}\0{row.parent_id}\0{row.smiles}".encode("utf-8")
        ).hexdigest(),
    )
    if not sweet:
        raise TasteGlobalGCEFullError("train split has no Sweet rows")
    selected: list[TrainParent] = []
    for start in range(0, len(sweet), int(batch_size)):
        batch = sweet[start : start + int(batch_size)]
        predictions = scorer.score_smiles([row.smiles for row in batch])
        if len(predictions) != len(batch):
            raise TasteGlobalGCEFullError("GINE train preselection count changed")
        for row, prediction in zip(batch, predictions, strict=True):
            if prediction.get("checkpoint_id") != scorer.checkpoint_id:
                raise TasteGlobalGCEFullError("GINE train prediction checkpoint changed")
            if int(prediction.get("predicted_label", -1)) == SOURCE_LABEL:
                selected.append(row)
    if len(selected) < 2:
        raise TasteGlobalGCEFullError("full train cohort has fewer than two predicted Sweet rows")
    cohort_hash = stable_sha256(
        [
            {"parent_id": row.parent_id, "smiles": row.smiles, "split": row.split}
            for row in selected
        ]
    )
    return selected, {
        "selection": "all_true_sweet_then_frozen_gine_predicted_sweet",
        "true_sweet_count": len(sweet),
        "selected_count": len(selected),
        "selected_cohort_sha256": cohort_hash,
        "train_only": True,
        "calibration_loaded": False,
        "test_loaded": False,
    }


def _checkpoint_path(output: Path) -> Path:
    return output / "checkpoint.json"


def write_checkpoint(
    output: Path,
    *,
    phase: str,
    resume_identity: Mapping[str, Any],
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": STAGE,
        "phase": phase,
        "resume_identity": dict(resume_identity),
        "resume_identity_sha256": stable_sha256(dict(resume_identity)),
        "detail": dict(detail or {}),
        "written_at": utc_now(),
    }
    atomic_json(_checkpoint_path(output), payload)
    return payload


def load_checkpoint(output: Path, resume_identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(_checkpoint_path(output))
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("stage") != STAGE
        or payload.get("resume_identity") != dict(resume_identity)
        or payload.get("resume_identity_sha256") != stable_sha256(dict(resume_identity))
    ):
        raise TasteGlobalGCEFullError("T13 checkpoint/resume identity changed")
    return payload


def _branch_manifest(root: Path) -> Path:
    return root / "branch_manifest.json"


def run_native_branch(
    *,
    target_label: int,
    generator: OfficialGlobalGCEMutagenicityGenerator,
    parents: Sequence[TrainParent],
    branch_root: Path,
    config: TasteGlobalGCEFullConfig,
) -> dict[str, Any]:
    """Run/resume one real official branch and close its rule artifacts."""

    if target_label not in TARGET_BRANCHES or generator.target_label != target_label:
        raise TasteGlobalGCEFullError("T13 branch target identity changed")
    branch_root.mkdir(parents=True, exist_ok=True)
    existing = _branch_manifest(branch_root)
    if existing.is_file():
        manifest = read_json(existing)
        if (
            manifest.get("schema_version") != BRANCH_MANIFEST_SCHEMA
            or manifest.get("status") != "PASS"
            or manifest.get("target_label") != target_label
            or manifest.get("config") != config.to_dict()
        ):
            raise TasteGlobalGCEFullError("T13 completed branch manifest changed")
        for name, identity in (manifest.get("files") or {}).items():
            path = branch_root / str(name)
            if (
                not path.is_file()
                or path.stat().st_size != int(identity.get("bytes", -1))
                or sha256_file(path) != identity.get("sha256")
            ):
                raise TasteGlobalGCEFullError("T13 completed branch bytes changed")
        return manifest
    result: NativeGenerationResult = generator.generate(
        parents,
        output_dir=branch_root,
        seed=config.seed,
        epochs=config.epochs,
        top_k_native=config.top_k_native,
        learning_rate=config.learning_rate,
        dropout=config.dropout,
        device="cuda:0",
        resume=True,
        generation_chunk_size=config.generation_chunk_size,
        generation_num_workers=0,
        memory_log_every_chunks=1,
        gspan_flush_every=config.gspan_flush_every,
        gspan_max_in_memory_candidates=config.gspan_max_in_memory_candidates,
        gspan_exact_top_k_pruning=False,
        gspan_adoption_proof=None,
        start_parent_offset=0,
        on_training_ready=None,
        on_chunk=None,
        rules_only=True,
        expected_resume_checkpoint=None,
        on_resume_checkpoint=None,
        after_epoch_checkpoint=None,
        on_generation_complete=None,
    )
    summary = result.training_summary
    required = (
        "native_rule_catalog.jsonl",
        "native_rule_rejections.jsonl",
        "globalgce_model.pt",
        "globalgce_rules.pt",
        "training_core_summary.json",
        "globalgce_training_checkpoints/training_checkpoint.pt",
        "globalgce_training_checkpoints/training_heartbeat.json",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in required:
        path = branch_root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise TasteGlobalGCEFullError(f"T13 target-{target_label} lacks {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    valid_rules = summary.get("valid_native_rule_count")
    if (
        summary.get("prediction_backend") != "frozen_gine_differentiable_bridge"
        or summary.get("classifier_family") != "gine"
        or summary.get("oracle_backend") != "gnn"
        or summary.get("rf_oracle_used") is not False
        or summary.get("num_classes") != NUM_CLASSES
        or summary.get("frozen_source_label") != SOURCE_LABEL
        or summary.get("frozen_target_label") != target_label
        or summary.get("generation_input_split") != "train"
        or summary.get("calibration_loaded") is not False
        or summary.get("test_loaded") is not False
        or type(valid_rules) is not int
        or valid_rules < MIN_RULES
    ):
        raise TasteGlobalGCEFullError(f"T13 target-{target_label} science contract failed")
    manifest = {
        "schema_version": BRANCH_MANIFEST_SCHEMA,
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "status": "PASS",
        "source_label": SOURCE_LABEL,
        "target_label": target_label,
        "num_classes": NUM_CLASSES,
        "config": config.to_dict(),
        "train_only": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "valid_native_rule_count": valid_rules,
        "training_resume_identity_sha256": summary.get(
            "training_resume_identity_sha256"
        ),
        "official_epoch_checkpoint_resume": True,
        "files": files,
        "completed_at": utc_now(),
    }
    atomic_json(existing, manifest)
    return manifest


def merge_branch_rules(branch_roots: Mapping[int, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(branch_roots) != set(TARGET_BRANCHES):
        raise TasteGlobalGCEFullError("T13 merge requires target branches 0 and 2")
    merged: dict[str, dict[str, Any]] = {}
    branch_counts: dict[int, int] = {}
    for target in TARGET_BRANCHES:
        rows = read_jsonl(branch_roots[target] / "native_rule_catalog.jsonl")
        if len(rows) < MIN_RULES:
            raise TasteGlobalGCEFullError(f"T13 target-{target} has fewer than 10 rules")
        seen: set[str] = set()
        for rank, raw in enumerate(rows, start=1):
            rule = GlobalGCENativeRule.from_payload(raw)
            # Native rank/index is branch-local metadata, not part of the
            # transformation.  Cross-branch dedup therefore hashes only the
            # exact LHS/RHS tensors and their atom/bond vocabularies.
            action_payload = rule.to_payload()
            action_payload.pop("rule_id", None)
            action_payload.pop("native_rule_index", None)
            content_hash = stable_sha256(action_payload)
            if content_hash in seen:
                raise TasteGlobalGCEFullError("one T13 branch repeats a canonical rule")
            seen.add(content_hash)
            candidate_id = f"globalgce_rule_{content_hash}"
            existing = merged.get(content_hash)
            if existing is None:
                row = dict(raw)
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "rule_content_hash": content_hash,
                        "action_kind": "lhs_rhs_graph_transformation_rule",
                        "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
                        "target_branches": [target],
                        "branch_native_ranks": {str(target): rank},
                        "source_split": "train",
                        "oracle_backend": "gnn",
                        "classifier_family": "gine",
                        "rf_oracle_used": False,
                    }
                )
                GlobalGCENativeRule.from_payload(row)
                merged[content_hash] = row
            else:
                existing["target_branches"].append(target)
                existing["target_branches"].sort()
                existing["branch_native_ranks"][str(target)] = rank
        branch_counts[target] = len(rows)
    result = sorted(
        merged.values(),
        key=lambda row: (
            min(int(value) for value in row["branch_native_ranks"].values()),
            row["candidate_id"],
        ),
    )
    if len(result) < MIN_RULES or len({row["candidate_id"] for row in result}) != len(result):
        raise TasteGlobalGCEFullError("T13 canonical merge has insufficient/duplicate rules")
    return result, {
        "target_0_rule_count": branch_counts[0],
        "target_2_rule_count": branch_counts[2],
        "premerge_rule_count": branch_counts[0] + branch_counts[2],
        "merged_unique_rule_count": len(result),
        "cross_branch_duplicate_count": branch_counts[0] + branch_counts[2] - len(result),
        "canonical_dedup_complete": True,
        "merged_rule_set_sha256": stable_sha256(
            [row["candidate_id"] for row in result]
        ),
    }


def _prediction(row: Mapping[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
    probabilities = row.get("probabilities")
    predicted = row.get("predicted_label")
    if (
        row.get("checkpoint_id") != checkpoint_id
        or row.get("num_classes") != NUM_CLASSES
        or row.get("source_label") != SOURCE_LABEL
        or str(row.get("backbone") or "").lower() != "gine"
        or type(predicted) is not int
        or predicted not in range(NUM_CLASSES)
        or not isinstance(probabilities, list)
        or len(probabilities) != NUM_CLASSES
        or any(not math.isfinite(float(value)) for value in probabilities)
        or max(range(NUM_CLASSES), key=lambda index: float(probabilities[index])) != predicted
    ):
        raise TasteGlobalGCEFullError("T13 prediction differs from frozen GINE contract")
    return {"predicted_label": predicted, "probabilities": [float(v) for v in probabilities]}


def evaluate_one_parent(
    *,
    parent: TrainParent,
    rules: Sequence[Mapping[str, Any]],
    scorer: FrozenTasteGINEScorer,
    provider: MolCLRNodeWassersteinDistance,
    split: str,
) -> list[dict[str, Any]]:
    before = _prediction(scorer.score_smiles([parent.smiles])[0], checkpoint_id=scorer.checkpoint_id)
    parsed_rules = [GlobalGCENativeRule.from_payload(row) for row in rules]
    applications: list[list[dict[str, Any]]] = []
    unique_smiles: dict[str, None] = {}
    failures: list[str | None] = []
    for rule in parsed_rules:
        try:
            rows = apply_rule_to_parent(parent.smiles, rule)
            valid = [dict(row) for row in rows if row.get("valid") is True]
            failures.append(None if valid else "no_legal_native_lhs_match_or_sanitized_rhs")
        except Exception as exc:
            valid = []
            failures.append(f"{type(exc).__name__}:{exc}")
        applications.append(valid)
        for row in valid:
            unique_smiles[str(row["canonical_smiles"])] = None
    ordered_smiles = sorted(unique_smiles)
    predictions = (
        scorer.score_smiles(ordered_smiles) if ordered_smiles else []
    )
    after_by_smiles = {
        smiles: _prediction(raw, checkpoint_id=scorer.checkpoint_id)
        for smiles, raw in zip(ordered_smiles, predictions, strict=True)
    }
    result: list[dict[str, Any]] = []
    for raw_rule, valid, failure in zip(rules, applications, failures, strict=True):
        evaluated: list[dict[str, Any]] = []
        for application in valid:
            candidate_smiles = str(application["canonical_smiles"])
            after = after_by_smiles[candidate_smiles]
            strict = (
                before["predicted_label"] == SOURCE_LABEL
                and after["predicted_label"] != SOURCE_LABEL
            )
            source_drop = (
                before["probabilities"][SOURCE_LABEL]
                - after["probabilities"][SOURCE_LABEL]
            )
            distance: float | None = None
            distance_failure: str | None = None
            if strict:
                distance_result = provider.distance(parent.smiles, candidate_smiles)
                value = distance_result.get("distance")
                if (
                    distance_result.get("ok") is True
                    and value is not None
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    distance = float(value)
                else:
                    distance_failure = str(
                        distance_result.get("error") or "wnode_distance_failed"
                    )
            evaluated.append(
                {
                    "application": application,
                    "after": after,
                    "strict": strict,
                    "source_drop": source_drop,
                    "distance": distance,
                    "distance_failure": distance_failure,
                }
            )
        legal = [row for row in evaluated if row["distance"] is not None]
        legal.sort(
            key=lambda row: (
                float(row["distance"]),
                str(row["application"]["match_id"]),
                str(row["application"]["canonical_smiles"]),
            )
        )
        if legal:
            chosen = legal[0]
        elif evaluated:
            chosen = min(
                evaluated,
                key=lambda row: (
                    not bool(row["strict"]),
                    -float(row["source_drop"]),
                    str(row["application"]["match_id"]),
                ),
            )
        else:
            chosen = None
        after = chosen["after"] if chosen else None
        application = chosen["application"] if chosen else None
        distance = float(legal[0]["distance"]) if legal else None
        result.append(
            {
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "split": split,
                "parent_id": parent.parent_id,
                "parent_smiles": parent.smiles,
                "candidate_id": raw_rule["candidate_id"],
                "target_branches": list(raw_rule["target_branches"]),
                "rule_content_hash": raw_rule["rule_content_hash"],
                "action_kind": "lhs_rhs_graph_transformation_rule",
                "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
                "applicable": bool(valid),
                "native_match_count": len(valid),
                "selected_match_id": application.get("match_id") if application else None,
                "canonical_smiles": application.get("canonical_smiles") if application else "",
                "pred_before": before["predicted_label"],
                "pred_after": after["predicted_label"] if after else None,
                "p_before": before["probabilities"],
                "p_after": after["probabilities"] if after else [],
                "p1_before": before["probabilities"][SOURCE_LABEL],
                "p1_after": after["probabilities"][SOURCE_LABEL] if after else None,
                "cf_drop": chosen["source_drop"] if chosen else None,
                "cf_flip": bool(chosen and chosen["strict"]),
                "pair_strict_flip": distance is not None,
                "destination_label": after["predicted_label"] if distance is not None else None,
                "wnode_distance": distance,
                "distance_for_selection": distance if distance is not None else "+inf",
                "failure_reason": (
                    None
                    if distance is not None
                    else (
                        chosen.get("distance_failure")
                        if chosen and chosen.get("distance_failure")
                        else "frozen_gine_not_strict_flip"
                        if chosen
                        else failure
                    )
                ),
                "cf_mode": CF_MODE,
                "source_label": SOURCE_LABEL,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "oracle_checkpoint_hash": scorer.checkpoint_id,
            }
        )
    return result


def evaluate_split_resumable(
    *,
    split: str,
    parents: Sequence[TrainParent],
    rules: Sequence[Mapping[str, Any]],
    scorer: FrozenTasteGINEScorer,
    provider: MolCLRNodeWassersteinDistance,
    output: Path,
    checkpoint_callback: Callable[[int], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks = output / "raw" / f"{split}_pair_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    candidate_ids = [str(row["candidate_id"]) for row in rules]
    if len(candidate_ids) < MIN_RULES or len(candidate_ids) != len(set(candidate_ids)):
        raise TasteGlobalGCEFullError("T13 evaluation rule identity is invalid")
    all_rows: list[dict[str, Any]] = []
    for position, parent in enumerate(parents):
        chunk = chunks / f"{position:08d}.jsonl"
        if chunk.is_file():
            rows = read_jsonl(chunk)
            if (
                len(rows) != len(rules)
                or [str(row.get("candidate_id")) for row in rows] != candidate_ids
                or any(
                    row.get("parent_id") != parent.parent_id or row.get("split") != split
                    for row in rows
                )
            ):
                raise TasteGlobalGCEFullError(f"T13 {split} resume chunk changed")
        else:
            rows = evaluate_one_parent(
                parent=parent,
                rules=rules,
                scorer=scorer,
                provider=provider,
                split=split,
            )
            atomic_jsonl(chunk, rows)
        all_rows.extend(rows)
        checkpoint_callback(position + 1)
    pair_path = output / "raw" / f"{split}_pair_details.jsonl"
    atomic_jsonl(pair_path, all_rows)
    return all_rows, {
        "split": split,
        "parent_count": len(parents),
        "candidate_count": len(rules),
        "pair_count": len(all_rows),
        "pair_details_sha256": sha256_file(pair_path),
        "parent_ids_sha256": stable_sha256(sorted(row.parent_id for row in parents)),
        "candidate_ids_sha256": stable_sha256(candidate_ids),
        "resumable_parent_chunks": True,
        "checkpointed_parent_count": len(parents),
    }


def select_rules_on_calibration(
    rules: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    theta_star: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = {
        str(row["candidate_id"]): [] for row in rules
    }
    for row in pair_rows:
        candidate = str(row.get("candidate_id") or "")
        if candidate not in by_candidate or row.get("split") != "calibration":
            raise TasteGlobalGCEFullError("calibration pair matrix escaped rule/split identity")
        by_candidate[candidate].append(row)
    rule_by_id = {str(row["candidate_id"]): dict(row) for row in rules}
    if any(not rows for rows in by_candidate.values()):
        raise TasteGlobalGCEFullError("calibration matrix lacks one or more native rules")
    selected: list[str] = []
    covered_theta: set[str] = set()
    covered_strict: set[str] = set()
    trace: list[dict[str, Any]] = []
    remaining = set(rule_by_id)
    while remaining and len(selected) < K_MAX:
        ranked: list[tuple[tuple[Any, ...], str, dict[str, Any]]] = []
        for candidate in remaining:
            rows = by_candidate[candidate]
            strict = {
                str(row["parent_id"])
                for row in rows
                if row.get("pair_strict_flip") is True
            }
            theta = {
                str(row["parent_id"])
                for row in rows
                if row.get("pair_strict_flip") is True
                and float(row["wnode_distance"]) <= theta_star
            }
            distances = [
                float(row["wnode_distance"])
                for row in rows
                if row.get("pair_strict_flip") is True
            ]
            mean_distance = sum(distances) / len(distances) if distances else math.inf
            stats = {
                "marginal_theta_coverage": len(theta - covered_theta),
                "marginal_strict_coverage": len(strict - covered_strict),
                "total_theta_coverage": len(theta),
                "total_strict_coverage": len(strict),
                "mean_strict_distance": mean_distance if math.isfinite(mean_distance) else None,
            }
            key = (
                -stats["marginal_theta_coverage"],
                -stats["marginal_strict_coverage"],
                -stats["total_theta_coverage"],
                -stats["total_strict_coverage"],
                mean_distance,
                candidate,
            )
            ranked.append((key, candidate, stats))
        _key, winner, stats = min(ranked)
        winner_rows = by_candidate[winner]
        covered_strict.update(
            str(row["parent_id"])
            for row in winner_rows
            if row.get("pair_strict_flip") is True
        )
        covered_theta.update(
            str(row["parent_id"])
            for row in winner_rows
            if row.get("pair_strict_flip") is True
            and float(row["wnode_distance"]) <= theta_star
        )
        selected.append(winner)
        remaining.remove(winner)
        trace.append(
            {
                "rank": len(selected),
                "candidate_id": winner,
                **stats,
                "cumulative_theta_coverage": len(covered_theta),
                "cumulative_strict_coverage": len(covered_strict),
            }
        )
    if len(selected) < MIN_RULES:
        raise TasteGlobalGCEFullError("T13 calibration selected fewer than ten rules")
    ordered = [rule_by_id[candidate] for candidate in selected]
    return ordered, {
        "selector": (
            "calibration_greedy_marginal_theta_then_strict_then_total_coverage_"
            "then_mean_wnode_then_rule_id_v1"
        ),
        "theta_star": theta_star,
        "candidate_count": len(rules),
        "selected_count": len(ordered),
        "ordered_rule_ids": selected,
        "ordered_rule_ids_sha256": stable_sha256(selected),
        "trace": trace,
        "selector_fitted_on_calibration": True,
        "test_loaded": False,
        "test_used_for_selection": False,
    }


def authorize_and_load_test_after_freeze(
    *,
    authority: InputAuthority,
    selection_manifest_path: Path,
) -> list[TrainParent]:
    """The only function in the science path allowed to open held-out test."""

    selection = read_json(selection_manifest_path)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != "FROZEN"
        or selection.get("selection_frozen") is not True
        or selection.get("selector_fitted_on_calibration") is not True
        or selection.get("test_loaded") is not False
        or selection.get("test_used_for_selection") is not False
    ):
        raise TasteGlobalGCEFullError("held-out test access requires frozen calibration order")
    return load_prepared_split(
        authority.test_path,
        expected_split="test",
        expected_sha256=authority.declared_test_sha256,
    )


def _median(values: Sequence[float]) -> float | str:
    if not values:
        return "N/A"
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def compute_standardized_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
    ordered_rule_ids: Sequence[str],
    threshold: ThresholdContract,
) -> dict[str, Any]:
    if not MIN_RULES <= len(ordered_rule_ids) <= K_MAX:
        raise TasteGlobalGCEFullError("T13 frozen rule count must be 10..20")
    by_parent: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in pair_rows:
        if row.get("split") != "test" or row.get("rf_oracle_used") is not False:
            raise TasteGlobalGCEFullError("T13 test pair provenance changed")
        parent = str(row.get("parent_id") or "")
        candidate = str(row.get("candidate_id") or "")
        if not parent or candidate not in ordered_rule_ids:
            raise TasteGlobalGCEFullError("T13 test pair escaped frozen identities")
        if candidate in by_parent.setdefault(parent, {}):
            raise TasteGlobalGCEFullError("T13 repeats one parent/rule test pair")
        if row.get("pair_strict_flip") is True:
            distance = row.get("wnode_distance")
            destination = row.get("destination_label")
            if (
                distance is None
                or not math.isfinite(float(distance))
                or float(distance) < 0.0
                or row.get("pred_before") != SOURCE_LABEL
                or destination not in DESTINATION_LABELS
            ):
                raise TasteGlobalGCEFullError("strict test pair lacks valid WNode/destination")
        elif row.get("wnode_distance") is not None:
            raise TasteGlobalGCEFullError("non-strict test pair unexpectedly has WNode")
        by_parent[parent][candidate] = row
    if not by_parent or any(set(rows) != set(ordered_rule_ids) for rows in by_parent.values()):
        raise TasteGlobalGCEFullError("T13 test matrix is not a complete Cartesian product")
    parents = sorted(by_parent)
    best: dict[str, tuple[float, float, str, int, bool] | None] = {
        parent: None for parent in parents
    }
    applicable = {parent: False for parent in parents}
    prefix: list[dict[str, Any]] = []
    parent_best: list[dict[str, Any]] = []
    for k, candidate in enumerate(ordered_rule_ids, start=1):
        for parent in parents:
            row = by_parent[parent][candidate]
            applicable[parent] = applicable[parent] or bool(row.get("applicable"))
            if row.get("pair_strict_flip") is not True:
                continue
            value = (
                float(row["wnode_distance"]),
                -float(row.get("cf_drop") or 0.0),
                candidate,
                int(row["destination_label"]),
                bool(row.get("applicable")),
            )
            if best[parent] is None or value[:3] < best[parent][:3]:
                best[parent] = value
        finite = [value for value in best.values() if value is not None]
        covered = [value for value in finite if value[0] <= threshold.theta_star]
        capped = [
            min(value[0], threshold.cost_cap) if value is not None else threshold.cost_cap
            for value in best.values()
        ]
        conditional = [value[0] for value in finite]
        cf_drops = [-value[1] for value in covered]
        row = {
            "dataset": DATASET,
            "method": METHOD,
            "k": k,
            "SuppCov": len(finite) / len(parents),
            "CCRCov": len(covered) / len(parents),
            "coverage": len(covered) / len(parents),
            "cost": sum(capped) / len(capped),
            "fixed_capped_mean_cost": sum(capped) / len(capped),
            "conditional_mean_cost": (
                sum(conditional) / len(conditional) if conditional else "N/A"
            ),
            "conditional_median_cost": _median(conditional),
            "CFDrop": sum(cf_drops) / len(cf_drops) if cf_drops else "N/A",
            "FlipRate": len(finite) / len(parents),
            "StructRed": "N/A",
            "CovRed": "N/A",
            "ValidRate": sum(applicable.values()) / len(parents),
            "AvgSize": "N/A",
            "applicable_rate": sum(applicable.values()) / len(parents),
        }
        prefix.append(row)
        for parent in parents:
            value = best[parent]
            parent_best.append(
                {
                    "dataset": DATASET,
                    "method": METHOD,
                    "k": k,
                    "parent_id": parent,
                    "best_distance": value[0] if value is not None else "N/A",
                    "capped_distance": (
                        min(value[0], threshold.cost_cap)
                        if value is not None
                        else threshold.cost_cap
                    ),
                    "best_candidate_id": value[2] if value is not None else "N/A",
                    "destination_label": value[3] if value is not None else "N/A",
                    "strict_recourse_available": value is not None,
                    "theta_star_covered": value is not None and value[0] <= threshold.theta_star,
                    "applicable": applicable[parent],
                }
            )
    for k in range(len(ordered_rule_ids) + 1, K_MAX + 1):
        plateau = dict(prefix[-1])
        plateau.update(
            {"k": k, "effective_rule_count": len(ordered_rule_ids), "plateau_after_effective_k": True}
        )
        prefix.append(plateau)
        for parent in parents:
            value = best[parent]
            parent_best.append(
                {
                    "dataset": DATASET,
                    "method": METHOD,
                    "k": k,
                    "parent_id": parent,
                    "best_distance": value[0] if value is not None else "N/A",
                    "capped_distance": min(value[0], threshold.cost_cap) if value else threshold.cost_cap,
                    "best_candidate_id": value[2] if value else "N/A",
                    "destination_label": value[3] if value else "N/A",
                    "strict_recourse_available": value is not None,
                    "theta_star_covered": value is not None and value[0] <= threshold.theta_star,
                    "applicable": applicable[parent],
                    "effective_rule_count": len(ordered_rule_ids),
                    "plateau_after_effective_k": True,
                }
            )
    k10 = {
        row["parent_id"]: row for row in parent_best if int(row["k"]) == TABLE2_K
    }
    figure4 = []
    for value in threshold.values:
        coverage = sum(
            row["best_distance"] != "N/A" and float(row["best_distance"]) <= value
            for row in k10.values()
        ) / len(k10)
        figure4.append(
            {
                "dataset": DATASET,
                "method": METHOD,
                "k": TABLE2_K,
                "threshold": value,
                "coverage": coverage,
                "CCRCov": coverage,
            }
        )
    destinations = [value for value in best.values() if value is not None]
    destination_rows = []
    for destination in DESTINATION_LABELS:
        count = sum(value[3] == destination for value in destinations)
        destination_rows.append(
            {
                "dataset": DATASET,
                "method": METHOD,
                "destination_label": destination,
                "count": count,
                "rate": count / len(destinations) if destinations else "N/A",
                "denominator": len(destinations),
                "distribution_scope": "K20 finite untargeted strict flips",
            }
        )
    return {
        "prefix": prefix,
        "parent_best": parent_best,
        "figure3": [
            {
                "dataset": DATASET,
                "method": METHOD,
                "k": row["k"],
                "coverage": row["CCRCov"],
                "cost": row["cost"],
            }
            for row in prefix
        ],
        "figure4": figure4,
        "table2": [dict(prefix[TABLE2_K - 1])],
        "destination": destination_rows,
        "parent_count": len(parents),
        "pair_count": len(pair_rows),
        "effective_rule_count": len(ordered_rule_ids),
    }


def _immutable_artifact_inventory(output: Path) -> dict[str, dict[str, Any]]:
    names = (
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "table2_globalgce_k10.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "raw/merged_rules.jsonl",
        "raw/calibration_pair_details.jsonl",
        "raw/selected_rules.jsonl",
        "raw/test_pair_details.jsonl",
        "raw/selection_manifest.json",
        "raw/test_evaluation_manifest.json",
    )
    inventory: dict[str, dict[str, Any]] = {}
    for name in names:
        path = output / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise TasteGlobalGCEFullError(f"T13 immutable artifact is absent: {name}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return inventory


def _common_manifest(
    *,
    authority: InputAuthority,
    output: Path,
    test_parent_ids_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "method": METHOD,
        "stage": STAGE,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint": str(authority.checkpoint_path),
        "oracle_hash": authority.checkpoint_id,
        "oracle_checkpoint_hash": authority.checkpoint_id,
        "dataset_hash": authority.dataset_hash,
        "test_parent_ids_sha256": test_parent_ids_sha256,
        "test_split_hash": authority.declared_test_sha256,
        "distance_line": DISTANCE_LINE,
        "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
        "cf_mode": CF_MODE,
        "threshold_config_hash": authority.threshold.config_hash,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "raw_output_root": str(output),
    }


def run_t13_full(
    *,
    authority: InputAuthority,
    output_dir: str | Path,
    config: TasteGlobalGCEFullConfig,
    resume: bool,
    device: str,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
) -> dict[str, Any]:
    config.validate()
    if device != "cuda:0":
        raise TasteGlobalGCEFullError("T13 science is bound to logical cuda:0")
    output = Path(output_dir).expanduser().absolute()
    resume_identity = authority.resume_identity(config)
    if resume:
        if not output.is_dir() or not _checkpoint_path(output).is_file():
            raise TasteGlobalGCEFullError("--resume requires an existing T13 checkpoint")
        checkpoint = load_checkpoint(output, resume_identity)
        if checkpoint.get("phase") in {"SEALED", "PASS"}:
            return read_json(output / "run_manifest.json")
    else:
        if output.exists():
            raise TasteGlobalGCEFullError("fresh T13 output root already exists")
        output.mkdir(parents=True)
        (output / "raw").mkdir()
        checkpoint = write_checkpoint(
            output, phase="INITIALIZED", resume_identity=resume_identity
        )
    if (output / "PASS").exists():
        raise TasteGlobalGCEFullError("T13 science cannot overwrite terminal PASS")

    payloads = _checkpoint_payloads(authority.checkpoint_path)
    scorer = FrozenTasteGINEScorer(
        payloads, device=device, batch_size=config.oracle_batch_size
    )
    if scorer.checkpoint_id != authority.checkpoint_id:
        raise TasteGlobalGCEFullError("T13 scorer checkpoint differs from authority")
    train_rows = load_full_train_split(authority)
    selected_train, train_selection = select_full_sweet_train_cohort(
        train_rows, scorer=scorer, batch_size=config.oracle_batch_size
    )
    atomic_json(output / "raw" / "train_cohort_manifest.json", train_selection)
    branch_roots = {target: output / "raw" / f"target_{target}" for target in TARGET_BRANCHES}
    branch_manifests: dict[int, dict[str, Any]] = {}
    for target in TARGET_BRANCHES:
        branch_already_complete = _branch_manifest(branch_roots[target]).is_file()
        if not branch_already_complete:
            write_checkpoint(
                output,
                phase=f"TARGET_{target}_RUNNING",
                resume_identity=resume_identity,
                detail={"target_label": target},
            )
        generator = OfficialGlobalGCEMutagenicityGenerator(
            authority.official_root,
            native_train_csv=authority.train_path,
            dataset_name=DATASET,
            min_freq=config.min_freq,
            frozen_gine_checkpoint=authority.checkpoint_path,
            source_label=SOURCE_LABEL,
            target_label=target,
            num_classes=NUM_CLASSES,
            require_isolated_imports=True,
        )
        branch_manifests[target] = run_native_branch(
            target_label=target,
            generator=generator,
            parents=selected_train,
            branch_root=branch_roots[target],
            config=config,
        )
        if not branch_already_complete:
            write_checkpoint(
                output,
                phase=f"TARGET_{target}_COMPLETE",
                resume_identity=resume_identity,
                detail={
                    "target_label": target,
                    "branch_manifest_sha256": sha256_file(_branch_manifest(branch_roots[target])),
                },
            )
    merged_rules, merge_manifest = merge_branch_rules(branch_roots)
    atomic_jsonl(output / "raw" / "merged_rules.jsonl", merged_rules)
    merge_manifest.update(
        {
            "dataset": DATASET,
            "method": METHOD,
            "source_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "merged_rules_sha256": sha256_file(output / "raw" / "merged_rules.jsonl"),
        }
    )
    atomic_json(output / "raw" / "merge_manifest.json", merge_manifest)
    write_checkpoint(output, phase="MERGE_COMPLETE", resume_identity=resume_identity)

    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=authority.molclr_root,
            molclr_ckpt=authority.molclr_checkpoint,
            cache_db=Path(wnode_cache_db).expanduser().absolute(),
            node_emb_cache_dir=Path(node_embedding_cache_dir).expanduser().absolute(),
            device=device,
            distance_namespace=DISTANCE_NAMESPACE,
        )
    )
    try:
        selection_path = output / "raw" / "selection_manifest.json"
        selected_rules_path = output / "raw" / "selected_rules.jsonl"
        if selection_path.is_file() or selected_rules_path.is_file():
            if not (selection_path.is_file() and selected_rules_path.is_file()):
                raise TasteGlobalGCEFullError("T13 frozen selection is only partially present")
            selection_manifest = read_json(selection_path)
            selected_rules = read_jsonl(selected_rules_path)
            ordered_ids = [str(row.get("candidate_id") or "") for row in selected_rules]
            if (
                selection_manifest.get("schema_version") != SELECTION_SCHEMA
                or selection_manifest.get("status") != "FROZEN"
                or selection_manifest.get("selection_frozen") is not True
                or selection_manifest.get("selector_fitted_on_calibration") is not True
                or selection_manifest.get("test_loaded") is not False
                or selection_manifest.get("test_used_for_selection") is not False
                or selection_manifest.get("threshold_config_hash")
                != authority.threshold.config_hash
                or selection_manifest.get("oracle_checkpoint_hash")
                != authority.checkpoint_id
                or selection_manifest.get("molclr_checkpoint_hash")
                != authority.molclr_checkpoint_sha256
                or ordered_ids != list(selection_manifest.get("ordered_rule_ids") or [])
                or not MIN_RULES <= len(ordered_ids) <= K_MAX
                or selection_manifest.get("selected_rules_sha256")
                != sha256_file(selected_rules_path)
            ):
                raise TasteGlobalGCEFullError("T13 frozen calibration selection changed")
        else:
            calibration_parents = load_prepared_split(
                authority.calibration_path,
                expected_split="calibration",
                expected_sha256=authority.calibration_sha256,
            )
            calibration_rows, calibration_manifest = evaluate_split_resumable(
                split="calibration",
                parents=calibration_parents,
                rules=merged_rules,
                scorer=scorer,
                provider=provider,
                output=output,
                checkpoint_callback=lambda count: write_checkpoint(
                    output,
                    phase="CALIBRATION_RUNNING",
                    resume_identity=resume_identity,
                    detail={"completed_parent_count": count, "parent_count": len(calibration_parents)},
                ),
            )
            selected_rules, selection = select_rules_on_calibration(
                merged_rules,
                calibration_rows,
                theta_star=authority.threshold.theta_star,
            )
            atomic_jsonl(selected_rules_path, selected_rules)
            frozen_at = utc_now()
            selection_manifest = {
                "schema_version": SELECTION_SCHEMA,
                "dataset": DATASET,
                "method": METHOD,
                "stage": STAGE,
                "status": "FROZEN",
                "selection_frozen": True,
                "frozen_at": frozen_at,
                **selection,
                **authority.threshold.to_dict(),
                "calibration_manifest": calibration_manifest,
                "selected_rules_sha256": sha256_file(selected_rules_path),
                "oracle_checkpoint_hash": authority.checkpoint_id,
                "molclr_checkpoint_hash": authority.molclr_checkpoint_sha256,
                "rf_oracle_used": False,
            }
            atomic_json(selection_path, selection_manifest)
            write_checkpoint(
                output,
                phase="CALIBRATION_SELECTION_FROZEN",
                resume_identity=resume_identity,
                detail={"selection_manifest_sha256": sha256_file(selection_path)},
            )
        selection_sha = sha256_file(selection_path)

        # Test access begins only after the fsynced selection manifest and its
        # checkpoint have been published above.
        test_started_at = utc_now()
        test_parents = authorize_and_load_test_after_freeze(
            authority=authority,
            selection_manifest_path=output / "raw" / "selection_manifest.json",
        )
        test_rows, test_manifest = evaluate_split_resumable(
            split="test",
            parents=test_parents,
            rules=selected_rules,
            scorer=scorer,
            provider=provider,
            output=output,
            checkpoint_callback=lambda count: write_checkpoint(
                output,
                phase="TEST_RUNNING",
                resume_identity=resume_identity,
                detail={"completed_parent_count": count, "parent_count": len(test_parents)},
            ),
        )
        provider_stats = provider.stats_dict()
    finally:
        provider.close()
    test_manifest.update(
        {
            "started_at": test_started_at,
            "completed_at": utc_now(),
            "selection_manifest_sha256": selection_sha,
            "selection_frozen_before_test": True,
            "test_used_for_selection": False,
        }
    )
    atomic_json(output / "raw" / "test_evaluation_manifest.json", test_manifest)
    metrics = compute_standardized_metrics(
        test_rows,
        [str(row["candidate_id"]) for row in selected_rules],
        authority.threshold,
    )
    atomic_csv(output / "figure3_coverage_vs_k.csv", metrics["figure3"])
    atomic_csv(output / "figure4_coverage_vs_threshold.csv", metrics["figure4"])
    atomic_csv(output / "prefix_metrics.csv", metrics["prefix"])
    atomic_json(output / "prefix_metrics.json", metrics["prefix"])
    atomic_csv(output / "parent_best_distances.csv", metrics["parent_best"])
    atomic_csv(output / "destination_distribution.csv", metrics["destination"])
    atomic_csv(output / "table2_globalgce_k10.csv", metrics["table2"])
    test_parent_hash = stable_sha256(sorted(row.parent_id for row in test_parents))
    common = _common_manifest(
        authority=authority, output=output, test_parent_ids_sha256=test_parent_hash
    )
    summary = {
        "schema_version": "tastemolnet_t13_summary_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "selection_frozen_before_test": True,
        "calibration_loaded": True,
        "test_loaded": True,
        "target_branches": [0, 2],
        "canonical_dedup_complete": True,
        "effective_rule_count": metrics["effective_rule_count"],
        "parent_count": metrics["parent_count"],
        "pair_count": metrics["pair_count"],
        "M_configured_max": None,
        "M_effective": None,
        "K_MAX": K_MAX,
        "MIN_RULES_FOR_MAIN_TABLE": MIN_RULES,
        "distance_provider_stats": provider_stats,
        "threshold_contract": authority.threshold.to_dict(),
        "branch_manifests": {
            str(target): sha256_file(_branch_manifest(branch_roots[target]))
            for target in TARGET_BRANCHES
        },
    }
    oracle_manifest = {
        "schema_version": "tastemolnet_t13_oracle_manifest_v1",
        **common,
        "temperature": scorer.temperature,
        "num_classes": scorer.num_classes,
        "source_label": scorer.source_label,
        "same_frozen_gine_for_generation_calibration_test": True,
        "calibration_loaded_for_training": False,
        "test_loaded_for_training": False,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "frozen": True,
    }
    evaluation_manifest = {
        "schema_version": "tastemolnet_t13_evaluation_manifest_v1",
        **common,
        "status": "SEALED",
        "selection_manifest_sha256": selection_sha,
        "test_evaluation_manifest_sha256": sha256_file(
            output / "raw" / "test_evaluation_manifest.json"
        ),
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "strict_flip_definition": "pred_before == 1 and pred_after != 1",
        "destination_labels": [0, 2],
        "full_cartesian_test_pairs": True,
        "frozen": True,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "oracle_manifest.json", oracle_manifest)
    atomic_json(output / "evaluation_manifest.json", evaluation_manifest)
    inventory = _immutable_artifact_inventory(output)
    freeze_manifest = {
        "schema_version": "tastemolnet_t13_freeze_manifest_v1",
        **common,
        "status": "SEALED",
        "frozen": True,
        "artifacts_frozen": True,
        "files": inventory,
        "inventory_sha256": stable_sha256(inventory),
        "sealed_at": utc_now(),
    }
    atomic_json(output / "freeze_manifest.json", freeze_manifest)
    run_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        **common,
        "status": "SEALED",
        "state": "SEALED",
        "run_complete": False,
        "raw_output_complete": True,
        "source_artifacts_complete": True,
        "frozen": True,
        "artifacts_frozen": True,
        "selection_frozen_before_test": True,
        "test_used_for_selection": False,
        "threshold_fitted_on_test": False,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "independent_terminal_verification_required": True,
        "worker_wrote_pass": False,
        "sealed_at": utc_now(),
    }
    atomic_json(output / "run_manifest.json", run_manifest)
    _atomic_bytes(output / "SEALED", b"SEALED\n")
    write_checkpoint(output, phase="SEALED", resume_identity=resume_identity)
    return run_manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def verify_t13_output(output_dir: str | Path) -> dict[str, Any]:
    """Independently replay immutable outputs and publish terminal PASS."""

    output = Path(output_dir).expanduser().resolve(strict=True)
    if not (output / "SEALED").is_file():
        raise TasteGlobalGCEFullError("T13 verifier requires SEALED science output")
    if (output / "PASS").exists():
        existing = read_json(output / "final_artifact_audit.json")
        if existing.get("passed") is True:
            return existing
        raise TasteGlobalGCEFullError("T13 PASS marker conflicts with failed audit")
    run_manifest = read_json(output / "run_manifest.json")
    freeze = read_json(output / "freeze_manifest.json")
    if (
        run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or run_manifest.get("status") != "SEALED"
        or freeze.get("status") != "SEALED"
        or freeze.get("frozen") is not True
    ):
        raise TasteGlobalGCEFullError("T13 verifier received a non-SEALED run")
    inventory = freeze.get("files")
    if not isinstance(inventory, dict) or freeze.get("inventory_sha256") != stable_sha256(inventory):
        raise TasteGlobalGCEFullError("T13 frozen inventory is malformed")
    for name, identity in inventory.items():
        path = output / name
        if (
            not path.is_file()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(path) != identity.get("sha256")
        ):
            raise TasteGlobalGCEFullError(f"T13 frozen artifact changed: {name}")
    selection = read_json(output / "raw" / "selection_manifest.json")
    test_manifest = read_json(output / "raw" / "test_evaluation_manifest.json")
    selected_rules = read_jsonl(output / "raw" / "selected_rules.jsonl")
    calibration_rows = read_jsonl(output / "raw" / "calibration_pair_details.jsonl")
    test_rows = read_jsonl(output / "raw" / "test_pair_details.jsonl")
    if (
        selection.get("status") != "FROZEN"
        or selection.get("selection_frozen") is not True
        or selection.get("selector_fitted_on_calibration") is not True
        or selection.get("test_loaded") is not False
        or selection.get("test_used_for_selection") is not False
        or any(row.get("split") != "calibration" for row in calibration_rows)
        or any(row.get("split") != "test" for row in test_rows)
        or test_manifest.get("selection_manifest_sha256")
        != sha256_file(output / "raw" / "selection_manifest.json")
        or test_manifest.get("selection_frozen_before_test") is not True
        or str(selection.get("frozen_at")) > str(test_manifest.get("started_at"))
    ):
        raise TasteGlobalGCEFullError("T13 calibration/test isolation replay failed")
    ordered = [str(row["candidate_id"]) for row in selected_rules]
    if ordered != list(selection.get("ordered_rule_ids") or []):
        raise TasteGlobalGCEFullError("T13 selected-rule bytes/order changed")
    threshold = ThresholdContract(
        values=tuple(float(value) for value in selection["thresholds"]),
        theta_star=float(selection["theta_star"]),
        cost_cap=float(selection["cost_cap"]),
        config_hash=str(selection["threshold_config_hash"]),
        source=str(selection["threshold_source"]),
        source_split=str(selection["threshold_source_split"]),
        file_sha256=str(selection["threshold_contract_file_sha256"]),
    )
    if threshold.config_hash != stable_json_sha256(list(threshold.values)):
        raise TasteGlobalGCEFullError("T13 verifier threshold grid hash changed")
    recomputed = compute_standardized_metrics(test_rows, ordered, threshold)
    expected_csvs = {
        "figure3_coverage_vs_k.csv": recomputed["figure3"],
        "figure4_coverage_vs_threshold.csv": recomputed["figure4"],
        "prefix_metrics.csv": recomputed["prefix"],
        "parent_best_distances.csv": recomputed["parent_best"],
        "destination_distribution.csv": recomputed["destination"],
        "table2_globalgce_k10.csv": recomputed["table2"],
    }
    for name, expected in expected_csvs.items():
        # CSV round-tripping changes scalar types.  Compare the canonical CSV
        # bytes produced by the same writer in a private temporary directory.
        with tempfile.TemporaryDirectory(prefix="t13-verify-") as temporary:
            candidate = Path(temporary) / name
            atomic_csv(candidate, expected)
            if candidate.read_bytes() != (output / name).read_bytes():
                raise TasteGlobalGCEFullError(f"T13 standardized replay differs: {name}")
    if read_json_value(output / "prefix_metrics.json") != recomputed["prefix"]:
        raise TasteGlobalGCEFullError("T13 prefix JSON replay differs")
    checks = {
        "frozen_inventory_reloaded": True,
        "official_branch_checkpoint_artifacts_present": True,
        "both_target_branches_present": True,
        "canonical_rule_dedup_replayed": True,
        "calibration_only_selector": True,
        "selection_frozen_before_test": True,
        "held_out_test_cartesian_complete": True,
        "standardized_metrics_recomputed": True,
        "same_gine_identity": True,
        "rf_oracle_absent": True,
        "destination_labels_0_or_2": True,
    }
    common = {
        key: run_manifest[key]
        for key in (
            "dataset",
            "method",
            "stage",
            "oracle_backend",
            "classifier_family",
            "rf_oracle_used",
            "oracle_checkpoint",
            "oracle_hash",
            "oracle_checkpoint_hash",
            "dataset_hash",
            "test_parent_ids_sha256",
            "test_split_hash",
            "distance_line",
            "molclr_checkpoint_hash",
            "cf_mode",
            "threshold_config_hash",
            "test_used_for_selection",
            "threshold_fitted_on_test",
            "raw_output_root",
        )
    }
    audit = {
        "schema_version": VERIFY_SCHEMA,
        **common,
        "status": "PASS",
        "passed": True,
        "audit_passed": True,
        "frozen": True,
        "artifacts_frozen": True,
        "raw_output_complete": True,
        "raw_artifacts_complete": True,
        "checks": checks,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "verified_at": utc_now(),
    }
    atomic_json(output / "final_artifact_audit.json", audit)
    run_manifest.update(
        {
            "status": "PASS",
            "state": "PASS",
            "run_complete": True,
            "frozen": True,
            "finalized": True,
            "raw_output_complete": True,
            "source_artifacts_complete": True,
            "independent_terminal_verification_required": False,
            "worker_wrote_pass": False,
            "terminal_verifier": "separate_verify_only_invocation",
            "final_artifact_audit_sha256": sha256_file(
                output / "final_artifact_audit.json"
            ),
            "completed_at": utc_now(),
        }
    )
    atomic_json(output / "run_manifest.json", run_manifest)
    registry = audit_explicit_candidate(output, dataset=DATASET, method=METHOD)
    if registry.status not in PASS_STATUSES:
        audit["status"] = "FAILED"
        audit["passed"] = False
        audit["audit_passed"] = False
        audit["registry_status"] = registry.status.value
        audit["registry_reason_codes"] = registry.reason_codes
        atomic_json(output / "final_artifact_audit.json", audit)
        run_manifest["status"] = "FAILED"
        run_manifest["state"] = "FAILED"
        run_manifest["run_complete"] = False
        atomic_json(output / "run_manifest.json", run_manifest)
        _atomic_bytes(output / "FAILED", b"FAILED\n")
        raise TasteGlobalGCEFullError(
            "T13 registry gate failed: " + ";".join(registry.reason_codes)
        )
    audit["registry_status"] = registry.status.value
    audit["registry_reason_codes"] = []
    atomic_json(output / "final_artifact_audit.json", audit)
    run_manifest["final_artifact_audit_sha256"] = sha256_file(
        output / "final_artifact_audit.json"
    )
    atomic_json(output / "run_manifest.json", run_manifest)
    _atomic_bytes(output / "PASS", b"PASS\n")
    checkpoint = read_json(_checkpoint_path(output))
    write_checkpoint(
        output,
        phase="PASS",
        resume_identity=checkpoint["resume_identity"],
        detail={"final_artifact_audit_sha256": sha256_file(output / "final_artifact_audit.json")},
    )
    return audit


__all__ = [
    "DATASET",
    "METHOD",
    "PASS_MARKER",
    "STAGE",
    "TasteGlobalGCEFullConfig",
    "TasteGlobalGCEFullError",
    "ThresholdContract",
    "authorize_and_load_test_after_freeze",
    "compute_standardized_metrics",
    "evaluate_one_parent",
    "evaluate_split_resumable",
    "load_input_authority",
    "load_prepared_split",
    "load_threshold_contract",
    "merge_branch_rules",
    "run_native_branch",
    "run_t13_full",
    "select_rules_on_calibration",
    "verify_t13_output",
]
