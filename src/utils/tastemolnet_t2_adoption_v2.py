"""Independent scientific adoption for the completed TasteMolNet T2 GINE.

The historical controller remains failed.  This verifier authenticates the
immutable scientific bundle, replays the validation predictions from the
frozen validation tensor cache, and publishes a new receipt under the v2 main
controller namespace.  It never repairs or rewrites historical evidence.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from src.data.molecular_graph_dataset import load_molecular_graph_cache
from src.data.molecular_graph_featurizer import MolecularFeatureSchema
from src.oracles.gnn_oracle import (
    classification_metrics,
    load_gnn_checkpoint_payloads,
)


SCHEMA_VERSION = "tastemolnet_t2_scientific_adoption_v2"
RECEIPT_NAMESPACE = Path("tastemolnet-main-v2/adoptions/T2_GINE")
PASS_MARKER = "[TASTE_T2_GINE_ADOPTION_PASS]"
EXPECTED_LABEL_MAP = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
EXPECTED_SOURCE_COMMIT = "583bf668896142d8cc292cd624fbbffc20faf688"
EXPECTED_FAILURE_REASON = "WORKER_PROCESS_IDENTITY_DRIFT"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_BUNDLE_FILES = frozenset(
    {
        "checkpoint_reload.json",
        "config.yaml",
        "data_use_policy_binding.json",
        "environment.json",
        "feature_schema.json",
        "git_state.json",
        "graph_cache_usage.json",
        "label_map.json",
        "last.pt",
        "last_checkpoint.json",
        "model.pt",
        "model_card.json",
        "oracle_manifest.json",
        "sha256sums.txt",
        "split_manifest.json",
        "temperature_scaling.json",
        "test_evaluation_status.json",
        "training_metrics.json",
        "validation_predictions.csv",
    }
)
CHECKPOINT_PAYLOAD_NAMES = frozenset(
    {
        "model.pt",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "test_evaluation_status.json",
        "temperature_scaling.json",
    }
)


class TasteT2AdoptionError(RuntimeError):
    """The historical T2 result cannot be independently adopted."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _absolute(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise TasteT2AdoptionError(f"{label} must be absolute")
    return Path(os.path.normpath(str(raw)))


def _file_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
    }


class HeldFile:
    """One physical single-link regular file held across verification."""

    def __init__(self, path: str | Path, *, label: str) -> None:
        self.path = _absolute(path, label=label)
        self.label = label
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.path, flags)
        info = os.fstat(self.fd)
        named = os.stat(self.path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(self.fd)
            raise TasteT2AdoptionError(f"{label} is not one physical regular file")
        self.identity = _file_identity(info)
        self.sha256 = self._hash()

    def _hash(self) -> str:
        digest = hashlib.sha256()
        os.lseek(self.fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(self.fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(self.fd, 0, os.SEEK_SET)
        return digest.hexdigest()

    def bytes(self) -> bytes:
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self.fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        os.lseek(self.fd, 0, os.SEEK_SET)
        data = b"".join(chunks)
        self.verify()
        return data

    def json(self) -> dict[str, Any]:
        try:
            value = json.loads(self.bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteT2AdoptionError(f"{self.label} is not a JSON object") from exc
        if type(value) is not dict:
            raise TasteT2AdoptionError(f"{self.label} is not a JSON object")
        return value

    def verify(self) -> None:
        info = os.fstat(self.fd)
        try:
            named = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise TasteT2AdoptionError(f"{self.label} disappeared") from exc
        if (
            _file_identity(info) != self.identity
            or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
            or self._hash() != self.sha256
        ):
            raise TasteT2AdoptionError(f"{self.label} changed while held")

    def stream(self) -> io.BufferedReader:
        self.verify()
        return os.fdopen(os.dup(self.fd), "rb", closefd=True)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class HeldBundle:
    """Descriptor-bound exact inventory of the historical scientific bundle."""

    def __init__(self, root: str | Path) -> None:
        self.root = _absolute(root, label="scientific artifact root")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        self.fd = os.open(self.root, flags)
        info = os.fstat(self.fd)
        named = os.stat(self.root, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            os.close(self.fd)
            raise TasteT2AdoptionError("scientific artifact root is not physical")
        self.identity = _file_identity(info)
        names = {entry.name for entry in os.scandir(self.fd)}
        if names != REQUIRED_BUNDLE_FILES:
            raise TasteT2AdoptionError(
                "scientific bundle inventory changed: "
                f"missing={sorted(REQUIRED_BUNDLE_FILES - names)} "
                f"unexpected={sorted(names - REQUIRED_BUNDLE_FILES)}"
            )
        self.files: dict[str, HeldFile] = {
            name: HeldFile(self.root / name, label=f"scientific bundle {name}")
            for name in sorted(names)
        }

    def verify(self) -> None:
        info = os.fstat(self.fd)
        named = os.stat(self.root, follow_symlinks=False)
        if (
            _file_identity(info) != self.identity
            or (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino)
            or {entry.name for entry in os.scandir(self.fd)} != set(self.files)
        ):
            raise TasteT2AdoptionError("scientific artifact root changed while held")
        for held in self.files.values():
            held.verify()

    def inventory(self) -> dict[str, Any]:
        return {
            name: {"sha256": held.sha256, "identity": held.identity}
            for name, held in sorted(self.files.items())
        }

    def close(self) -> None:
        for held in self.files.values():
            held.close()
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _parse_sha256s(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TasteT2AdoptionError("sha256sums.txt is not UTF-8") from exc
    for line in lines:
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not SHA256_RE.fullmatch(digest)
            or Path(name).name != name
            or name in result
        ):
            raise TasteT2AdoptionError("sha256sums.txt is malformed")
        result[name] = digest
    expected = REQUIRED_BUNDLE_FILES - {"sha256sums.txt"}
    if set(result) != expected:
        raise TasteT2AdoptionError("sha256sums.txt does not close the exact bundle")
    return result


def _load_config(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            import yaml

            value = yaml.safe_load(data.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - AutoDL has PyYAML.
            raise TasteT2AdoptionError("config.yaml cannot be decoded safely") from exc
    if type(value) is not dict:
        raise TasteT2AdoptionError("config.yaml must contain one mapping")
    return value


def _finite(value: Any, *, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TasteT2AdoptionError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TasteT2AdoptionError(f"{label} is not finite")
    return number


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _audit_historical_temperature(
    payload: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate old validation-only evidence without adopting it as T3."""

    if (
        payload.get("schema_version") != "temperature_scaling_v1"
        or payload.get("selection_split") != "validation"
        or payload.get("test_used_for_fit") is not False
        or payload.get("argmax_invariant") is not True
        or payload.get("num_classes") != 3
        or payload.get("num_examples") != len(stored["molecule_ids"])
    ):
        raise TasteT2AdoptionError("historical temperature evidence changed")
    scalar = _finite(payload.get("temperature"), label="historical temperature")
    if scalar <= 0:
        raise TasteT2AdoptionError("historical temperature is not positive")
    logits = np.asarray(stored["logits"], dtype=np.float64)
    labels = np.asarray(stored["labels"], dtype=np.int64)
    before = _softmax(logits)
    shifted = logits / scalar
    shifted -= shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    after = exponentials / exponentials.sum(axis=1, keepdims=True)
    if not np.array_equal(before.argmax(axis=1), after.argmax(axis=1)):
        raise TasteT2AdoptionError("historical temperature changed argmax")
    before_metrics = classification_metrics(labels, before, num_classes=3)
    after_metrics = classification_metrics(labels, after, num_classes=3)
    selected_before = before[np.arange(labels.size), labels]
    selected_after = after[np.arange(labels.size), labels]
    recomputed = {
        "nll_before": float(-np.log(np.clip(selected_before, 1e-300, 1.0)).mean()),
        "nll_after": float(-np.log(np.clip(selected_after, 1e-300, 1.0)).mean()),
        "ece_before": float(before_metrics["ece"]),
        "ece_after": float(after_metrics["ece"]),
        "brier_before": float(before_metrics["brier_score"]),
        "brier_after": float(after_metrics["brier_score"]),
    }
    for key, observed in recomputed.items():
        expected = _finite(payload.get(key), label=f"historical temperature {key}")
        if not math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-10):
            raise TasteT2AdoptionError(f"historical temperature {key} changed")
    return {
        "status": "PASS",
        "role": "authenticated_historical_evidence_not_T3_publication",
        "temperature": scalar,
        "selection_split": "validation",
        "test_used_for_fit": False,
        "recomputed_metrics": recomputed,
        "fresh_T3_refit_still_required": True,
    }


def _read_validation_predictions(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TasteT2AdoptionError("validation predictions are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {
        "molecule_id",
        "smiles",
        "split",
        "label",
        "predicted_label",
        "logits",
        "probabilities",
        "source_graph_hash",
    }
    if not required.issubset(reader.fieldnames or ()):
        raise TasteT2AdoptionError("validation prediction schema changed")
    rows = [dict(row) for row in reader]
    if not rows:
        raise TasteT2AdoptionError("validation prediction file is empty")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    logits: list[list[float]] = []
    probabilities: list[list[float]] = []
    graph_hashes: list[str] = []
    for index, row in enumerate(rows):
        molecule_id = str(row.get("molecule_id") or "")
        graph_hash = str(row.get("source_graph_hash") or "")
        if not molecule_id or not SHA256_RE.fullmatch(graph_hash):
            raise TasteT2AdoptionError(f"validation row {index} identity is malformed")
        if str(row.get("split", "")).strip().lower() not in {"val", "validation"}:
            raise TasteT2AdoptionError(f"validation row {index} has another split")
        try:
            label = int(str(row["label"]))
            prediction = int(str(row["predicted_label"]))
            row_logits = json.loads(row["logits"])
            row_probabilities = json.loads(row["probabilities"])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise TasteT2AdoptionError(f"validation row {index} is malformed") from exc
        if label not in range(3) or prediction not in range(3):
            raise TasteT2AdoptionError(f"validation row {index} label is invalid")
        if not isinstance(row_logits, list) or not isinstance(row_probabilities, list):
            raise TasteT2AdoptionError(f"validation row {index} vectors are malformed")
        if len(row_logits) != 3 or len(row_probabilities) != 3:
            raise TasteT2AdoptionError(f"validation row {index} vectors are not 3-wide")
        parsed_logits = [_finite(value, label=f"validation[{index}].logit") for value in row_logits]
        parsed_probabilities = [
            _finite(value, label=f"validation[{index}].probability")
            for value in row_probabilities
        ]
        if prediction != int(np.argmax(parsed_probabilities)):
            raise TasteT2AdoptionError(f"validation row {index} predicted class changed")
        ids.append(molecule_id)
        labels.append(label)
        predictions.append(prediction)
        logits.append(parsed_logits)
        probabilities.append(parsed_probabilities)
        graph_hashes.append(graph_hash)
    if len(set(ids)) != len(ids):
        raise TasteT2AdoptionError("validation molecule IDs are duplicated")
    logit_array = np.asarray(logits, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if not np.allclose(_softmax(logit_array), probability_array, rtol=0.0, atol=1e-7):
        raise TasteT2AdoptionError("validation probabilities do not reproduce from logits")
    label_array = np.asarray(labels, dtype=np.int64)
    metrics = classification_metrics(label_array, probability_array, num_classes=3)
    predicted_classes = sorted(set(predictions))
    if predicted_classes != [0, 1, 2]:
        raise TasteT2AdoptionError("validation predictions collapsed to fewer than 3 classes")
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping):
        raise TasteT2AdoptionError("validation per-class metrics are absent")
    for label in ("0", "1", "2"):
        if _finite(per_class.get(label, {}).get("recall"), label=f"recall[{label}]") <= 0:
            raise TasteT2AdoptionError("one validation class has zero recall")
    for name in ("macro_f1", "macro_ovr_roc_auc"):
        _finite(metrics.get(name), label=name)
    return {
        "rows": rows,
        "molecule_ids": ids,
        "labels": label_array,
        "logits": logit_array,
        "probabilities": probability_array,
        "graph_hashes": graph_hashes,
        "metrics": metrics,
        "predicted_classes": predicted_classes,
        "row_ids_sha256": _stable_sha256(ids),
    }


def _git_commit_trace(project_root: Path, source_commit: str) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(source_commit):
        raise TasteT2AdoptionError("source Git commit is malformed")
    command = ["git", "-C", str(project_root), "cat-file", "-e", f"{source_commit}^{{commit}}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise TasteT2AdoptionError("source Git commit is unavailable to verifier")
    verifier = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise TasteT2AdoptionError("independent verifier checkout is not clean")
    if not COMMIT_RE.fullmatch(verifier):
        raise TasteT2AdoptionError("independent verifier commit is malformed")
    return {"source_commit_present": True, "independent_verifier_commit": verifier}


def _verify_historical_state(
    controller_root: Path,
    training_state_root: Path,
) -> tuple[list[HeldFile], dict[str, Any]]:
    held = [
        HeldFile(controller_root / "controller_state.json", label="historical controller state"),
        HeldFile(controller_root / "controller_spec.json", label="historical controller spec"),
        HeldFile(training_state_root / "training_complete.json", label="historical training completion"),
    ]
    state = held[0].json()
    phase = state.get("phase", state.get("state"))
    if phase != "FAILED" or state.get("reason") != EXPECTED_FAILURE_REASON:
        for item in held:
            item.close()
        raise TasteT2AdoptionError("historical controller is not the retained identity-drift failure")
    completion = held[2].json()
    if completion.get("status") != "PASS":
        for item in held:
            item.close()
        raise TasteT2AdoptionError("historical training completion is not PASS")
    return held, {
        "old_terminal_state": "FAILED",
        "old_failure_reason": EXPECTED_FAILURE_REASON,
        "controller_state_sha256": held[0].sha256,
        "controller_spec_sha256": held[1].sha256,
        "training_complete_sha256": held[2].sha256,
    }


def _hold_declared_inputs(
    split_manifest: Mapping[str, Any],
    cache_usage: Mapping[str, Any],
) -> tuple[list[HeldFile], dict[str, Any], HeldFile]:
    held: list[HeldFile] = []
    split_hashes: dict[str, Any] = {}
    files = split_manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {"train", "validation", "calibration", "test"}:
        raise TasteT2AdoptionError("split manifest file set changed")
    for split in ("train", "validation", "calibration", "test"):
        entry = files.get(split)
        if not isinstance(entry, Mapping):
            raise TasteT2AdoptionError(f"split manifest {split} entry is malformed")
        item = HeldFile(str(entry.get("path") or ""), label=f"declared {split} split")
        if item.sha256 != entry.get("sha256"):
            item.close()
            raise TasteT2AdoptionError(f"declared {split} split hash changed")
        held.append(item)
        split_hashes[split] = {"path": str(item.path), "sha256": item.sha256}

    cache_contract = cache_usage.get("cache_contract")
    cache_files = cache_usage.get("cache_files")
    if (
        cache_usage.get("loaded_splits") != ["train", "validation"]
        or cache_usage.get("calibration_loaded") is not False
        or cache_usage.get("test_loaded") is not False
        or cache_usage.get("graph_cache_used") is not True
        or not isinstance(cache_contract, Mapping)
        or not isinstance(cache_files, Mapping)
    ):
        raise TasteT2AdoptionError("graph-cache data boundary changed")
    manifest_entry = cache_contract.get("manifest")
    if not isinstance(manifest_entry, Mapping):
        raise TasteT2AdoptionError("graph-cache manifest authority is absent")
    cache_manifest = HeldFile(
        str(manifest_entry.get("path") or ""), label="declared graph-cache manifest"
    )
    if cache_manifest.sha256 != manifest_entry.get("sha256"):
        cache_manifest.close()
        raise TasteT2AdoptionError("graph-cache manifest hash changed")
    held.append(cache_manifest)
    cache_hashes: dict[str, Any] = {
        "manifest": {"path": str(cache_manifest.path), "sha256": cache_manifest.sha256}
    }
    validation_cache: HeldFile | None = None
    for split in ("train", "validation"):
        entry = cache_files.get(split)
        if not isinstance(entry, Mapping):
            raise TasteT2AdoptionError(f"graph-cache {split} entry is absent")
        item = HeldFile(str(entry.get("path") or ""), label=f"declared {split} graph cache")
        if item.sha256 != entry.get("sha256"):
            item.close()
            raise TasteT2AdoptionError(f"declared {split} graph cache hash changed")
        held.append(item)
        cache_hashes[split] = {"path": str(item.path), "sha256": item.sha256}
        if split == "validation":
            validation_cache = item
    if validation_cache is None:  # pragma: no cover - guarded above.
        raise TasteT2AdoptionError("validation cache authority is absent")
    return held, {"split_files": split_hashes, "graph_cache": cache_hashes}, validation_cache


def _reproduce_validation(
    bundle: HeldBundle,
    validation_cache: HeldFile,
    stored: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    payloads = {name: bundle.files[name].bytes() for name in CHECKPOINT_PAYLOAD_NAMES}
    model, metadata = load_gnn_checkpoint_payloads(payloads, device=device)
    feature_schema: MolecularFeatureSchema = metadata["feature_schema"]
    split_manifest = metadata["split_manifest"]
    validation_manifest = split_manifest.get("validation_manifest")
    if not isinstance(validation_manifest, Mapping):
        raise TasteT2AdoptionError("validation manifest is absent")
    with validation_cache.stream() as handle:
        dataset = load_molecular_graph_cache(
            handle,
            expected_num_classes=3,
            expected_source_sha256=str(validation_manifest.get("source_sha256")),
            expected_feature_schema=feature_schema,
        )
    if len(dataset) != len(stored["molecule_ids"]):
        raise TasteT2AdoptionError("validation cache and prediction counts differ")
    if [dataset[index].molecule_id for index in range(len(dataset))] != list(
        stored["molecule_ids"]
    ):
        raise TasteT2AdoptionError("validation cache molecule order changed")
    if [dataset[index].graph_sha256 for index in range(len(dataset))] != list(
        stored["graph_hashes"]
    ):
        raise TasteT2AdoptionError("validation cache graph hashes changed")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL gate owns torch.
        raise TasteT2AdoptionError("PyTorch is unavailable to verifier") from exc
    model.eval()
    observed: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(len(dataset), start + batch_size))]
            batch = dataset.collate(rows).to(device)
            observed.append(model(batch).detach().cpu().numpy().astype(np.float64))
    observed_logits = np.concatenate(observed, axis=0)
    expected_logits = np.asarray(stored["logits"], dtype=np.float64)
    if observed_logits.shape != expected_logits.shape:
        raise TasteT2AdoptionError("replayed validation logit shape changed")
    max_logit_difference = float(np.max(np.abs(observed_logits - expected_logits)))
    observed_probabilities = _softmax(observed_logits)
    expected_probabilities = np.asarray(stored["probabilities"], dtype=np.float64)
    max_probability_difference = float(
        np.max(np.abs(observed_probabilities - expected_probabilities))
    )
    if max_logit_difference > 2e-4 or max_probability_difference > 2e-5:
        raise TasteT2AdoptionError(
            "validation predictions do not reproduce within frozen numeric tolerance"
        )
    if not np.array_equal(
        observed_probabilities.argmax(axis=1), expected_probabilities.argmax(axis=1)
    ):
        raise TasteT2AdoptionError("replayed validation classes changed")
    return {
        "status": "PASS",
        "device": str(device),
        "num_examples": len(dataset),
        "batch_size": int(batch_size),
        "max_logit_abs_difference": max_logit_difference,
        "max_probability_abs_difference": max_probability_difference,
        "logit_atol": 2e-4,
        "probability_atol": 2e-5,
        "predicted_classes_exact": True,
    }


def _write_exclusive(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise TasteT2AdoptionError("short write while publishing adoption receipt")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists():
        raise TasteT2AdoptionError("adoption receipt UUID unexpectedly exists")
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            result = renameat2(
                ctypes.c_int(-100),
                ctypes.c_char_p(os.fsencode(source)),
                ctypes.c_int(-100),
                ctypes.c_char_p(os.fsencode(destination)),
                ctypes.c_uint(1),
            )
            if result == 0:
                return
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise TasteT2AdoptionError("adoption receipt UUID was reused")
            raise OSError(code, os.strerror(code), str(destination))
    os.rename(source, destination)


def _publish_receipt(
    *,
    control_root: Path,
    receipt_id: str,
    documents: Mapping[str, Mapping[str, Any]],
    source_holds: Sequence[HeldFile],
    bundle: HeldBundle,
) -> Path:
    parent = control_root / RECEIPT_NAMESPACE
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.resolve() != parent or control_root.resolve() not in parent.parents:
        raise TasteT2AdoptionError("adoption namespace escapes the physical control root")
    staging = parent / f".{receipt_id}.staging"
    final = parent / receipt_id
    os.mkdir(staging, 0o700)
    try:
        for name, payload in sorted(documents.items()):
            _write_exclusive(
                staging / name,
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
                + b"\n",
            )
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.iterdir())
        }
        _write_exclusive(
            staging / "sha256s.txt",
            "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode(
                "utf-8"
            ),
        )
        _write_exclusive(staging / "PASS", (PASS_MARKER + "\n").encode("utf-8"))
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        bundle.verify()
        for held in source_holds:
            held.verify()
        _rename_noreplace(staging, final)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final
    except Exception:
        # A UUID staging directory is retained on failure for manual forensics;
        # it is never renamed to PASS and never reused.
        raise


def adopt_tastemolnet_t2_scientific_result(
    *,
    control_root: str | Path,
    artifact_root: str | Path,
    controller_root: str | Path,
    training_state_root: str | Path,
    project_root: str | Path,
    source_run_id: str,
    source_controller_id: str,
    expected_source_commit: str = EXPECTED_SOURCE_COMMIT,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Verify and atomically publish one fresh T2 scientific adoption receipt."""

    if not source_run_id or not source_controller_id:
        raise TasteT2AdoptionError("source run/controller IDs are required")
    if type(batch_size) is not int or batch_size <= 0:
        raise TasteT2AdoptionError("batch_size must be one positive native integer")
    control = _absolute(control_root, label="control root")
    project = _absolute(project_root, label="independent verifier project root")
    controller = _absolute(controller_root, label="historical controller root")
    training = _absolute(training_state_root, label="historical training state root")
    bundle = HeldBundle(artifact_root)
    historical_holds: list[HeldFile] = []
    input_holds: list[HeldFile] = []
    try:
        sha256s = _parse_sha256s(bundle.files["sha256sums.txt"].bytes())
        for name, expected in sha256s.items():
            if bundle.files[name].sha256 != expected:
                raise TasteT2AdoptionError(f"scientific bundle hash mismatch: {name}")

        model_card = bundle.files["model_card.json"].json()
        oracle = bundle.files["oracle_manifest.json"].json()
        cache_usage = bundle.files["graph_cache_usage.json"].json()
        split_manifest = bundle.files["split_manifest.json"].json()
        metrics = bundle.files["training_metrics.json"].json()
        policy = bundle.files["data_use_policy_binding.json"].json()
        reload_receipt = bundle.files["checkpoint_reload.json"].json()
        test_status = bundle.files["test_evaluation_status.json"].json()
        git_state = bundle.files["git_state.json"].json()
        config = _load_config(bundle.files["config.yaml"].bytes())
        label_map = bundle.files["label_map.json"].json()

        gnn_config = config.get("gnn")
        training_config = config.get("training")
        health = model_card.get("health_gate")
        if (
            model_card.get("dataset") != "tastemolnet"
            or model_card.get("backbone") != "gine"
            or model_card.get("num_classes") != 3
            or model_card.get("seed") != 7
            or model_card.get("source_label") != 1
            or model_card.get("oracle_backend") != "gnn"
            or model_card.get("rf_oracle_used") is not False
            or model_card.get("test_loaded_during_training") is not False
            or model_card.get("calibration_used_for_model_fit_or_selection") is not False
            or not isinstance(health, Mapping)
            or health.get("status") != "PASS"
            or health.get("predicted_classes") != [0, 1, 2]
            or not isinstance(gnn_config, Mapping)
            or gnn_config.get("backbone") != "gine"
            or gnn_config.get("num_classes") != 3
            or not isinstance(training_config, Mapping)
            or training_config.get("primary_seed") != 7
            or label_map != EXPECTED_LABEL_MAP
        ):
            raise TasteT2AdoptionError("Taste three-class GINE contract changed")
        if (
            oracle.get("status") != "PASS"
            or oracle.get("label_map") != EXPECTED_LABEL_MAP
            or oracle.get("num_classes") != 3
            or oracle.get("source_label") != 1
            or oracle.get("rf_oracle_used") is not False
            or oracle.get("test_loaded") is not False
            or oracle.get("test_evaluated") is not False
            or reload_receipt.get("checkpoint_reload_pass") is not True
            or reload_receipt.get("batch_single_probability_equivalence") is not True
            or reload_receipt.get("all_probabilities_finite") is not True
            or test_status.get("status") != "NOT_EVALUATED"
            or test_status.get("test_loaded") is not False
        ):
            raise TasteT2AdoptionError("Taste oracle/checkpoint/test closure changed")
        if (
            policy.get("data_redistribution_allowed") is not False
            or policy.get("dataset_redistributed") is not False
            or policy.get("hpc_execution_authorized") is not False
        ):
            raise TasteT2AdoptionError("Taste data-use policy changed")
        if model_card.get("checkpoint_id") != bundle.files["model.pt"].sha256:
            raise TasteT2AdoptionError("model SHA-256 differs from model card")
        feature_schema = MolecularFeatureSchema.from_dict(
            bundle.files["feature_schema.json"].json()
        )
        if model_card.get("feature_schema_sha256") != feature_schema.to_dict()[
            "schema_sha256"
        ]:
            raise TasteT2AdoptionError("feature schema differs from model card")
        if git_state.get("commit") != expected_source_commit or model_card.get(
            "training_commit"
        ) != expected_source_commit:
            raise TasteT2AdoptionError("scientific bundle source commit changed")

        final_validation = metrics.get("final_validation")
        if not isinstance(final_validation, Mapping):
            raise TasteT2AdoptionError("final validation metrics are absent")
        for name in ("macro_f1", "macro_ovr_roc_auc"):
            _finite(final_validation.get(name), label=f"final_validation.{name}")
        for label in ("0", "1", "2"):
            recall = final_validation.get("per_class", {}).get(label, {}).get("recall")
            if _finite(recall, label=f"final_validation.recall[{label}]") <= 0:
                raise TasteT2AdoptionError("one recorded validation class has zero recall")

        stored_predictions = _read_validation_predictions(
            bundle.files["validation_predictions.csv"].bytes()
        )
        if len(stored_predictions["rows"]) != split_manifest.get(
            "validation_manifest", {}
        ).get("num_records"):
            raise TasteT2AdoptionError("validation prediction count changed")
        for name in ("macro_f1", "macro_ovr_roc_auc"):
            if not math.isclose(
                float(stored_predictions["metrics"][name]),
                float(final_validation[name]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise TasteT2AdoptionError(f"validation {name} no longer reproduces")

        historical_holds, historical = _verify_historical_state(controller, training)
        input_holds, input_hashes, validation_cache = _hold_declared_inputs(
            split_manifest, cache_usage
        )
        trace = _git_commit_trace(project, expected_source_commit)
        replay = _reproduce_validation(
            bundle,
            validation_cache,
            stored_predictions,
            device=device,
            batch_size=batch_size,
        )
        temperature_audit = _audit_historical_temperature(
            bundle.files["temperature_scaling.json"].json(), stored_predictions
        )
        bundle.verify()
        for held in [*historical_holds, *input_holds]:
            held.verify()

        receipt_id = str(uuid.uuid4())
        artifact_inventory = bundle.inventory()
        artifact_hashes = {name: row["sha256"] for name, row in artifact_inventory.items()}
        source_evidence = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": receipt_id,
            "source_run_id": source_run_id,
            "source_controller_id": source_controller_id,
            **historical,
            "artifact_root": str(bundle.root),
            "artifact_hashes": artifact_hashes,
            "artifact_inventory_sha256": _stable_sha256(artifact_inventory),
            "input_hashes": input_hashes,
            "git_commit": expected_source_commit,
            "config_hash": bundle.files["config.yaml"].sha256,
            "feature_schema_sha256": bundle.files["feature_schema.json"].sha256,
            "validation_row_ids_sha256": stored_predictions["row_ids_sha256"],
            "independent_verifier_commit": trace["independent_verifier_commit"],
            "old_failure_superseded_for_scientific_artifact": True,
            "old_process_evidence_not_rewritten": True,
            "worker_identity_drift_polluted_scientific_files": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
        }
        source_evidence["source_evidence_sha256"] = _stable_sha256(source_evidence)
        verification = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": receipt_id,
            "verified_at": _utc_timestamp(),
            "verification_result": "PASS",
            "source_evidence_sha256": source_evidence["source_evidence_sha256"],
            "bundle_hashes_pass": True,
            "three_class_contract_pass": True,
            "validation_metrics_pass": True,
            "validation_replay": replay,
            "temperature_evidence_audit": temperature_audit,
            "source_commit_trace": trace,
        }
        gate = {
            "schema_version": SCHEMA_VERSION,
            "stage": "T2_GINE",
            "state": "ADOPTED_SCIENTIFIC_PASS",
            "status": "PASS",
            "receipt_id": receipt_id,
            "artifact_root": str(bundle.root),
            "model_sha256": bundle.files["model.pt"].sha256,
            "verification_sha256": _stable_sha256(verification),
            "source_evidence_sha256": source_evidence["source_evidence_sha256"],
            "marker": PASS_MARKER,
            "downstream_released": True,
        }
        documents = {
            "artifact_hashes.json": {
                "schema_version": SCHEMA_VERSION,
                "artifact_root": str(bundle.root),
                "artifact_hashes": artifact_hashes,
            },
            "input_hashes.json": {"schema_version": SCHEMA_VERSION, **input_hashes},
            "source_evidence.json": source_evidence,
            "verification.json": verification,
            "gate.json": gate,
        }
        receipt_root = _publish_receipt(
            control_root=control,
            receipt_id=receipt_id,
            documents=documents,
            source_holds=[*historical_holds, *input_holds],
            bundle=bundle,
        )
        return {
            "status": "PASS",
            "state": "ADOPTED_SCIENTIFIC_PASS",
            "receipt_id": receipt_id,
            "receipt_root": str(receipt_root),
            "artifact_root": str(bundle.root),
            "model_sha256": bundle.files["model.pt"].sha256,
            "source_evidence_sha256": source_evidence["source_evidence_sha256"],
            "independent_verifier_commit": trace["independent_verifier_commit"],
            "marker": PASS_MARKER,
        }
    finally:
        for held in [*historical_holds, *input_holds]:
            held.close()
        bundle.close()


__all__ = [
    "EXPECTED_FAILURE_REASON",
    "EXPECTED_LABEL_MAP",
    "EXPECTED_SOURCE_COMMIT",
    "PASS_MARKER",
    "RECEIPT_NAMESPACE",
    "TasteT2AdoptionError",
    "adopt_tastemolnet_t2_scientific_result",
]
