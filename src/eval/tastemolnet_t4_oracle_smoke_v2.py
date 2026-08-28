"""Managed-v2 successor for the bounded TasteMolNet T4 oracle smoke.

The scientific worker writes only aggregate candidate evidence.  A distinct
method verifier reopens the worker's SEALED tree, independently repeats the
bounded calibration-cache smoke on the same physical GPU binding, and delegates
the only PASS/gate writes to :mod:`src.utils.terminal_publisher_v2`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Callable, Mapping

from src.data.molecular_graph_dataset import (
    MolecularGraphDataset,
    load_molecular_graph_cache,
)
from src.data.molecular_graph_featurizer import MolecularFeatureSchema
from src.eval.tastemolnet_gnn_stages import (
    NUM_CLASSES,
    SOURCE_LABEL,
    PhysicalDirectory,
    TasteGNNStageError,
    _load_gnn_oracle_anchored,
    _physical_directory,
    run_bounded_oracle_smoke,
)
from src.eval.tastemolnet_t3_calibration_v2 import (
    CANDIDATE_CHECKPOINT_FILES as T3_CHECKPOINT_FILES,
    CANDIDATE_NAME as T3_CANDIDATE_NAME,
    PASS_MARKER as T3_SCIENCE_MARKER,
    SCHEMA_VERSION as T3_SCIENCE_SCHEMA,
    STAGE as T3_STAGE,
)
from src.utils.managed_execution_v2 import (
    GATE_SCHEMA,
    VERIFICATION_SCHEMA,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
)
from src.utils.autodl_tastemolnet_main_v2 import (
    DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    TasteMainV2AuthorityError,
    hold_taste_main_v2_controller_authority,
    probe_physical_gpus,
)
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


SCHEMA_VERSION = "tastemolnet_t4_oracle_smoke_v2"
STAGE = "T4_ORACLE_SMOKE"
TASK_ID = "T4_ORACLE_SMOKE"
PASS_MARKER = "[TASTE_T4_ORACLE_SMOKE_PASS]"
PHYSICAL_GPU_INDEX = 1
VISIBLE_DEVICE = "cuda:0"
CUDA_VISIBLE_DEVICES = "1"
SOURCE_COUNT = 16
DELETIONS_PER_PARENT = 4
PUBLISHED_T3_ROOT = Path(
    "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/gnn_oracles/"
    "tastemolnet/gine/seed7/calibrated-20260828T054900Z-746545ed"
)
T4_CANDIDATE_NAME = "t4_oracle_smoke_candidate.json"
METHOD_DOCUMENTS = frozenset(
    {
        "oracle_smoke.json",
        "oracle_provenance.json",
        "data_access_manifest.json",
        "t3_binding.json",
    }
)
CANDIDATE_FILES = frozenset(
    set(METHOD_DOCUMENTS) | {T4_CANDIDATE_NAME, "sha256sums.txt"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


class TasteT4OracleSmokeError(RuntimeError):
    """The T4 authority, science, candidate, or independent check failed."""


def _require_published_t3_root(path: str | Path) -> Path:
    selected = Path(path)
    if selected != PUBLISHED_T3_ROOT:
        raise TasteT4OracleSmokeError(
            "T4 must consume the exact reviewed managed T3 publication"
        )
    return selected


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteT4OracleSmokeError(f"{label} is not valid JSON") from exc
    if type(value) is not dict:
        raise TasteT4OracleSmokeError(f"{label} must be one JSON object")
    return value


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise TasteT4OracleSmokeError(f"short write for {path.name}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


class _HeldFile:
    def __init__(self, path: Path, *, label: str) -> None:
        self.path = path
        self.label = label
        self.descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        info = os.fstat(self.descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(self.descriptor)
            raise TasteT4OracleSmokeError(f"{label} is not one physical file")
        self.identity = _identity(info)
        self.sha256 = self._hash()

    def _hash(self) -> str:
        digest = hashlib.sha256()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()

    def bytes(self) -> bytes:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            block = os.read(self.descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        result = b"".join(chunks)
        self.verify()
        return result

    def json(self) -> dict[str, Any]:
        return _json(self.bytes(), label=self.label)

    def verify(self) -> None:
        info = os.fstat(self.descriptor)
        named = os.stat(self.path, follow_symlinks=False)
        if (
            _identity(info) != self.identity
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            or self._hash() != self.sha256
        ):
            raise TasteT4OracleSmokeError(f"{self.label} changed while held")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _parse_sha256s(data: bytes, *, expected: set[str], label: str) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TasteT4OracleSmokeError(f"{label} is not UTF-8") from exc
    result: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or not _SHA256.fullmatch(digest)
            or Path(name).name != name
            or name in result
        ):
            raise TasteT4OracleSmokeError(f"{label} is malformed")
        result[name] = digest
    if set(result) != expected:
        raise TasteT4OracleSmokeError(f"{label} inventory changed")
    return result


class HeldPublishedT3:
    """Retain the exact published managed-v2 T3 root and checkpoint tree."""

    ROOT_FILES = frozenset(
        {
            ".generation_token.json",
            "raw_evidence.json",
            "worker_exit.json",
            "SEALED.json",
            "verification.json",
            "gate.json",
            "PASS",
        }
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.resolve(strict=True) != self.root:
            raise TasteT4OracleSmokeError("T3 root must be an exact physical path")
        if not self.root.name.startswith("calibrated-"):
            raise TasteT4OracleSmokeError("T3 root must be a calibrated-* publication")
        if self.root.parts[-4:-1] != ("tastemolnet", "gine", "seed7"):
            raise TasteT4OracleSmokeError(
                "T3 root must be the TasteMolNet GINE seed7 publication"
            )
        self.root_directory = _physical_directory(self.root, field="managed T3 root")
        self.artifact_directory = _physical_directory(
            self.root / "artifacts", field="managed T3 artifact root"
        )
        self.checkpoint_directory = _physical_directory(
            self.root / "artifacts/checkpoint", field="managed T3 checkpoint"
        )
        self.files: dict[str, _HeldFile] = {}
        try:
            self._require_inventory()
            for name in sorted(self.ROOT_FILES):
                self.files[name] = _HeldFile(self.root / name, label=f"T3 {name}")
            self.files["artifacts/.generation_token.json"] = _HeldFile(
                self.root / "artifacts/.generation_token.json",
                label="T3 artifact generation token",
            )
            for name in sorted(T3_CHECKPOINT_FILES):
                relative = f"artifacts/checkpoint/{name}"
                self.files[relative] = _HeldFile(
                    self.root / relative, label=f"T3 checkpoint {name}"
                )
            self.binding = self._validate()
        except BaseException:
            self.close()
            raise

    def _require_inventory(self) -> None:
        root_names = {entry.name for entry in os.scandir(self.root_directory.descriptor)}
        artifact_names = {
            entry.name for entry in os.scandir(self.artifact_directory.descriptor)
        }
        checkpoint_names = {
            entry.name for entry in os.scandir(self.checkpoint_directory.descriptor)
        }
        if root_names != self.ROOT_FILES | {"artifacts"}:
            raise TasteT4OracleSmokeError("published T3 root inventory changed")
        if artifact_names != {".generation_token.json", "checkpoint"}:
            raise TasteT4OracleSmokeError("published T3 artifact inventory changed")
        if checkpoint_names != set(T3_CHECKPOINT_FILES):
            raise TasteT4OracleSmokeError("published T3 checkpoint inventory changed")

    def _validate(self) -> dict[str, Any]:
        gate = self.files["gate.json"].json()
        verification = self.files["verification.json"].json()
        generation = self.files[".generation_token.json"].json()
        artifact_generation = self.files["artifacts/.generation_token.json"].json()
        seal = self.files["SEALED.json"].json()
        raw = self.files["raw_evidence.json"].json()
        worker_exit = self.files["worker_exit.json"].json()
        science = verification.get("verification")
        if type(science) is not dict:
            raise TasteT4OracleSmokeError("T3 scientific verification is absent")
        attempt_id = gate.get("attempt_id")
        generation_token = gate.get("generation_token")
        if (
            gate.get("schema_version") != GATE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("independent_verifier") is not True
            or gate.get("science_adopted") is not True
            or gate.get("downstream_released") is not True
            or gate.get("auto_terminate_uncontrolled_children") is not False
            or gate.get("verification_sha256") != self.files["verification.json"].sha256
            or gate.get("sealed_sha256") != self.files["SEALED.json"].sha256
            or verification.get("schema_version") != VERIFICATION_SCHEMA
            or verification.get("status") != "PASS"
            or verification.get("independent_verifier") is not True
            or verification.get("attempt_id") != attempt_id
            or verification.get("generation_token") != generation_token
            or verification.get("sealed_sha256") != gate.get("sealed_sha256")
            or verification.get("published_inventory_sha256")
            != gate.get("published_inventory_sha256")
            or generation.get("attempt_id") != attempt_id
            or generation.get("generation_token") != generation_token
            or artifact_generation.get("attempt_id") != attempt_id
            or artifact_generation.get("generation_token") != generation_token
            or seal.get("attempt_id") != attempt_id
            or seal.get("generation_token") != generation_token
            or raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
            or raw.get("attempt_id") != attempt_id
            or raw.get("generation_token") != generation_token
            or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
            or worker_exit.get("attempt_id") != attempt_id
            or worker_exit.get("generation_token") != generation_token
            or self.files["PASS"].bytes() != b"[MANAGED_EXECUTION_V2_PASS]\n"
        ):
            raise TasteT4OracleSmokeError("managed T3 gate cross-binding failed")
        if (
            science.get("schema_version") != T3_SCIENCE_SCHEMA
            or science.get("status") != "PASS"
            or science.get("stage") != T3_STAGE
            or science.get("marker") != T3_SCIENCE_MARKER
            or science.get("independent_scientific_verifier") is not True
            or science.get("temperature_refit_performed") is not True
            or science.get("selection_split") != "validation"
            or science.get("calibration_payload_loaded") is not False
            or science.get("test_payload_loaded") is not False
            or science.get("rf_oracle_used") is not False
            or science.get("argmax_invariant") is not True
            or science.get("downstream_same_model_temperature_schema_required") is not True
            or type(science.get("t2_receipt_id")) is not str
            or not science.get("t2_receipt_id")
            or not _SHA256.fullmatch(str(science.get("t2_receipt_gate_sha256", "")))
            or not _SHA256.fullmatch(str(science.get("source_evidence_sha256", "")))
        ):
            raise TasteT4OracleSmokeError("nested T3 scientific verification changed")
        checkpoint_hashes = _parse_sha256s(
            self.files["artifacts/checkpoint/sha256sums.txt"].bytes(),
            expected=set(T3_CHECKPOINT_FILES) - {"sha256sums.txt"},
            label="T3 checkpoint sha256sums.txt",
        )
        for name, digest in checkpoint_hashes.items():
            if self.files[f"artifacts/checkpoint/{name}"].sha256 != digest:
                raise TasteT4OracleSmokeError(f"T3 checkpoint hash changed: {name}")
        candidate = self.files[
            f"artifacts/checkpoint/{T3_CANDIDATE_NAME}"
        ].json()
        temperature = self.files[
            "artifacts/checkpoint/temperature_scaling.json"
        ].json()
        model_card = self.files["artifacts/checkpoint/model_card.json"].json()
        feature_schema = self.files[
            "artifacts/checkpoint/feature_schema.json"
        ].json()
        model_sha = self.files["artifacts/checkpoint/model.pt"].sha256
        temperature_sha = self.files[
            "artifacts/checkpoint/temperature_scaling.json"
        ].sha256
        feature_file_sha = self.files[
            "artifacts/checkpoint/feature_schema.json"
        ].sha256
        if (
            candidate.get("schema_version") != T3_SCIENCE_SCHEMA
            or candidate.get("stage") != T3_STAGE
            or candidate.get("candidate_status") != "SEALED_CANDIDATE"
            or candidate.get("managed_attempt_id") != attempt_id
            or candidate.get("managed_generation_token") != generation_token
            or candidate.get("model_sha256") != model_sha
            or candidate.get("temperature_scaling_sha256") != temperature_sha
            or candidate.get("feature_schema_file_sha256") != feature_file_sha
            or candidate.get("calibration_payload_loaded") is not False
            or candidate.get("test_payload_loaded") is not False
            or candidate.get("rf_oracle_used") is not False
            or temperature.get("temperature_refit_performed") is not True
            or temperature.get("selection_split") != "validation"
            or temperature.get("calibration_payload_loaded") is not False
            or temperature.get("test_payload_loaded") is not False
            or model_card.get("checkpoint_id") != model_sha
            or not _SHA256.fullmatch(
                str(model_card.get("graph_cache_manifest_sha256", ""))
            )
            or science.get("model_sha256") != model_sha
            or science.get("temperature_scaling_sha256") != temperature_sha
            or science.get("feature_schema_file_sha256") != feature_file_sha
            or science.get("feature_schema_sha256") != feature_schema.get("schema_sha256")
            or not isinstance(temperature.get("temperature"), (int, float))
            or isinstance(temperature.get("temperature"), bool)
            or not math.isfinite(float(temperature["temperature"]))
            or float(temperature["temperature"]) <= 0.0
            or float(science.get("temperature")) != float(temperature["temperature"])
        ):
            raise TasteT4OracleSmokeError("T3 checkpoint/science binding changed")
        inventory = [
            {"path": name, "sha256": held.sha256}
            for name, held in sorted(self.files.items())
        ]
        return {
            "schema_version": "tastemolnet_t4_t3_binding_v2",
            "t3_root": str(self.root),
            "t3_gate_sha256": self.files["gate.json"].sha256,
            "t3_verification_sha256": self.files["verification.json"].sha256,
            "t3_sealed_sha256": self.files["SEALED.json"].sha256,
            "t3_attempt_id": attempt_id,
            "t3_generation_token": generation_token,
            "t3_root_inventory_sha256": _sha256(_canonical_json_bytes(inventory)),
            "checkpoint_dir": str(self.root / "artifacts/checkpoint"),
            "checkpoint_sha256s_sha256": self.files[
                "artifacts/checkpoint/sha256sums.txt"
            ].sha256,
            "checkpoint_id": model_sha,
            "model_sha256": model_sha,
            "temperature": float(temperature["temperature"]),
            "temperature_scaling_sha256": temperature_sha,
            "feature_schema_file_sha256": feature_file_sha,
            "feature_schema_sha256": feature_schema["schema_sha256"],
            "graph_cache_manifest_sha256": model_card[
                "graph_cache_manifest_sha256"
            ],
            "source_t2_receipt_id": science.get("t2_receipt_id"),
            "source_t2_gate_sha256": science.get("t2_receipt_gate_sha256"),
            "source_t2_evidence_sha256": science.get("source_evidence_sha256"),
            "temperature_refit_performed": True,
            "selection_split": "validation",
            "calibration_payload_loaded": False,
            "test_payload_loaded": False,
            "rf_oracle_used": False,
        }

    def verify(self) -> None:
        self.root_directory.verify(label="held managed T3 root")
        self.artifact_directory.verify(label="held managed T3 artifacts")
        self.checkpoint_directory.verify(label="held managed T3 checkpoint")
        self._require_inventory()
        for item in self.files.values():
            item.verify()
        if self._validate() != self.binding:
            raise TasteT4OracleSmokeError("managed T3 semantic binding drifted")

    def close(self) -> None:
        for item in getattr(self, "files", {}).values():
            item.close()
        for name in ("checkpoint_directory", "artifact_directory", "root_directory"):
            item = getattr(self, name, None)
            if item is not None:
                item.close()


class HeldCalibrationCache:
    """Hold only manifest.json and calibration.pt from the existing cache."""

    def __init__(self, root: str | Path, *, expected_manifest_sha256: str) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.resolve(strict=True) != self.root:
            raise TasteT4OracleSmokeError("graph-cache root must be exact and physical")
        self.directory = _physical_directory(self.root, field="Taste graph-cache root")
        self.manifest = _HeldFile(self.root / "manifest.json", label="graph-cache manifest")
        self.calibration = _HeldFile(
            self.root / "calibration.pt", label="calibration graph cache"
        )
        try:
            if self.manifest.sha256 != expected_manifest_sha256:
                raise TasteT4OracleSmokeError("graph-cache manifest differs from T3 model")
            payload = self.manifest.json()
            splits = payload.get("splits")
            if (
                payload.get("schema_version") != "molecular_graph_cache_manifest_v1"
                or payload.get("dataset") != "tastemolnet"
                or payload.get("num_classes") != NUM_CLASSES
                or payload.get("split_order")
                != ["train", "validation", "calibration", "test"]
                or type(splits) is not dict
                or set(splits) != {"train", "validation", "calibration", "test"}
            ):
                raise TasteT4OracleSmokeError("graph-cache manifest contract changed")
            entry = splits.get("calibration")
            if (
                type(entry) is not dict
                or entry.get("cache_file") != "calibration.pt"
                or entry.get("num_classes") != NUM_CLASSES
                or entry.get("safe_load_verified") is not True
                or entry.get("cache_sha256") != self.calibration.sha256
                or type(entry.get("graph_count")) is not int
                or entry["graph_count"] <= 0
                or not _SHA256.fullmatch(str(entry.get("source_csv_sha256", "")))
            ):
                raise TasteT4OracleSmokeError("calibration cache entry changed")
            self.entry = dict(entry)
        except BaseException:
            self.close()
            raise

    def load(self, feature_schema: MolecularFeatureSchema) -> MolecularGraphDataset:
        dataset = load_molecular_graph_cache(
            io.BytesIO(self.calibration.bytes()),
            expected_num_classes=NUM_CLASSES,
            expected_source_sha256=self.entry["source_csv_sha256"],
            expected_feature_schema=feature_schema,
        )
        if len(dataset) != self.entry["graph_count"] or any(
            str(dataset[index].split) != "calibration" for index in range(len(dataset))
        ):
            raise TasteT4OracleSmokeError("calibration payload split/count changed")
        self.verify()
        return dataset

    def binding(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_t4_calibration_cache_binding_v2",
            "graph_cache_root": str(self.root),
            "graph_cache_manifest_sha256": self.manifest.sha256,
            "calibration_cache_sha256": self.calibration.sha256,
            "calibration_source_csv_sha256": self.entry["source_csv_sha256"],
            "calibration_graph_count": self.entry["graph_count"],
            "opened_payload_splits": ["calibration"],
            "train_payload_opened": False,
            "validation_payload_opened": False,
            "test_payload_opened": False,
            "csv_payload_opened": False,
            "graph_cache_rebuilt": False,
            "data_reprepared": False,
        }

    def verify(self) -> None:
        self.directory.verify(label="held graph-cache root")
        self.manifest.verify()
        self.calibration.verify()

    def close(self) -> None:
        for item in (getattr(self, "calibration", None), getattr(self, "manifest", None)):
            if item is not None:
                item.close()
        directory = getattr(self, "directory", None)
        if directory is not None:
            directory.close()


@dataclass(slots=True)
class T4ScienceRun:
    documents: dict[str, dict[str, Any]]
    input_hashes: dict[str, str]
    _revalidate: Callable[[], None]
    _close: Callable[[], None]

    def revalidate(self) -> None:
        self._revalidate()

    def close(self) -> None:
        self._close()


ScienceRunner = Callable[..., T4ScienceRun]


def collect_t4_managed_input_hashes(
    *,
    t3_root: str | Path,
    graph_cache_root: str | Path,
    controller_launcher_receipt_sha256: str,
    controller_receipt_sha256: str,
    controller_anchor_heartbeat_sha256: str,
    gpu_lease_sha256: str,
) -> dict[str, str]:
    """Hold the published T3/cache metadata and derive exact attempt pins."""

    t3 = HeldPublishedT3(_require_published_t3_root(t3_root))
    cache: HeldCalibrationCache | None = None
    try:
        cache = HeldCalibrationCache(
            graph_cache_root,
            expected_manifest_sha256=t3.binding["graph_cache_manifest_sha256"],
        )
        result = {
            "t3_gate": t3.binding["t3_gate_sha256"],
            "t3_verification": t3.binding["t3_verification_sha256"],
            "graph_cache_manifest": cache.manifest.sha256,
            "calibration_cache": cache.calibration.sha256,
            "controller_launcher_receipt": controller_launcher_receipt_sha256,
            "controller_receipt": controller_receipt_sha256,
            "controller_anchor_heartbeat": controller_anchor_heartbeat_sha256,
            "gpu1_lease": gpu_lease_sha256,
        }
        if any(not _SHA256.fullmatch(value) for value in result.values()):
            raise TasteT4OracleSmokeError("T4 managed input hash is malformed")
        t3.verify()
        cache.verify()
        return result
    finally:
        if cache is not None:
            cache.close()
        t3.close()


def _require_gpu1_environment(*, gpu_uuid: str) -> dict[str, Any]:
    if not _GPU_UUID.fullmatch(gpu_uuid):
        raise TasteT4OracleSmokeError("physical GPU UUID is malformed")
    if os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != str(PHYSICAL_GPU_INDEX):
        raise TasteT4OracleSmokeError("T4 lacks physical GPU1 index binding")
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != gpu_uuid:
        raise TasteT4OracleSmokeError("T4 GPU UUID differs from controller binding")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_VISIBLE_DEVICES:
        raise TasteT4OracleSmokeError("CUDA visibility is not physical GPU1")
    try:
        observed = probe_physical_gpus()
    except TasteMainV2AuthorityError as exc:
        raise TasteT4OracleSmokeError(str(exc)) from exc
    if observed.get(PHYSICAL_GPU_INDEX) != gpu_uuid:
        raise TasteT4OracleSmokeError("nvidia-smi GPU1 UUID binding changed")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL owns torch.
        raise TasteT4OracleSmokeError("PyTorch is required for T4") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TasteT4OracleSmokeError("T4 must see exactly one CUDA device")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise TasteT4OracleSmokeError("physical GPU1 is not mapped to cuda:0")
    return {
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "physical_gpu_uuid": gpu_uuid,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "visible_device": VISIBLE_DEVICE,
        "visible_device_count": 1,
    }


def _validate_aggregate_smoke(smoke: Mapping[str, Any]) -> None:
    distribution = smoke.get("destination_distribution")
    overall = distribution.get("overall") if type(distribution) is dict else None
    transitions = overall.get("transitions") if type(overall) is dict else None
    if (
        smoke.get("status") != "PASS"
        or smoke.get("selected_count") != SOURCE_COUNT
        or smoke.get("parent_deletion_counts_by_position")
        != [DELETIONS_PER_PARENT] * SOURCE_COUNT
        or smoke.get("valid_deletion_count") != SOURCE_COUNT * DELETIONS_PER_PARENT
        or smoke.get("all_selected_true_source") is not True
        or smoke.get("all_selected_predicted_source") is not True
        or smoke.get("all_selected_have_four_connected_deletions") is not True
        or smoke.get("checkpoint_load_count") != 1
        or smoke.get("num_classes") != NUM_CLASSES
        or smoke.get("source_label") != SOURCE_LABEL
        or smoke.get("strict_flip") != "pred_before == 1 and pred_after != 1"
        or smoke.get("all_three_probabilities_validated") is not True
        or smoke.get("empty_deletion_failed_closed") is not True
        or smoke.get("invalid_deletion_failed_closed") is not True
        or smoke.get("per_example_predictions_written") is not False
        or smoke.get("smiles_written") is not False
        or smoke.get("molecule_identifiers_written") is not False
        or type(transitions) is not dict
        or type(transitions.get("1->0")) is not dict
        or type(transitions.get("1->1")) is not dict
        or type(transitions.get("1->2")) is not dict
        or transitions["1->0"].get("count", 0) <= 0
        or transitions["1->1"].get("count", 0) <= 0
        or transitions["1->2"].get("count", 0) <= 0
    ):
        raise TasteT4OracleSmokeError("T4 aggregate smoke contract changed")


def _execute_science(
    *,
    t3_root: str | Path,
    graph_cache_root: str | Path,
    gpu_uuid: str,
    batch_size: int,
) -> T4ScienceRun:
    if type(batch_size) is not int or batch_size <= 0:
        raise TasteT4OracleSmokeError("batch size must be a positive native integer")
    gpu = _require_gpu1_environment(gpu_uuid=gpu_uuid)
    t3 = HeldPublishedT3(_require_published_t3_root(t3_root))
    cache: HeldCalibrationCache | None = None
    try:
        model_card = t3.files["artifacts/checkpoint/model_card.json"].json()
        expected_manifest = model_card.get("graph_cache_manifest_sha256")
        if not _SHA256.fullmatch(str(expected_manifest or "")):
            raise TasteT4OracleSmokeError("T3 model lacks graph-cache manifest binding")
        cache = HeldCalibrationCache(
            graph_cache_root, expected_manifest_sha256=str(expected_manifest)
        )
        feature_schema = MolecularFeatureSchema.from_dict(
            t3.files["artifacts/checkpoint/feature_schema.json"].json()
        )
        dataset = cache.load(feature_schema)
        # This is the sole checkpoint construction in this scientific process.
        oracle = _load_gnn_oracle_anchored(
            t3.checkpoint_directory,
            feature_schema=feature_schema,
            device=VISIBLE_DEVICE,
            batch_size=batch_size,
        )
        if (
            oracle.checkpoint_id != t3.binding["checkpoint_id"]
            or oracle.backbone != "gine"
            or oracle.num_classes != NUM_CLASSES
            or oracle.source_label != SOURCE_LABEL
            or float(oracle.temperature) != t3.binding["temperature"]
        ):
            raise TasteT4OracleSmokeError("loaded oracle differs from T3 authority")
        smoke = run_bounded_oracle_smoke(
            dataset=dataset,
            oracle=oracle,
            feature_schema=feature_schema,
            batch_size=batch_size,
            source_count=SOURCE_COUNT,
            max_deletions_per_parent=DELETIONS_PER_PARENT,
        )
        _validate_aggregate_smoke(smoke)
        cache_binding = cache.binding()
        t3_binding = dict(t3.binding)
        provenance = {
            "schema_version": "tastemolnet_t4_oracle_provenance_v2",
            "dataset": "tastemolnet",
            "stage": STAGE,
            "t3_root": str(t3.root),
            "checkpoint_dir": t3_binding["checkpoint_dir"],
            "checkpoint_id": t3_binding["checkpoint_id"],
            "model_sha256": t3_binding["model_sha256"],
            "temperature": t3_binding["temperature"],
            "temperature_scaling_sha256": t3_binding[
                "temperature_scaling_sha256"
            ],
            "feature_schema_file_sha256": t3_binding[
                "feature_schema_file_sha256"
            ],
            "feature_schema_sha256": t3_binding["feature_schema_sha256"],
            **gpu,
            "checkpoint_load_count": 1,
            "rf_oracle_used": False,
            "calibration_payload_loaded": True,
            "train_payload_loaded": False,
            "validation_payload_loaded": False,
            "test_payload_loaded": False,
            "model_load_scope": "once_per_scientific_process",
        }
        access = {
            "schema_version": "tastemolnet_t4_data_access_v2",
            **cache_binding,
            "allowed_checkpoint_root": t3_binding["checkpoint_dir"],
            "checkpoint_csv_payload_opened": False,
            "per_example_output_written": False,
            "dataset_redistributed": False,
        }
        documents = {
            "oracle_smoke.json": dict(smoke),
            "oracle_provenance.json": provenance,
            "data_access_manifest.json": access,
            "t3_binding.json": t3_binding,
        }
        input_hashes = {
            "t3_gate": t3_binding["t3_gate_sha256"],
            "t3_verification": t3_binding["t3_verification_sha256"],
            "graph_cache_manifest": cache_binding["graph_cache_manifest_sha256"],
            "calibration_cache": cache_binding["calibration_cache_sha256"],
        }
        t3.verify()
        cache.verify()

        def revalidate() -> None:
            t3.verify()
            assert cache is not None
            cache.verify()

        def close() -> None:
            assert cache is not None
            cache.close()
            t3.close()

        return T4ScienceRun(documents, input_hashes, revalidate, close)
    except (TasteGNNStageError, OSError, ValueError) as exc:
        if cache is not None:
            cache.close()
        t3.close()
        raise TasteT4OracleSmokeError(str(exc)) from exc
    except BaseException:
        if cache is not None:
            cache.close()
        t3.close()
        raise


def _validate_documents(documents: Mapping[str, Any]) -> None:
    if set(documents) != set(METHOD_DOCUMENTS):
        raise TasteT4OracleSmokeError("T4 method document inventory changed")
    if any(type(value) is not dict for value in documents.values()):
        raise TasteT4OracleSmokeError("T4 method document is not one JSON object")
    _validate_aggregate_smoke(documents["oracle_smoke.json"])
    provenance = documents["oracle_provenance.json"]
    access = documents["data_access_manifest.json"]
    t3_binding = documents["t3_binding.json"]
    if (
        provenance.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
        or provenance.get("visible_device") != VISIBLE_DEVICE
        or provenance.get("cuda_visible_devices") != CUDA_VISIBLE_DEVICES
        or provenance.get("checkpoint_load_count") != 1
        or provenance.get("model_sha256") != t3_binding.get("model_sha256")
        or provenance.get("temperature_scaling_sha256")
        != t3_binding.get("temperature_scaling_sha256")
        or access.get("opened_payload_splits") != ["calibration"]
        or access.get("train_payload_opened") is not False
        or access.get("validation_payload_opened") is not False
        or access.get("test_payload_opened") is not False
        or access.get("csv_payload_opened") is not False
        or access.get("per_example_output_written") is not False
        or t3_binding.get("temperature_refit_performed") is not True
        or t3_binding.get("selection_split") != "validation"
    ):
        raise TasteT4OracleSmokeError("T4 aggregate provenance/access contract changed")
    encoded = _canonical_json_bytes(documents)
    forbidden = (b'"smiles":', b'"molecule_id":', b'"parent_smiles":', b'"rows":')
    if any(token in encoded for token in forbidden):
        raise TasteT4OracleSmokeError("T4 aggregate documents contain row-level data")


def build_t4_candidate(
    *,
    t3_root: str | Path,
    graph_cache_root: str | Path,
    artifact_root: str | Path,
    attempt_id: str,
    generation_token: str,
    gpu_uuid: str,
    controller_launcher_receipt_path: str | Path,
    controller_receipt_path: str | Path,
    controller_anchor_heartbeat_path: str | Path,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_git_tree: str,
    expected_controller_launcher_receipt_sha256: str,
    expected_controller_receipt_sha256: str,
    expected_controller_anchor_heartbeat_sha256: str,
    expected_gpu_lease_uuid: str,
    expected_gpu_lease_sha256: str,
    controller_max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    controller_barrier_timeout_seconds: int = 45,
    batch_size: int = 32,
    science_runner: ScienceRunner = _execute_science,
) -> dict[str, Any]:
    """Worker-only candidate construction; never writes a gate or PASS."""

    output = Path(artifact_root)
    if not output.is_absolute() or output.resolve(strict=True) != output:
        raise TasteT4OracleSmokeError("managed artifact root must be exact and physical")
    if {entry.name for entry in os.scandir(output)} != {".generation_token.json"}:
        raise TasteT4OracleSmokeError("managed artifact root is not fresh")
    if controller_barrier_timeout_seconds != 45:
        raise TasteT4OracleSmokeError("production controller barrier must be 45 seconds")
    deadline = time.monotonic() + controller_barrier_timeout_seconds
    while True:
        try:
            authority_context = hold_taste_main_v2_controller_authority(
                controller_receipt_path,
                controller_anchor_heartbeat_path,
                expected_controller_id,
                expected_git_commit,
                expected_git_tree,
                controller_max_heartbeat_age_seconds,
                expected_launcher_receipt_path=controller_launcher_receipt_path,
                expected_launcher_receipt_sha256=(
                    expected_controller_launcher_receipt_sha256
                ),
                expected_receipt_sha256=expected_controller_receipt_sha256,
                expected_heartbeat_sha256=(
                    expected_controller_anchor_heartbeat_sha256
                ),
                expected_task_id=TASK_ID,
                expected_gpu_index=PHYSICAL_GPU_INDEX,
                expected_gpu_uuid=gpu_uuid,
                expected_lease_uuid=expected_gpu_lease_uuid,
                expected_lease_sha256=expected_gpu_lease_sha256,
                expected_attempt_id=attempt_id,
                expected_generation_token=generation_token,
                expected_activation_phase="WORKER_ACTIVE",
            )
            break
        except (OSError, ValueError, TasteMainV2AuthorityError) as exc:
            if time.monotonic() >= deadline:
                raise TasteT4OracleSmokeError(
                    f"controller ACTIVE barrier rejected: {exc}"
                ) from exc
            time.sleep(0.25)
    with authority_context as authority:
        initial_authority = dict(authority.evidence)
        if initial_authority.get("anchor_heartbeat_sequence") != 1:
            raise TasteT4OracleSmokeError(
                "T4 managed input must anchor controller heartbeat sequence 1"
            )
        run = science_runner(
            t3_root=t3_root,
            graph_cache_root=graph_cache_root,
            gpu_uuid=gpu_uuid,
            batch_size=batch_size,
        )
        try:
            managed_input_hashes = {
                **run.input_hashes,
                "controller_launcher_receipt": initial_authority[
                    "launcher_receipt_sha256"
                ],
                "controller_receipt": authority.receipt.sha256,
                "controller_anchor_heartbeat": initial_authority[
                    "anchor_heartbeat_sha256"
                ],
                "gpu1_lease": initial_authority["lease_sha256"],
            }
            if (
                managed_input_hashes["controller_launcher_receipt"]
                != expected_controller_launcher_receipt_sha256
                or managed_input_hashes["controller_receipt"]
                != expected_controller_receipt_sha256
                or managed_input_hashes["controller_anchor_heartbeat"]
                != expected_controller_anchor_heartbeat_sha256
                or managed_input_hashes["gpu1_lease"] != expected_gpu_lease_sha256
            ):
                raise TasteT4OracleSmokeError("controller authority input hashes drifted")
            _validate_documents(run.documents)
            run.revalidate()
            document_hashes: dict[str, str] = {}
            for name in sorted(METHOD_DOCUMENTS):
                data = _canonical_json_bytes(run.documents[name])
                _write_exclusive(output / name, data)
                document_hashes[name] = _sha256(data)
            candidate = {
                "schema_version": SCHEMA_VERSION,
                "stage": STAGE,
                "candidate_status": "SEALED_CANDIDATE",
                "managed_attempt_id": attempt_id,
                "managed_generation_token": generation_token,
                "input_hashes": dict(managed_input_hashes),
                "controller_authority": {
                    "worker_initial": initial_authority,
                    "worker_final": dict(authority.revalidate()),
                    "held_across_science": True,
                    "release_authority": False,
                },
                "document_hashes": document_hashes,
                "selected_count": SOURCE_COUNT,
                "deletions_per_parent": DELETIONS_PER_PARENT,
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_uuid": gpu_uuid,
                "visible_device": VISIBLE_DEVICE,
                "model_load_count": 1,
                "independent_verification_required": True,
                "worker_terminal_authority": False,
                "per_example_output_written": False,
                "matrix_method_cell": False,
            }
            candidate_data = _canonical_json_bytes(candidate)
            _write_exclusive(output / T4_CANDIDATE_NAME, candidate_data)
            hashes = {**document_hashes, T4_CANDIDATE_NAME: _sha256(candidate_data)}
            _write_exclusive(
                output / "sha256sums.txt",
                "".join(
                    f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
                ).encode("utf-8"),
            )
            descriptor = os.open(
                output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            run.revalidate()
            authority.revalidate()
            return {
                "state": "SEALED_CANDIDATE",
                "stage": STAGE,
                "artifact_root": str(output),
                "input_hashes": dict(managed_input_hashes),
                "selected_count": SOURCE_COUNT,
                "valid_deletion_count": SOURCE_COUNT * DELETIONS_PER_PARENT,
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_uuid": gpu_uuid,
                "model_load_count": 1,
                "independent_verification_required": True,
            }
        finally:
            run.close()


def _read_held(held: HeldSealedArtifactV2, relative: str) -> bytes:
    held.revalidate()
    matches = [item for item in held.files if item.evidence.relative_path == relative]
    if len(matches) != 1:
        raise TasteT4OracleSmokeError(f"SEALED T4 file is absent: {relative}")
    item = matches[0]
    os.lseek(item.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(item.descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    os.lseek(item.descriptor, 0, os.SEEK_SET)
    data = b"".join(chunks)
    if _sha256(data) != item.evidence.sha256:
        raise TasteT4OracleSmokeError(f"SEALED T4 file changed: {relative}")
    held.revalidate()
    return data


def _candidate_payloads(held: HeldSealedArtifactV2) -> dict[str, bytes]:
    expected = {
        ".generation_token.json",
        "raw_evidence.json",
        "worker_exit.json",
        "artifacts/.generation_token.json",
        *{f"artifacts/{name}" for name in CANDIDATE_FILES},
    }
    if {item.evidence.relative_path for item in held.files} != expected or {
        item.relative_path for item in held.inventory.directories
    } != {"artifacts"}:
        raise TasteT4OracleSmokeError("SEALED T4 candidate inventory changed")
    return {
        name: _read_held(held, f"artifacts/{name}")
        for name in sorted(CANDIDATE_FILES)
    }


def _equivalent(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10)
    return left == right


def _verify_worker_evidence(
    held: HeldSealedArtifactV2,
    *,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_config_hash: str,
    expected_input_hashes: Mapping[str, str],
) -> None:
    raw = _json(_read_held(held, "raw_evidence.json"), label="T4 raw evidence")
    exited = _json(_read_held(held, "worker_exit.json"), label="T4 worker exit")
    evidence = raw.get("evidence")
    exit_body = exited.get("exit")
    if type(evidence) is not dict or type(exit_body) is not dict:
        raise TasteT4OracleSmokeError("T4 managed evidence body is absent")
    attempt = evidence.get("attempt_manifest")
    lineage = evidence.get("process_lineage")
    audit = exit_body.get("process_audit")
    if (
        type(attempt) is not dict
        or type(lineage) is not dict
        or type(audit) is not dict
        or raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or exited.get("schema_version") != WORKER_EXIT_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or exited.get("attempt_id") != held.sealed.attempt_id
        or exited.get("generation_token") != held.sealed.generation_token
        or attempt.get("attempt_id") != held.sealed.attempt_id
        or attempt.get("controller_id") != expected_controller_id
        or attempt.get("task_id") != TASK_ID
        or attempt.get("git_commit") != expected_git_commit
        or attempt.get("config_hash") != expected_config_hash
        or attempt.get("input_hashes") != dict(expected_input_hashes)
        or attempt.get("auto_terminate_uncontrolled_children") is not False
        or lineage.get("controller_id") != expected_controller_id
        or lineage.get("attempt_id") != held.sealed.attempt_id
        or exit_body.get("exit_code") != 0
        or exit_body.get("worker_closed_artifact_writers") is not True
        or audit.get("state") != "EXITED"
        or audit.get("controller_id") != expected_controller_id
        or audit.get("attempt_id") != held.sealed.attempt_id
    ):
        raise TasteT4OracleSmokeError("T4 managed worker evidence is not releasable")


def verify_and_publish_t4(
    *,
    sealed_path: str | Path,
    final_path: str | Path,
    t3_root: str | Path,
    graph_cache_root: str | Path,
    gpu_uuid: str,
    expected_attempt_id: str,
    expected_generation_token: str,
    expected_controller_id: str,
    expected_git_commit: str,
    expected_git_tree: str,
    expected_config_hash: str,
    controller_launcher_receipt_path: str | Path,
    controller_receipt_path: str | Path,
    controller_anchor_heartbeat_path: str | Path,
    expected_controller_launcher_receipt_sha256: str,
    expected_controller_receipt_sha256: str,
    expected_controller_anchor_heartbeat_sha256: str,
    expected_gpu_lease_uuid: str,
    expected_gpu_lease_sha256: str,
    controller_max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    controller_barrier_timeout_seconds: int = 45,
    batch_size: int = 32,
    science_runner: ScienceRunner = _execute_science,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Independent method verifier and sole route into terminal publication."""

    destination = Path(final_path)
    predecessor = Path(t3_root)
    if (
        not destination.is_absolute()
        or destination.parent != predecessor.parent
        or not destination.name.startswith("t4-oracle-smoke-")
    ):
        raise TasteT4OracleSmokeError(
            "T4 final path must be a t4-oracle-smoke-* sibling of T3"
        )
    if controller_barrier_timeout_seconds != 45:
        raise TasteT4OracleSmokeError("production controller barrier must be 45 seconds")
    deadline = time.monotonic() + controller_barrier_timeout_seconds
    while True:
        try:
            authority_context = hold_taste_main_v2_controller_authority(
            controller_receipt_path,
            controller_anchor_heartbeat_path,
            expected_controller_id,
            expected_git_commit,
            expected_git_tree,
            controller_max_heartbeat_age_seconds,
            expected_launcher_receipt_path=controller_launcher_receipt_path,
            expected_launcher_receipt_sha256=(
                expected_controller_launcher_receipt_sha256
            ),
            expected_receipt_sha256=expected_controller_receipt_sha256,
            expected_heartbeat_sha256=(
                expected_controller_anchor_heartbeat_sha256
            ),
            expected_task_id=TASK_ID,
            expected_gpu_index=PHYSICAL_GPU_INDEX,
            expected_gpu_uuid=gpu_uuid,
            expected_lease_uuid=expected_gpu_lease_uuid,
            expected_lease_sha256=expected_gpu_lease_sha256,
            expected_attempt_id=expected_attempt_id,
            expected_generation_token=expected_generation_token,
            expected_activation_phase="VERIFIER_ACTIVE",
            )
            break
        except (OSError, ValueError, TasteMainV2AuthorityError) as exc:
            if time.monotonic() >= deadline:
                raise TasteT4OracleSmokeError(
                    f"controller VERIFIER barrier rejected: {exc}"
                ) from exc
            time.sleep(0.25)
    with authority_context as authority, open_sealed_worker_artifact(
        sealed_path,
        expected_attempt_id=expected_attempt_id,
        expected_generation_token=expected_generation_token,
    ) as held:
        if authority.evidence.get("anchor_heartbeat_sequence") != 1:
            raise TasteT4OracleSmokeError(
                "T4 verifier must anchor controller heartbeat sequence 1"
            )
        run = science_runner(
            t3_root=t3_root,
            graph_cache_root=graph_cache_root,
            gpu_uuid=gpu_uuid,
            batch_size=batch_size,
        )
        try:
            expected_input_hashes = {
                **run.input_hashes,
                "controller_launcher_receipt": (
                    expected_controller_launcher_receipt_sha256
                ),
                "controller_receipt": expected_controller_receipt_sha256,
                "controller_anchor_heartbeat": (
                    expected_controller_anchor_heartbeat_sha256
                ),
                "gpu1_lease": expected_gpu_lease_sha256,
            }
            _validate_documents(run.documents)
            _verify_worker_evidence(
                held,
                expected_controller_id=expected_controller_id,
                expected_git_commit=expected_git_commit,
                expected_config_hash=expected_config_hash,
                expected_input_hashes=expected_input_hashes,
            )
            payloads = _candidate_payloads(held)
            hashes = _parse_sha256s(
                payloads["sha256sums.txt"],
                expected=set(CANDIDATE_FILES) - {"sha256sums.txt"},
                label="T4 candidate sha256sums.txt",
            )
            for name, digest in hashes.items():
                if _sha256(payloads[name]) != digest:
                    raise TasteT4OracleSmokeError(f"T4 candidate hash changed: {name}")
            candidate_documents = {
                name: _json(payloads[name], label=f"candidate {name}")
                for name in METHOD_DOCUMENTS
            }
            if not _equivalent(candidate_documents, run.documents):
                raise TasteT4OracleSmokeError("independent T4 replay differs from worker")
            candidate = _json(
                payloads[T4_CANDIDATE_NAME], label="T4 candidate manifest"
            )
            if (
                candidate.get("schema_version") != SCHEMA_VERSION
                or candidate.get("stage") != STAGE
                or candidate.get("candidate_status") != "SEALED_CANDIDATE"
                or candidate.get("managed_attempt_id") != held.sealed.attempt_id
                or candidate.get("managed_generation_token") != held.sealed.generation_token
                or candidate.get("input_hashes") != expected_input_hashes
                or candidate.get("document_hashes")
                != {name: hashes[name] for name in METHOD_DOCUMENTS}
                or candidate.get("selected_count") != SOURCE_COUNT
                or candidate.get("deletions_per_parent") != DELETIONS_PER_PARENT
                or candidate.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
                or candidate.get("physical_gpu_uuid") != gpu_uuid
                or candidate.get("visible_device") != VISIBLE_DEVICE
                or candidate.get("model_load_count") != 1
                or candidate.get("independent_verification_required") is not True
                or candidate.get("worker_terminal_authority") is not False
                or candidate.get("per_example_output_written") is not False
                or candidate.get("matrix_method_cell") is not False
            ):
                raise TasteT4OracleSmokeError("T4 candidate manifest contract changed")
            candidate_authority = candidate.get("controller_authority")
            if type(candidate_authority) is not dict:
                raise TasteT4OracleSmokeError("T4 candidate lacks held controller authority")
            initial_authority = candidate_authority.get("worker_initial")
            final_authority = candidate_authority.get("worker_final")
            if (
                type(initial_authority) is not dict
                or type(final_authority) is not dict
                or candidate_authority.get("held_across_science") is not True
                or candidate_authority.get("release_authority") is not False
                or initial_authority.get("controller_id") != expected_controller_id
                or initial_authority.get("git_commit") != expected_git_commit
                or initial_authority.get("git_tree") != expected_git_tree
                or initial_authority.get("receipt_sha256")
                != expected_controller_receipt_sha256
                or initial_authority.get("launcher_receipt_sha256")
                != expected_controller_launcher_receipt_sha256
                or initial_authority.get("anchor_heartbeat_sha256")
                != expected_controller_anchor_heartbeat_sha256
                or initial_authority.get("anchor_heartbeat_sequence") != 1
                or initial_authority.get("activation_phase") != "WORKER_ACTIVE"
                or initial_authority.get("lease_uuid") != expected_gpu_lease_uuid
                or initial_authority.get("lease_sha256") != expected_gpu_lease_sha256
                or initial_authority.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
                or initial_authority.get("physical_gpu_uuid") != gpu_uuid
                or final_authority.get("controller_id") != expected_controller_id
                or final_authority.get("receipt_sha256")
                != expected_controller_receipt_sha256
                or final_authority.get("lease_uuid") != expected_gpu_lease_uuid
                or final_authority.get("lease_sha256") != expected_gpu_lease_sha256
                or final_authority.get("anchor_heartbeat_sha256")
                != expected_controller_anchor_heartbeat_sha256
                or final_authority.get("anchor_heartbeat_sequence") != 1
            ):
                raise TasteT4OracleSmokeError("T4 worker controller authority changed")
            run.revalidate()
            verifier_authority = authority.revalidate()
            smoke = run.documents["oracle_smoke.json"]
            t3_binding = run.documents["t3_binding.json"]
            verification = {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "stage": STAGE,
                "marker": PASS_MARKER,
                "independent_scientific_verifier": True,
                "verifier_git_commit": expected_git_commit,
                "verifier_git_tree": expected_git_tree,
                "controller_authority": verifier_authority,
                "controller_receipt_sha256": expected_controller_receipt_sha256,
                "controller_launcher_receipt_sha256": (
                    expected_controller_launcher_receipt_sha256
                ),
                "controller_anchor_heartbeat_sha256": (
                    expected_controller_anchor_heartbeat_sha256
                ),
                "worker_initial_heartbeat_sha256": initial_authority[
                    "heartbeat_sha256"
                ],
                "worker_final_heartbeat_sha256": final_authority[
                    "heartbeat_sha256"
                ],
                "verifier_heartbeat_sha256": verifier_authority["heartbeat_sha256"],
                "gpu1_lease_sha256": expected_gpu_lease_sha256,
                "gpu1_lease_uuid": expected_gpu_lease_uuid,
                "t3_root": str(predecessor),
                "t3_gate_sha256": t3_binding["t3_gate_sha256"],
                "t3_verification_sha256": t3_binding["t3_verification_sha256"],
                "checkpoint_dir": t3_binding["checkpoint_dir"],
                "checkpoint_id": t3_binding["checkpoint_id"],
                "model_sha256": t3_binding["model_sha256"],
                "temperature": t3_binding["temperature"],
                "temperature_scaling_sha256": t3_binding[
                    "temperature_scaling_sha256"
                ],
                "feature_schema_sha256": t3_binding["feature_schema_sha256"],
                "graph_cache_manifest_sha256": run.input_hashes[
                    "graph_cache_manifest"
                ],
                "calibration_cache_sha256": run.input_hashes["calibration_cache"],
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_uuid": gpu_uuid,
                "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                "visible_device": VISIBLE_DEVICE,
                "selected_count": SOURCE_COUNT,
                "valid_deletion_count": SOURCE_COUNT * DELETIONS_PER_PARENT,
                "batch_single_max_abs_difference": smoke[
                    "batch_single_max_abs_difference"
                ],
                "all_three_probabilities_validated": True,
                "strict_flip_to_bitter_observed": True,
                "strict_flip_to_tasteless_observed": True,
                "no_flip_observed": True,
                "invalid_deletion_failed_closed": True,
                "full_parent_deletion_failed_closed": True,
                "model_load_count_per_scientific_process": 1,
                "calibration_payload_loaded": True,
                "train_payload_loaded": False,
                "validation_payload_loaded": False,
                "test_payload_loaded": False,
                "rf_oracle_used": False,
                "per_example_output_written": False,
                "matrix_method_cell": False,
                "worker_candidate_sha256": hashes[T4_CANDIDATE_NAME],
                "method_documents_sha256": _sha256(
                    _canonical_json_bytes(
                        {name: hashes[name] for name in sorted(METHOD_DOCUMENTS)}
                    )
                ),
            }
            run.revalidate()
            verification["controller_authority"] = authority.revalidate()
            verification["verifier_heartbeat_sha256"] = verification[
                "controller_authority"
            ]["heartbeat_sha256"]
            publication = verify_and_publish_sealed_attempt(
                held, final_path=destination, verification=verification
            )
            return publication, verification
        finally:
            run.close()


__all__ = [
    "CANDIDATE_FILES",
    "CUDA_VISIBLE_DEVICES",
    "DELETIONS_PER_PARENT",
    "METHOD_DOCUMENTS",
    "PASS_MARKER",
    "PHYSICAL_GPU_INDEX",
    "PUBLISHED_T3_ROOT",
    "SCHEMA_VERSION",
    "SOURCE_COUNT",
    "STAGE",
    "T4ScienceRun",
    "TASK_ID",
    "TasteT4OracleSmokeError",
    "build_t4_candidate",
    "collect_t4_managed_input_hashes",
    "verify_and_publish_t4",
]
