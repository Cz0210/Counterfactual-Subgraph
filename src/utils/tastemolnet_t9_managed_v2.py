"""Managed-v2 worker and independent verifier for TasteMolNet T9.

This successor deliberately uses ``TRUSTED_SINGLE_OPERATOR_ROOT`` for launch
authority.  Trust in the operator replaces the unfinished controller receipt,
not scientific provenance: both the worker and verifier independently retain
T2/T3/T4, the frozen GINE bundle, train CSV, config, immutable Git checkout,
and pinned official COMRECGC sources.  The worker can publish only aggregate
raw evidence and ``SEALED.json``.  Only the verifier can atomically publish a
generic managed-v2 PASS directory.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

from src.baselines.comrecgc.held_upstream import (
    OFFICIAL_SOURCE_SHA256,
    HeldImportedCOMRECGC,
    hold_imported_comrecgc,
)
from src.baselines.tastemolnet_comrecgc_smoke import (
    DATASET,
    METHOD,
    OFFICIAL_COMRECGC_COMMIT,
    PASS_MARKER,
    SMOKE_SOURCE_POOL,
    STAGE,
    TASK_ID,
    TasteComRecGCSmokeError,
    execute_native_comrecgc_smoke,
    validate_native_comrecgc_smoke_result,
)
from src.utils.managed_execution_v2 import (
    ATTEMPT_MANIFEST_SCHEMA,
    HeldWorkerStagingV2,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    create_managed_attempt,
    create_worker_staging,
    load_verified_gate,
    write_worker_exit,
    write_worker_raw_evidence,
)
from src.utils.process_identity_v2 import canonical_json_bytes
from src.utils.retained_output_directory import RetainedOutputTree
from src.utils.terminal_publisher_v2 import (
    HeldSealedArtifactV2,
    SealedWorkerArtifactV2,
    TerminalPublicationV2,
    open_sealed_worker_artifact,
    seal_worker_staging,
    verify_and_publish_sealed_attempt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUST_MODEL = "TRUSTED_SINGLE_OPERATOR_ROOT"
PHYSICAL_GPU_INDEX = 1
T9_INPUT_AUTHORITY_SCHEMA = "tastemolnet_t9_trusted_input_authority_v2"
T9_WORKER_RAW_SCHEMA = "tastemolnet_t9_managed_worker_raw_v2"
T9_VERIFICATION_SCHEMA = "tastemolnet_t9_independent_verification_v2"
SCIENCE_FILE = "artifacts/comrecgc_smoke.json"
INPUT_AUTHORITY_FILE = "artifacts/input_authority.json"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_SAFE_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_AUTHORITY_CHECKPOINT_PAYLOADS = (
    "model.pt",
    "config.yaml",
    "model_card.json",
    "feature_schema.json",
    "label_map.json",
    "split_manifest.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
)
_MODEL_LOAD_CHECKPOINT_PAYLOADS = (
    "model.pt",
    "model_card.json",
    "feature_schema.json",
    "label_map.json",
    "split_manifest.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
)
_T2_RECEIPT_FILES = frozenset(
    {
        "PASS",
        "artifact_hashes.json",
        "gate.json",
        "input_hashes.json",
        "sha256s.txt",
        "source_evidence.json",
        "verification.json",
    }
)


class TasteT9ManagedV2Error(TasteComRecGCSmokeError):
    """The trusted-root T9 managed-v2 closure is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TasteT9ManagedV2Error(f"{label} is not lowercase SHA-256")
    return value


def _absolute(value: str | Path, *, label: str, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteT9ManagedV2Error(f"{label} must be one normalized absolute path")
    if must_exist:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise TasteT9ManagedV2Error(f"{label} is unavailable") from exc
        if resolved != path:
            raise TasteT9ManagedV2Error(f"{label} contains a symlink or alias")
    return path


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def _json_object(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteT9ManagedV2Error(f"{label} is malformed JSON") from exc
    if type(value) is not dict:
        raise TasteT9ManagedV2Error(f"{label} must be one JSON object")
    return value


def _checkpoint_payloads_for_model_load(
    payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Project the full held authority onto the strict GNN loader contract."""

    if type(payloads) is not dict or set(payloads) != set(
        _AUTHORITY_CHECKPOINT_PAYLOADS
    ):
        raise TasteT9ManagedV2Error("T9 held checkpoint payload set changed")
    if any(
        type(payloads[name]) is not bytes or not payloads[name]
        for name in payloads
    ):
        raise TasteT9ManagedV2Error("T9 held checkpoint payload bytes changed")
    projected = {
        name: payloads[name] for name in _MODEL_LOAD_CHECKPOINT_PAYLOADS
    }
    if set(projected) != set(_MODEL_LOAD_CHECKPOINT_PAYLOADS):
        raise TasteT9ManagedV2Error("T9 model-load checkpoint payload set changed")
    return projected


def inspect_clean_execution() -> dict[str, str]:
    """Return the exact commit/tree only for a clean current worktree."""

    commands = (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("tree", ["git", "rev-parse", "HEAD^{tree}"]),
        (
            "status",
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        ),
    )
    observed: dict[str, str] = {}
    for name, command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise TasteT9ManagedV2Error(f"immutable Git {name} audit failed")
        observed[name] = completed.stdout.strip()
    if observed["status"]:
        raise TasteT9ManagedV2Error("T9 requires one clean immutable execution tree")
    if (
        _SHA1_RE.fullmatch(observed["commit"]) is None
        or _SHA1_RE.fullmatch(observed["tree"]) is None
    ):
        raise TasteT9ManagedV2Error("T9 Git commit/tree identity is malformed")
    return {"commit": observed["commit"], "tree": observed["tree"]}


def _validate_t9_input_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema_version",
        "trust_model",
        "operator",
        "execution",
        "config",
        "gpu",
        "t2_adoption_binding",
        "t3_stage_evidence",
        "t4_stage_evidence",
        "checkpoint",
        "train",
        "official",
        "data_access",
    }
    if type(value) is not dict or set(value) != keys:
        raise TasteT9ManagedV2Error("T9 trusted input authority keys changed")
    operator = value.get("operator")
    execution = value.get("execution")
    config = value.get("config")
    gpu = value.get("gpu")
    checkpoint = value.get("checkpoint")
    train = value.get("train")
    official = value.get("official")
    access = value.get("data_access")
    official_keys = {
        "schema_version",
        "commit",
        "root",
        "root_identity",
        "file_sha256",
        "module_names",
        "descriptor_loaded",
    }
    official_modules = [
        "util",
        "data",
        "neurosed.models",
        "distance",
        "gnn",
        "comrecgc",
        "common_recourse",
    ]
    if (
        value.get("schema_version") != T9_INPUT_AUTHORITY_SCHEMA
        or value.get("trust_model") != TRUST_MODEL
        or type(operator) is not dict
        or set(operator) != {"run_id", "task_id"}
        or type(operator.get("run_id")) is not str
        or _SAFE_RUN_RE.fullmatch(operator["run_id"]) is None
        or operator.get("task_id") != TASK_ID
        or type(execution) is not dict
        or set(execution) != {"commit", "tree"}
        or _SHA1_RE.fullmatch(str(execution.get("commit"))) is None
        or _SHA1_RE.fullmatch(str(execution.get("tree"))) is None
        or type(config) is not dict
        or set(config) != {"path", "sha256"}
        or config.get("path") != str(REPO_ROOT / "configs/hpc.yaml")
        or type(gpu) is not dict
        or set(gpu) != {"physical_index", "uuid", "logical_device"}
        or type(gpu.get("physical_index")) is not int
        or gpu["physical_index"] != PHYSICAL_GPU_INDEX
        or type(gpu.get("uuid")) is not str
        or _GPU_UUID_RE.fullmatch(gpu["uuid"]) is None
        or gpu.get("logical_device") != "cuda:0"
        or type(checkpoint) is not dict
        or type(train) is not dict
        or type(official) is not dict
        or set(official) != official_keys
        or official.get("schema_version")
        != "comrecgc_held_official_sources_v1"
        or official.get("commit") != OFFICIAL_COMRECGC_COMMIT
        or type(official.get("root")) is not str
        or not Path(official["root"]).is_absolute()
        or type(official.get("root_identity")) is not dict
        or set(official["root_identity"])
        != {"device", "inode", "mode", "uid", "gid"}
        or any(
            type(item) is not int or item < 0
            for item in official["root_identity"].values()
        )
        or official.get("file_sha256") != dict(OFFICIAL_SOURCE_SHA256)
        or official.get("module_names") != official_modules
        or official.get("descriptor_loaded") is not True
        or access
        != {
            "train_loaded": True,
            "validation_loaded": False,
            "calibration_payload_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "dataset_redistributed": False,
        }
    ):
        raise TasteT9ManagedV2Error("T9 trusted input authority values changed")
    _require_sha256(config.get("sha256"), label="config SHA-256")
    for name, digest in official["file_sha256"].items():
        _require_sha256(digest, label=f"official {name} SHA-256")
    t2 = value["t2_adoption_binding"]
    t3 = value["t3_stage_evidence"]
    t4 = value["t4_stage_evidence"]
    if (
        type(t2) is not dict
        or set(t2)
        != {
            "schema_version",
            "status",
            "root",
            "receipt_id",
            "gate_sha256",
            "source_evidence_file_sha256",
            "source_evidence_sha256",
            "receipt_inventory_sha256",
            "file_sha256",
        }
        or t2.get("schema_version") != "tastemolnet_t9_t2_receipt_binding_v1"
        or t2.get("status") != "PASS"
        or type(t2.get("root")) is not str
        or not Path(t2["root"]).is_absolute()
        or type(t2.get("receipt_id")) is not str
        or not t2["receipt_id"]
        or type(t2.get("file_sha256")) is not dict
        or set(t2["file_sha256"]) != _T2_RECEIPT_FILES
    ):
        raise TasteT9ManagedV2Error("T9 T2 receipt authority changed")
    for key in (
        "gate_sha256",
        "source_evidence_file_sha256",
        "source_evidence_sha256",
        "receipt_inventory_sha256",
    ):
        _require_sha256(t2.get(key), label=f"T2 {key}")
    for name, digest in t2["file_sha256"].items():
        _require_sha256(digest, label=f"T2 {name} SHA-256")
    if (
        type(t3) is not dict
        or t3.get("schema_version") != "tastemolnet_t4_t3_binding_v2"
        or t3.get("source_t2_receipt_id") != t2["receipt_id"]
        or t3.get("source_t2_gate_sha256") != t2["gate_sha256"]
        or t3.get("source_t2_evidence_sha256")
        != t2["source_evidence_sha256"]
        or t3.get("calibration_payload_loaded") is not False
        or t3.get("test_payload_loaded") is not False
        or t3.get("rf_oracle_used") is not False
    ):
        raise TasteT9ManagedV2Error("T9 T3 managed authority changed")
    if (
        type(t4) is not dict
        or t4.get("schema_version")
        != "tastemolnet_t9_t4_managed_binding_v1"
        or t4.get("status") != "PASS"
        or type(t4.get("science")) is not dict
    ):
        raise TasteT9ManagedV2Error("T9 T4 managed authority changed")
    t4_science = t4["science"]
    for left, right in (
        (t4_science.get("t3_root"), t3.get("t3_root")),
        (t4_science.get("t3_gate_sha256"), t3.get("t3_gate_sha256")),
        (
            t4_science.get("t3_verification_sha256"),
            t3.get("t3_verification_sha256"),
        ),
        (t4_science.get("checkpoint_dir"), t3.get("checkpoint_dir")),
        (t4_science.get("checkpoint_id"), t3.get("checkpoint_id")),
        (
            t4_science.get("temperature_scaling_sha256"),
            t3.get("temperature_scaling_sha256"),
        ),
        (
            t4_science.get("feature_schema_sha256"),
            t3.get("feature_schema_sha256"),
        ),
    ):
        if left != right:
            raise TasteT9ManagedV2Error("T9 T4 differs from held T3")
    expected_checkpoint_keys = {
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "payload_sha256",
    }
    if set(checkpoint) != expected_checkpoint_keys:
        raise TasteT9ManagedV2Error("T9 checkpoint authority keys changed")
    for field in expected_checkpoint_keys - {"payload_sha256"}:
        if checkpoint[field] != t3.get(field):
            raise TasteT9ManagedV2Error("T9 checkpoint differs from T3/T4")
    payload_hashes = checkpoint.get("payload_sha256")
    if type(payload_hashes) is not dict or set(payload_hashes) != set(
        _AUTHORITY_CHECKPOINT_PAYLOADS
    ):
        raise TasteT9ManagedV2Error("T9 checkpoint payload inventory changed")
    for name, digest in payload_hashes.items():
        _require_sha256(digest, label=f"checkpoint {name} SHA-256")
    expected_train_keys = {
        "path",
        "sha256",
        "num_records",
        "label_counts",
        "graph_schema_sha256",
        "sweet_source_pool_count",
    }
    if (
        set(train) != expected_train_keys
        or type(train.get("path")) is not str
        or not Path(train["path"]).is_absolute()
        or type(train.get("num_records")) is not int
        or train["num_records"] <= 0
        or type(train.get("label_counts")) is not dict
        or type(train.get("sweet_source_pool_count")) is not int
        or train["sweet_source_pool_count"] != SMOKE_SOURCE_POOL
    ):
        raise TasteT9ManagedV2Error("T9 train authority changed")
    _require_sha256(train.get("sha256"), label="train CSV SHA-256")
    _require_sha256(
        train.get("graph_schema_sha256"), label="train graph schema SHA-256"
    )
    return json.loads(json.dumps(value))


def t9_managed_input_hashes(authority: Mapping[str, Any]) -> dict[str, str]:
    frozen = _validate_t9_input_authority(authority)
    names = (
        "operator",
        "execution",
        "config",
        "gpu",
        "t2_adoption_binding",
        "t3_stage_evidence",
        "t4_stage_evidence",
        "checkpoint",
        "train",
        "official",
        "data_access",
    )
    return {f"authority.{name}": _sha256_mapping(frozen[name]) for name in names}


def _current_t2_binding(receipt: Any) -> dict[str, Any]:
    """Reopen the held current-campaign seven-file T2 receipt."""

    from src.utils.tastemolnet_t2_adoption_v2 import PASS_MARKER as T2_PASS_MARKER

    receipt.verify()
    files = {name: item.sha256 for name, item in sorted(receipt.files.items())}
    if set(files) != _T2_RECEIPT_FILES:
        raise TasteT9ManagedV2Error("T9 T2 receipt inventory changed")
    gate = _json_object(receipt.files["gate.json"].bytes(), label="T2 gate")
    source = _json_object(
        receipt.files["source_evidence.json"].bytes(), label="T2 source evidence"
    )
    verification = _json_object(
        receipt.files["verification.json"].bytes(), label="T2 verification"
    )
    source_sha = source.get("source_evidence_sha256")
    if (
        receipt.files["PASS"].bytes() != (T2_PASS_MARKER + "\n").encode("utf-8")
        or gate.get("status") != "PASS"
        or gate.get("state") != "ADOPTED_SCIENTIFIC_PASS"
        or gate.get("stage") != "T2_GINE"
        or gate.get("receipt_id") != receipt.root.name
        or gate.get("marker") != T2_PASS_MARKER
        or gate.get("source_evidence_sha256") != source_sha
        or source.get("receipt_id") != receipt.root.name
        or source.get("calibration_loaded") is not False
        or source.get("test_loaded") is not False
        or source.get("rf_oracle_used") is not False
        or verification.get("verification_result") != "PASS"
        or verification.get("source_evidence_sha256") != source_sha
    ):
        raise TasteT9ManagedV2Error("T9 T2 managed receipt changed")
    _require_sha256(source_sha, label="T2 embedded source evidence SHA-256")
    result = {
        "schema_version": "tastemolnet_t9_t2_receipt_binding_v1",
        "status": "PASS",
        "root": str(receipt.root),
        "receipt_id": receipt.root.name,
        "gate_sha256": files["gate.json"],
        "source_evidence_file_sha256": files["source_evidence.json"],
        "source_evidence_sha256": source_sha,
        "receipt_inventory_sha256": _sha256_mapping(files),
        "file_sha256": files,
    }
    receipt.verify()
    return result


@dataclass(slots=True)
class HeldTasteT4Final:
    """Retain one independently verified managed-v2 T4 publication."""

    root: Path
    root_descriptor: int
    tree: RetainedOutputTree
    evidence: Mapping[str, Any]

    def revalidate(self) -> dict[str, Any]:
        from src.eval.tastemolnet_t4_oracle_smoke_v2 import (
            MIN_FLIPPED_PARENTS,
            MIN_STRICT_FLIPS,
            PASS_MARKER as T4_MARKER,
            SCHEMA_VERSION as T4_SCHEMA,
            STAGE as T4_STAGE,
        )
        from src.utils.managed_execution_v2 import GATE_SCHEMA, VERIFICATION_SCHEMA

        inventory = self.tree.revalidate()
        gate_data = self.tree.read_bytes("gate.json")
        verification_data = self.tree.read_bytes("verification.json")
        gate = _json_object(gate_data, label="T4 gate")
        verification = _json_object(
            verification_data, label="T4 independent verification"
        )
        science = verification.get("verification")
        if (
            gate.get("schema_version") != GATE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("independent_verifier") is not True
            or gate.get("science_adopted") is not True
            or gate.get("downstream_released") is not True
            or gate.get("auto_terminate_uncontrolled_children") is not False
            or gate.get("verification_sha256") != _sha256_bytes(verification_data)
            or gate.get("sealed_sha256")
            != _sha256_bytes(self.tree.read_bytes("SEALED.json"))
            or verification.get("schema_version") != VERIFICATION_SCHEMA
            or verification.get("status") != "PASS"
            or verification.get("independent_verifier") is not True
            or verification.get("attempt_id") != gate.get("attempt_id")
            or verification.get("generation_token")
            != gate.get("generation_token")
            or verification.get("sealed_sha256") != gate.get("sealed_sha256")
            or verification.get("published_inventory_sha256")
            != gate.get("published_inventory_sha256")
            or self.tree.read_bytes("PASS") != b"[MANAGED_EXECUTION_V2_PASS]\n"
            or type(science) is not dict
            or science.get("schema_version") != T4_SCHEMA
            or science.get("status") != "PASS"
            or science.get("stage") != T4_STAGE
            or science.get("marker") != T4_MARKER
            or science.get("independent_scientific_verifier") is not True
            or science.get("adaptive_calibration_search") is not True
            or science.get("strict_flip_gate_pass") is not True
            or type(science.get("strict_flip_count")) is not int
            or science["strict_flip_count"] < MIN_STRICT_FLIPS
            or type(science.get("distinct_flipped_parent_count")) is not int
            or science["distinct_flipped_parent_count"] < MIN_FLIPPED_PARENTS
            or science.get("physical_gpu_index") != PHYSICAL_GPU_INDEX
            or science.get("visible_device") != "cuda:0"
            or science.get("train_payload_loaded") is not False
            or science.get("validation_payload_loaded") is not False
            or science.get("test_payload_loaded") is not False
            or science.get("rf_oracle_used") is not False
            or science.get("per_example_output_written") is not False
            or science.get("matrix_method_cell") is not False
        ):
            raise TasteT9ManagedV2Error("T9 T4 managed predecessor changed")
        for key in (
            "t3_gate_sha256",
            "t3_verification_sha256",
            "checkpoint_id",
            "temperature_scaling_sha256",
            "feature_schema_sha256",
        ):
            _require_sha256(science.get(key), label=f"T4 {key}")
        result = {
            "schema_version": "tastemolnet_t9_t4_managed_binding_v1",
            "status": "PASS",
            "root": str(self.root),
            "root_inventory_sha256": inventory["inventory_sha256"],
            "gate_sha256": _sha256_bytes(gate_data),
            "verification_sha256": _sha256_bytes(verification_data),
            "sealed_sha256": gate["sealed_sha256"],
            "attempt_id": gate["attempt_id"],
            "generation_token": gate["generation_token"],
            "science": json.loads(json.dumps(science)),
        }
        if self.evidence and result != dict(self.evidence):
            raise TasteT9ManagedV2Error("T9 retained T4 authority changed")
        return result

    def close(self) -> None:
        self.tree.close()
        if self.root_descriptor >= 0:
            os.close(self.root_descriptor)
            self.root_descriptor = -1

    def __enter__(self) -> "HeldTasteT4Final":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_t4_managed_final(root: str | Path) -> HeldTasteT4Final:
    selected = _absolute(root, label="managed T4 root")
    if not selected.name.startswith("t4-oracle-smoke-"):
        raise TasteT9ManagedV2Error("T9 T4 root name changed")
    descriptor = os.open(
        selected,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    tree: RetainedOutputTree | None = None
    try:
        tree = RetainedOutputTree.capture(descriptor)
        provisional = HeldTasteT4Final(selected, descriptor, tree, {})
        evidence = provisional.revalidate()
        result = HeldTasteT4Final(selected, descriptor, tree, evidence)
        result.revalidate()
        return result
    except BaseException:
        if tree is not None:
            tree.close()
        os.close(descriptor)
        raise


@dataclass(slots=True)
class HeldTasteT9Inputs:
    stack: ExitStack
    t2: Any
    t3: Any
    t4: HeldTasteT4Final
    train_file: Any
    config_file: Any
    official: HeldImportedCOMRECGC
    checkpoint_payloads: Mapping[str, bytes]
    source_rows: tuple[Any, ...]
    graph_schema: Any
    authority: Mapping[str, Any]

    def revalidate(self) -> dict[str, Any]:
        self.config_file.revalidate()
        t2 = _current_t2_binding(self.t2)
        self.t3.verify()
        payload_hashes = {
            name: _sha256_bytes(
                self.t3.files[f"artifacts/checkpoint/{name}"].bytes()
            )
            for name in _AUTHORITY_CHECKPOINT_PAYLOADS
        }
        t3 = dict(self.t3.binding)
        t3["checkpoint_inventory_sha256"] = _sha256_mapping(payload_hashes)
        t4 = self.t4.revalidate()
        self.train_file.revalidate()
        official = self.official.revalidate()
        execution = inspect_clean_execution()
        if (
            t2 != self.authority["t2_adoption_binding"]
            or t3 != self.authority["t3_stage_evidence"]
            or t4 != self.authority["t4_stage_evidence"]
            or official != self.authority["official"]
            or execution != self.authority["execution"]
            or payload_hashes != self.authority["checkpoint"]["payload_sha256"]
            or _sha256_mapping(payload_hashes)
            != self.authority["checkpoint"]["checkpoint_inventory_sha256"]
        ):
            raise TasteT9ManagedV2Error("T9 retained input authority changed")
        return _validate_t9_input_authority(self.authority)

    def close(self) -> None:
        self.stack.close()

    def __enter__(self) -> "HeldTasteT9Inputs":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def hold_t9_inputs(
    *,
    config_path: str | Path,
    run_id: str,
    gpu_uuid: str,
    t2_adoption_root: str | Path,
    t2_adoption_gate_sha256: str,
    t2_adoption_receipt_sha256: str,
    t2_source_evidence_sha256: str,
    t3_output_root: str | Path,
    t4_output_root: str | Path,
    checkpoint_dir: str | Path,
    train_csv: str | Path,
    official_root: str | Path,
) -> HeldTasteT9Inputs:
    """Retain every exact T9 scientific input without opening held-out data."""

    if type(run_id) is not str or _SAFE_RUN_RE.fullmatch(run_id) is None:
        raise TasteT9ManagedV2Error("T9 run_id is malformed")
    if type(gpu_uuid) is not str or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise TasteT9ManagedV2Error("T9 GPU UUID is malformed")
    config = _absolute(config_path, label="T9 config")
    if config != REPO_ROOT / "configs/hpc.yaml":
        raise TasteT9ManagedV2Error("T9 requires this checkout's configs/hpc.yaml")
    stack = ExitStack()
    try:
        from src.baselines.tastemolnet_gcf_smoke import load_train_rows
        from src.eval.tastemolnet_t3_calibration_v2 import HeldT2Receipt
        from src.eval.tastemolnet_t4_oracle_smoke_v2 import HeldPublishedT3
        from src.utils.retained_readonly_file import hold_readonly_file

        config_file = stack.enter_context(
            hold_readonly_file(config, expected_sha256=_sha256_bytes(config.read_bytes()))
        )
        t2 = HeldT2Receipt(
            _absolute(t2_adoption_root, label="T2 adoption root")
        )
        stack.callback(t2.close)
        t2_evidence = _current_t2_binding(t2)
        if (
            _require_sha256(
                t2_adoption_gate_sha256, label="T2 gate SHA-256"
            )
            != t2_evidence["gate_sha256"]
            or _require_sha256(
                t2_adoption_receipt_sha256,
                label="T2 receipt-inventory SHA-256",
            )
            != t2_evidence["receipt_inventory_sha256"]
            or _require_sha256(
                t2_source_evidence_sha256,
                label="T2 source-evidence file SHA-256",
            )
            != t2_evidence["source_evidence_file_sha256"]
        ):
            raise TasteT9ManagedV2Error("T9 T2 command-line pins changed")

        t3 = HeldPublishedT3(_absolute(t3_output_root, label="T3 root"))
        stack.callback(t3.close)
        t3.verify()
        payloads = {
            name: t3.files[f"artifacts/checkpoint/{name}"].bytes()
            for name in _AUTHORITY_CHECKPOINT_PAYLOADS
        }
        payload_hashes = {
            name: _sha256_bytes(data) for name, data in sorted(payloads.items())
        }
        t3_evidence = dict(t3.binding)
        t3_evidence["checkpoint_inventory_sha256"] = _sha256_mapping(
            payload_hashes
        )

        t4 = stack.enter_context(
            hold_t4_managed_final(_absolute(t4_output_root, label="T4 root"))
        )
        t4_evidence = t4.revalidate()
        selected_checkpoint = _absolute(checkpoint_dir, label="GINE checkpoint")
        if selected_checkpoint != Path(t3_evidence["checkpoint_dir"]):
            raise TasteT9ManagedV2Error(
                "T9 checkpoint must be the held managed-v2 T3 checkpoint"
            )
        split = _json_object(payloads["split_manifest.json"], label="split manifest")
        files = split.get("files")
        train_manifest = split.get("train_manifest")
        if (
            split.get("schema_version") != "molecular_gnn_split_manifest_v1"
            or split.get("dataset") != DATASET
            or split.get("calibration_loaded_for_training") is not False
            or split.get("test_loaded_for_training") is not False
            or split.get("test_evaluated_during_training") is not False
            or split.get("test_used_for_checkpoint_selection") is not False
            or type(files) is not dict
            or type(files.get("train")) is not dict
            or type(train_manifest) is not dict
        ):
            raise TasteT9ManagedV2Error("T9 frozen split isolation changed")
        frozen_train = _absolute(files["train"].get("path"), label="frozen train CSV")
        requested_train = _absolute(train_csv, label="requested train CSV")
        if frozen_train != requested_train:
            raise TasteT9ManagedV2Error("T9 train path differs from frozen GINE")
        train_sha = _require_sha256(
            files["train"].get("sha256"), label="frozen train SHA-256"
        )
        train_file = stack.enter_context(
            hold_readonly_file(frozen_train, expected_sha256=train_sha)
        )
        train_bytes = train_file.read_bytes()
        num_records = train_manifest.get("num_records")
        label_counts = train_manifest.get("label_counts")
        if type(num_records) is not int or num_records <= 0 or type(label_counts) is not dict:
            raise TasteT9ManagedV2Error("T9 frozen train manifest changed")
        loaded = load_train_rows(
            train_bytes,
            source_path=frozen_train,
            expected_num_records=num_records,
            expected_label_counts=label_counts,
        )
        if len(loaded.sweet_rows) < SMOKE_SOURCE_POOL:
            raise TasteT9ManagedV2Error("T9 train cohort lacks 64 Sweet sources")
        official = stack.enter_context(
            hold_imported_comrecgc(
                _absolute(official_root, label="official COMRECGC root"),
                expected_file_sha256=dict(OFFICIAL_SOURCE_SHA256),
            )
        )
        official_evidence = official.revalidate()
        execution = inspect_clean_execution()
        authority = {
            "schema_version": T9_INPUT_AUTHORITY_SCHEMA,
            "trust_model": TRUST_MODEL,
            "operator": {"run_id": run_id, "task_id": TASK_ID},
            "execution": execution,
            "config": {
                "path": str(config),
                "sha256": config_file.sha256,
            },
            "gpu": {
                "physical_index": PHYSICAL_GPU_INDEX,
                "uuid": gpu_uuid,
                "logical_device": "cuda:0",
            },
            "t2_adoption_binding": t2_evidence,
            "t3_stage_evidence": t3_evidence,
            "t4_stage_evidence": t4_evidence,
            "checkpoint": {
                "checkpoint_dir": str(selected_checkpoint),
                "checkpoint_id": t3_evidence["checkpoint_id"],
                "checkpoint_inventory_sha256": t3_evidence[
                    "checkpoint_inventory_sha256"
                ],
                "checkpoint_sha256s_sha256": t3_evidence[
                    "checkpoint_sha256s_sha256"
                ],
                "payload_sha256": payload_hashes,
            },
            "train": {
                "path": str(frozen_train),
                "sha256": train_sha,
                "num_records": num_records,
                "label_counts": label_counts,
                "graph_schema_sha256": loaded.evidence["graph_schema_sha256"],
                "sweet_source_pool_count": SMOKE_SOURCE_POOL,
            },
            "official": official_evidence,
            "data_access": {
                "train_loaded": True,
                "validation_loaded": False,
                "calibration_payload_loaded": False,
                "test_loaded": False,
                "rf_oracle_used": False,
                "dataset_redistributed": False,
            },
        }
        frozen_authority = _validate_t9_input_authority(authority)
        held = HeldTasteT9Inputs(
            stack=stack,
            t2=t2,
            t3=t3,
            t4=t4,
            train_file=train_file,
            config_file=config_file,
            official=official,
            checkpoint_payloads=payloads,
            source_rows=tuple(loaded.sweet_rows[:SMOKE_SOURCE_POOL]),
            graph_schema=loaded.schema,
            authority=frozen_authority,
        )
        held.revalidate()
        return held
    except BaseException:
        stack.close()
        raise


def require_gpu1_runtime(gpu_uuid: str) -> None:
    """Require the trusted launcher to expose physical GPU1 as logical cuda:0."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise TasteT9ManagedV2Error("T9 requires CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != str(PHYSICAL_GPU_INDEX):
        raise TasteT9ManagedV2Error("T9 physical GPU index binding is absent")
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != gpu_uuid:
        raise TasteT9ManagedV2Error("T9 physical GPU UUID binding changed")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TasteT9ManagedV2Error("T9 requires exactly one visible CUDA device")


def _write_artifact_json(
    staging: HeldWorkerStagingV2, name: str, value: Mapping[str, Any]
) -> None:
    if "/" in name or name in {"", ".", "..", ".generation_token.json"}:
        raise TasteT9ManagedV2Error("T9 artifact name is unsafe")
    staging.revalidate()
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=staging.artifact_descriptor,
    )
    try:
        data = _json_bytes(value)
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise TasteT9ManagedV2Error("T9 artifact write was short")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(staging.artifact_descriptor)


def _validate_attempt_manifest(
    manifest: Mapping[str, Any],
    *,
    attempt_id: str,
    attempt_path: Path,
    attempt_generation_token: str,
    authority: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "attempt_id",
        "controller_id",
        "task_id",
        "git_commit",
        "config_hash",
        "input_hashes",
        "created_at",
        "hostname",
        "boot_id",
        "attempt_path",
        "generation_token",
        "auto_terminate_uncontrolled_children",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != ATTEMPT_MANIFEST_SCHEMA
        or manifest.get("status") != "ACTIVE"
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("controller_id") != authority["operator"]["run_id"]
        or manifest.get("task_id") != TASK_ID
        or manifest.get("git_commit") != authority["execution"]["commit"]
        or manifest.get("config_hash") != authority["config"]["sha256"]
        or manifest.get("input_hashes") != t9_managed_input_hashes(authority)
        or manifest.get("attempt_path") != str(attempt_path)
        or manifest.get("generation_token") != attempt_generation_token
        or manifest.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteT9ManagedV2Error("T9 managed attempt manifest changed")


def seal_t9_worker_evidence(
    staging: HeldWorkerStagingV2,
    *,
    science: Mapping[str, Any],
    input_authority: Mapping[str, Any],
    expected_final_path: str | Path,
) -> SealedWorkerArtifactV2:
    """Worker-only aggregate evidence close; never writes verifier outputs."""

    staging.revalidate()
    frozen_science = validate_native_comrecgc_smoke_result(dict(science))
    authority = _validate_t9_input_authority(input_authority)
    final_path = _absolute(
        expected_final_path, label="T9 final path", must_exist=False
    )
    _absolute(final_path.parent, label="T9 final parent")
    if final_path.exists() or final_path.is_symlink():
        raise TasteT9ManagedV2Error("T9 final path must remain fresh")
    manifest = dict(staging.attempt.revalidate())
    _validate_attempt_manifest(
        manifest,
        attempt_id=staging.attempt.attempt_id,
        attempt_path=staging.attempt.attempt_path,
        attempt_generation_token=staging.attempt.generation_token,
        authority=authority,
    )
    _write_artifact_json(staging, "comrecgc_smoke.json", frozen_science)
    _write_artifact_json(staging, "input_authority.json", authority)
    raw = write_worker_raw_evidence(
        staging,
        {
            "schema_version": T9_WORKER_RAW_SCHEMA,
            "status": "RAW_EVIDENCE_ONLY",
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "task_id": TASK_ID,
            "trust_model": TRUST_MODEL,
            "expected_final_path": str(final_path),
            "attempt_manifest": manifest,
            "input_authority_sha256": _sha256_mapping(authority),
            "science_sha256": _sha256_mapping(frozen_science),
            "science_file": SCIENCE_FILE,
            "input_authority_file": INPUT_AUTHORITY_FILE,
            "worker_wrote_verification": False,
            "worker_wrote_gate": False,
            "worker_wrote_pass": False,
            "independent_verification_required": True,
            "data_redistributed": False,
            "test_loaded": False,
        },
    )
    raw.close()
    worker_exit = write_worker_exit(
        staging,
        {
            "exit_code": 0,
            "science_complete": True,
            "random_walk_steps": 500,
            "checkpoint_reload_step": 250,
            "worker_closed_science_state_writers": True,
            "worker_wrote_verifier_output": False,
        },
    )
    worker_exit.close()
    return seal_worker_staging(staging)


def run_t9_worker(
    *,
    stage_root: str | Path,
    final_path: str | Path,
    inputs: HeldTasteT9Inputs,
) -> dict[str, Any]:
    """Execute the frozen M=500 science core and return only a SEALED receipt."""

    authority = inputs.revalidate()
    require_gpu1_runtime(authority["gpu"]["uuid"])
    root = _absolute(stage_root, label="T9 managed stage root")
    with create_managed_attempt(
        stage_root=root,
        controller_id=authority["operator"]["run_id"],
        task_id=TASK_ID,
        git_commit=authority["execution"]["commit"],
        config_hash=authority["config"]["sha256"],
        input_hashes=t9_managed_input_hashes(authority),
    ) as attempt, create_worker_staging(attempt) as staging:
        inputs.revalidate()
        science = execute_native_comrecgc_smoke(
            modules=inputs.official.modules,
            checkpoint_payloads=_checkpoint_payloads_for_model_load(
                inputs.checkpoint_payloads
            ),
            source_rows=inputs.source_rows,
            graph_schema=inputs.graph_schema,
            device="cuda:0",
        )
        inputs.revalidate()
        sealed = seal_t9_worker_evidence(
            staging,
            science=science,
            input_authority=authority,
            expected_final_path=final_path,
        )
        return {
            "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION",
            "stage": STAGE,
            "attempt_id": sealed.attempt_id,
            "generation_token": sealed.generation_token,
            "staging_path": str(sealed.staging_path),
            "seal_path": str(sealed.seal_path),
            "seal_sha256": sealed.seal_sha256,
            "inventory_sha256": sealed.inventory_sha256,
            "expected_final_path": str(
                _absolute(final_path, label="T9 final path", must_exist=False)
            ),
        }


def _read_held_json(held: HeldSealedArtifactV2, relative_path: str) -> dict[str, Any]:
    matches = [item for item in held.files if item.evidence.relative_path == relative_path]
    if len(matches) != 1:
        raise TasteT9ManagedV2Error(f"T9 SEALED evidence lacks {relative_path}")
    item = matches[0]
    item.revalidate()
    data = bytearray()
    offset = 0
    while offset < item.evidence.size:
        block = os.pread(
            item.descriptor,
            min(1024 * 1024, item.evidence.size - offset),
            offset,
        )
        if not block:
            raise TasteT9ManagedV2Error("T9 SEALED JSON ended early")
        data.extend(block)
        offset += len(block)
    if os.pread(item.descriptor, 1, item.evidence.size):
        raise TasteT9ManagedV2Error("T9 SEALED JSON grew")
    value = _json_object(bytes(data), label=relative_path)
    if canonical_json_bytes(value) != bytes(data):
        raise TasteT9ManagedV2Error(f"{relative_path} is not canonical JSON")
    item.revalidate()
    return value


def _validate_sealed_t9(
    held: HeldSealedArtifactV2,
    *,
    expected_authority: Mapping[str, Any],
    final_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = _validate_t9_input_authority(expected_authority)
    held.revalidate()
    files = {item.evidence.relative_path for item in held.files}
    expected_files = {
        ".generation_token.json",
        "artifacts/.generation_token.json",
        SCIENCE_FILE,
        INPUT_AUTHORITY_FILE,
        "raw_evidence.json",
        "worker_exit.json",
    }
    directories = {item.relative_path for item in held.inventory.directories}
    if files != expected_files or directories != {"artifacts"}:
        raise TasteT9ManagedV2Error("T9 SEALED aggregate-only file set changed")
    science = validate_native_comrecgc_smoke_result(
        _read_held_json(held, SCIENCE_FILE)
    )
    persisted_authority = _validate_t9_input_authority(
        _read_held_json(held, INPUT_AUTHORITY_FILE)
    )
    raw = _read_held_json(held, "raw_evidence.json")
    worker_exit = _read_held_json(held, "worker_exit.json")
    if persisted_authority != authority:
        raise TasteT9ManagedV2Error("T9 worker input authority differs from verifier")
    evidence = raw.get("evidence")
    expected_evidence = {
        "schema_version": T9_WORKER_RAW_SCHEMA,
        "status": "RAW_EVIDENCE_ONLY",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": TASK_ID,
        "trust_model": TRUST_MODEL,
        "expected_final_path": str(final_path),
        "attempt_manifest": raw.get("evidence", {}).get("attempt_manifest"),
        "input_authority_sha256": _sha256_mapping(authority),
        "science_sha256": _sha256_mapping(science),
        "science_file": SCIENCE_FILE,
        "input_authority_file": INPUT_AUTHORITY_FILE,
        "worker_wrote_verification": False,
        "worker_wrote_gate": False,
        "worker_wrote_pass": False,
        "independent_verification_required": True,
        "data_redistributed": False,
        "test_loaded": False,
    }
    if (
        set(raw)
        != {"schema_version", "attempt_id", "generation_token", "recorded_at", "evidence"}
        or raw.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
        or raw.get("attempt_id") != held.sealed.attempt_id
        or raw.get("generation_token") != held.sealed.generation_token
        or type(raw.get("recorded_at")) is not str
        or not raw["recorded_at"]
        or evidence != expected_evidence
    ):
        raise TasteT9ManagedV2Error("T9 worker raw evidence changed")
    attempt_path = held.staging_path.parent.parent
    manifest = evidence["attempt_manifest"]
    _validate_attempt_manifest(
        manifest,
        attempt_id=held.sealed.attempt_id,
        attempt_path=attempt_path,
        attempt_generation_token=str(manifest.get("generation_token")),
        authority=authority,
    )
    exit_payload = worker_exit.get("exit")
    if (
        set(worker_exit)
        != {"schema_version", "attempt_id", "generation_token", "recorded_at", "exit"}
        or worker_exit.get("schema_version") != WORKER_EXIT_SCHEMA
        or worker_exit.get("attempt_id") != held.sealed.attempt_id
        or worker_exit.get("generation_token") != held.sealed.generation_token
        or type(worker_exit.get("recorded_at")) is not str
        or not worker_exit["recorded_at"]
        or exit_payload
        != {
            "exit_code": 0,
            "science_complete": True,
            "random_walk_steps": 500,
            "checkpoint_reload_step": 250,
            "worker_closed_science_state_writers": True,
            "worker_wrote_verifier_output": False,
        }
    ):
        raise TasteT9ManagedV2Error("T9 worker exit evidence changed")
    held.revalidate()
    return science, authority


def verify_and_publish_t9_sealed(
    held: HeldSealedArtifactV2,
    *,
    final_path: str | Path,
    expected_authority: Mapping[str, Any],
    revalidate_inputs: Callable[[], Mapping[str, Any]],
    force_cross_filesystem: bool = False,
) -> tuple[TerminalPublicationV2, dict[str, Any]]:
    """Independently verify and atomically publish one aggregate-only T9 smoke."""

    destination = _absolute(final_path, label="T9 final path", must_exist=False)
    _absolute(destination.parent, label="T9 final parent")
    if destination.exists() or destination.is_symlink():
        raise TasteT9ManagedV2Error("T9 final path is not fresh")
    authority = _validate_t9_input_authority(expected_authority)
    if _validate_t9_input_authority(revalidate_inputs()) != authority:
        raise TasteT9ManagedV2Error("T9 verifier input authority changed")
    science, sealed_authority = _validate_sealed_t9(
        held, expected_authority=authority, final_path=destination
    )
    if sealed_authority != authority:
        raise TasteT9ManagedV2Error("T9 sealed authority changed")
    if _validate_t9_input_authority(revalidate_inputs()) != authority:
        raise TasteT9ManagedV2Error("T9 verifier inputs changed before publication")
    verification = {
        "schema_version": T9_VERIFICATION_SCHEMA,
        "status": "PASS",
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "task_id": TASK_ID,
        "domain_marker": PASS_MARKER,
        "trust_model": TRUST_MODEL,
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "gpu_uuid": authority["gpu"]["uuid"],
        "execution_commit": authority["execution"]["commit"],
        "execution_tree": authority["execution"]["tree"],
        "input_authority_sha256": _sha256_mapping(authority),
        "science_sha256": _sha256_mapping(science),
        "random_walk_steps": science["random_walk_steps"],
        "checkpoint_step": science["checkpoint_reload"]["checkpoint_step"],
        "strict_counterfactual_count": science["bridge"][
            "evaluated_strict_graph_count"
        ],
        "same_three_class_gine": True,
        "official_comrecgc_commit": OFFICIAL_COMRECGC_COMMIT,
        "train_only": True,
        "validation_loaded": False,
        "calibration_payload_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "data_redistributed": False,
        "paper_result_eligible": False,
        "worker_self_signed": False,
        "independent_verifier": True,
    }
    publication = verify_and_publish_sealed_attempt(
        held,
        final_path=destination,
        verification=verification,
        force_cross_filesystem=force_cross_filesystem,
    )
    return publication, verification


def open_t9_sealed(
    path: str | Path,
    *,
    expected_attempt_id: str,
    expected_generation_token: str,
) -> HeldSealedArtifactV2:
    return open_sealed_worker_artifact(
        path,
        expected_attempt_id=expected_attempt_id,
        expected_generation_token=expected_generation_token,
    )


def load_t9_verified_gate(path: str | Path) -> Mapping[str, Any]:
    return load_verified_gate(path)


__all__ = [
    "INPUT_AUTHORITY_FILE",
    "PHYSICAL_GPU_INDEX",
    "SCIENCE_FILE",
    "T9_INPUT_AUTHORITY_SCHEMA",
    "T9_VERIFICATION_SCHEMA",
    "T9_WORKER_RAW_SCHEMA",
    "TRUST_MODEL",
    "HeldTasteT9Inputs",
    "TasteT9ManagedV2Error",
    "hold_t9_inputs",
    "inspect_clean_execution",
    "load_t9_verified_gate",
    "open_t9_sealed",
    "require_gpu1_runtime",
    "run_t9_worker",
    "seal_t9_worker_evidence",
    "t9_managed_input_hashes",
    "verify_and_publish_t9_sealed",
]
