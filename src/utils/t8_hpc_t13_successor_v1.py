"""Sealed predeployment contracts for T8 HPC import and AutoDL T13.

The contracts intentionally separate three authorities:

* the import owner may verify/copy only train-side HPC artifacts;
* the T13 owner is the only GPU science owner and performs chemistry, GINE,
  strict-flip, calibration, held-out test, and export on AutoDL;
* the existing fast16 publisher is the only matrix writer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping
import uuid

from src.baselines.globalgce_hpc_exact import OFFICIAL_GLOBALGCE_COMMIT


SPEC_SET_SCHEMA = "t8_hpc_t13_successor_spec_set_v1"
IMPORT_SPEC_SCHEMA = "t8_hpc_autodl_import_task_spec_v1"
T13_SPEC_SCHEMA = "t13_from_hpc_task_spec_v1"
PUBLISHER_SPEC_SCHEMA = "t8_hpc_t13_publisher_task_spec_v1"
RELEASE_SCHEMA = "t8_hpc_t13_release_v1"
LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
DATASET = "TasteMolNet"
METHOD = "GlobalGCE"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class T8HPCT13SpecError(RuntimeError):
    """One successor spec is unsafe, incomplete, or internally inconsistent."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    value[field] = canonical_sha256(value)
    return value


def require_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    claimed = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise T8HPCT13SpecError(f"{label} lacks a SHA-256 self-hash")
    if canonical_sha256(unsigned) != claimed:
        raise T8HPCT13SpecError(f"{label} self-hash changed")


def _absolute(value: str | Path, *, label: str, exists: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise T8HPCT13SpecError(f"{label} must be an absolute non-symlink path")
    try:
        return path.resolve(strict=exists)
    except OSError as exc:
        raise T8HPCT13SpecError(f"{label} is absent: {path}") from exc


def _executable(value: str | Path, *, label: str) -> Path:
    """Resolve an absolute interpreter path (conda commonly exposes a symlink)."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise T8HPCT13SpecError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise T8HPCT13SpecError(f"{label} is absent: {path}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise T8HPCT13SpecError(f"{label} is not executable: {resolved}")
    return resolved


def _clean_commit(repo_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if _COMMIT.fullmatch(commit) is None:
        raise T8HPCT13SpecError("execution commit is malformed")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise T8HPCT13SpecError("successor specs require a clean immutable checkout")
    return commit


def _uuid4(value: str | None = None) -> str:
    parsed = uuid.uuid4() if value is None else uuid.UUID(value)
    if parsed.version != 4 or str(parsed) != str(value or parsed):
        raise T8HPCT13SpecError("attempt ID must be canonical UUIDv4")
    return str(parsed)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_no_replace(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable claim, accepting only an identical prior claim."""

    destination = _absolute(path, label="no-replace JSON output")
    encoded = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != encoded:
                raise T8HPCT13SpecError(
                    "existing immutable publisher locator conflicts"
                )
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise T8HPCT13SpecError(f"{label} is absent or indirect")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T8HPCT13SpecError(f"{label} is malformed") from exc
    if type(value) is not dict:
        raise T8HPCT13SpecError(f"{label} must contain one object")
    return value


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_spec_set(
    *,
    repo_root: str | Path,
    python: str | Path,
    output_root: str | Path,
    relay_import_parent: str | Path,
    import_output_root: str | Path,
    t13_output_root: str | Path,
    t13_locator: str | Path,
    matrix_authority_root: str | Path,
    publisher_lease_path: str | Path,
    gpu_lease_path: str | Path,
    gnn_checkpoint: str | Path,
    train_csv: str | Path,
    calibration_csv: str | Path,
    test_csv: str | Path,
    official_root: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    threshold_contract: str | Path,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
    expected_hpc_execution_commit: str,
    expected_scientific_input_sha256: str,
    expected_partition_manifest_sha256: str,
    gpu_index: int = 1,
    gpu_uuid: str | None = None,
    import_attempt_id: str | None = None,
    t13_attempt_id: str | None = None,
    publisher_id: str = "taste-globalgce-final16-canonical",
) -> dict[str, Any]:
    """Build three immutable specs; no science or publisher is started."""

    repository = _absolute(repo_root, label="repo_root", exists=True)
    interpreter = _executable(python, label="python")
    root = _absolute(output_root, label="output_root")
    if root.exists() or root.is_symlink():
        raise T8HPCT13SpecError("spec output root must be fresh")
    execution_commit = _clean_commit(repository)
    if _COMMIT.fullmatch(expected_hpc_execution_commit) is None:
        raise T8HPCT13SpecError("expected HPC execution commit is malformed")
    if _SHA256.fullmatch(expected_scientific_input_sha256) is None:
        raise T8HPCT13SpecError("scientific input SHA is malformed")
    if _SHA256.fullmatch(expected_partition_manifest_sha256) is None:
        raise T8HPCT13SpecError("partition manifest SHA is malformed")
    if type(gpu_index) is not int or not 0 <= gpu_index <= 15:
        raise T8HPCT13SpecError("GPU index is invalid")
    if gpu_uuid is not None and not gpu_uuid.startswith("GPU-"):
        raise T8HPCT13SpecError("GPU UUID is malformed")
    if not publisher_id or "/" in publisher_id or "\x00" in publisher_id:
        raise T8HPCT13SpecError("publisher ID is malformed")
    relay_parent = _absolute(relay_import_parent, label="relay_import_parent")
    import_root = _absolute(import_output_root, label="import_output_root")
    science_root = _absolute(t13_output_root, label="t13_output_root")
    locator = _absolute(t13_locator, label="t13_locator")
    authority = _absolute(matrix_authority_root, label="matrix_authority_root")
    publisher_lease = _absolute(publisher_lease_path, label="publisher_lease_path")
    gpu_lease = _absolute(gpu_lease_path, label="gpu_lease_path")
    if len({str(import_root), str(science_root), str(root)}) != 3:
        raise T8HPCT13SpecError("spec, import, and T13 roots must be distinct")
    for path, label in (
        (import_root, "import output"),
        (science_root, "T13 output"),
        (locator, "T13 locator"),
    ):
        if path.exists() or path.is_symlink():
            raise T8HPCT13SpecError(f"{label} must be fresh")
    input_paths = {
        "gnn_checkpoint": _absolute(gnn_checkpoint, label="gnn_checkpoint"),
        "train_csv": _absolute(train_csv, label="train_csv"),
        "calibration_csv": _absolute(calibration_csv, label="calibration_csv"),
        # Merely resolving lexically does not open held-out bytes.
        "test_csv": _absolute(test_csv, label="test_csv"),
        "official_root": _absolute(official_root, label="official_root"),
        "molclr_root": _absolute(molclr_root, label="molclr_root"),
        "molclr_checkpoint": _absolute(
            molclr_checkpoint, label="molclr_checkpoint"
        ),
        "threshold_contract": _absolute(
            threshold_contract, label="threshold_contract"
        ),
        "wnode_cache_db": _absolute(wnode_cache_db, label="wnode_cache_db"),
        "node_embedding_cache_dir": _absolute(
            node_embedding_cache_dir, label="node_embedding_cache_dir"
        ),
    }
    import_attempt = _uuid4(import_attempt_id)
    t13_attempt = _uuid4(t13_attempt_id)
    created_at = _utc_now()
    import_spec = _self_hashed(
        {
            "schema_version": IMPORT_SPEC_SCHEMA,
            "status": "PREDEPLOYED_WAITING_HPC_PACKAGE",
            "task_id": f"t8-hpc-import-{import_attempt}",
            "attempt_id": import_attempt,
            "created_at": created_at,
            "execution_commit": execution_commit,
            "repo_root": str(repository),
            "python": str(interpreter),
            "relay_import_parent": str(relay_parent),
            "relay_ready_marker": "HPC_PACKAGE_READY.json",
            "output_root": str(import_root),
            "expected_hpc_execution_commit": expected_hpc_execution_commit,
            "expected_scientific_input_sha256": expected_scientific_input_sha256,
            "expected_partition_manifest_sha256": expected_partition_manifest_sha256,
            "expected_official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
            "expected_shard_count": 16,
            "package_discovery": "EXACT_IDENTITY_MATCH_UNDER_RELAY_PARENT",
            "owner_entrypoint": str(
                repository / "scripts/autodl/run_t8_hpc_import_owner_v1.py"
            ),
            "fresh_output_required": True,
            "hpc_permissions": {
                "matrix_write": False,
                "gine_inference": False,
                "calibration": False,
                "test": False,
            },
            "ready_semantics": "NO_READY_BEFORE_DEEP_IMPORT_PASS",
        },
        "task_spec_sha256",
    )
    import_manifest = import_root / "import_manifest.json"
    t13_command = [
        str(interpreter),
        str(repository / "scripts/autodl/run_t13_from_hpc_import_v1.py"),
        "--config",
        str(repository / "configs/hpc.yaml"),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--spec-root",
        str(root),
        "--hpc-import-root",
        str(import_root),
        "--gnn-checkpoint",
        str(input_paths["gnn_checkpoint"]),
        "--train-csv",
        str(input_paths["train_csv"]),
        "--calibration-csv",
        str(input_paths["calibration_csv"]),
        "--test-csv",
        str(input_paths["test_csv"]),
        "--official-root",
        str(input_paths["official_root"]),
        "--molclr-root",
        str(input_paths["molclr_root"]),
        "--molclr-checkpoint",
        str(input_paths["molclr_checkpoint"]),
        "--threshold-contract",
        str(input_paths["threshold_contract"]),
        "--wnode-cache-db",
        str(input_paths["wnode_cache_db"]),
        "--node-embedding-cache-dir",
        str(input_paths["node_embedding_cache_dir"]),
        "--output-dir",
        str(science_root),
        "--device",
        "cuda:0",
        "--epochs",
        "100",
        "--seed",
        "7",
    ]
    t13_spec = _self_hashed(
        {
            "schema_version": T13_SPEC_SCHEMA,
            "status": "PREDEPLOYED_WAITING_IMPORT_PASS",
            "task_id": f"t13-from-hpc-{t13_attempt}",
            "attempt_id": t13_attempt,
            "predecessor_task_id": import_spec["task_id"],
            "created_at": created_at,
            "execution_commit": execution_commit,
            "repo_root": str(repository),
            "python": str(interpreter),
            "required_import_root": str(import_root),
            "required_import_manifest": str(import_manifest),
            "required_import_manifest_sha256": "RESOLVE_AFTER_IMPORT_PASS",
            "output_root": str(science_root),
            "gpu_index": gpu_index,
            "gpu_uuid": gpu_uuid,
            "gpu_lease_path": str(gpu_lease),
            "owner_entrypoint": str(
                repository / "scripts/autodl/run_t13_from_hpc_owner_v1.py"
            ),
            "owner_acquires_gpu_only_after_import_pass": True,
            "command": t13_command,
            "science_contract": {
                "dataset": DATASET,
                "method": METHOD,
                "source_label": 1,
                "target_branches": [0, 2],
                "seed": 7,
                "epochs": 100,
                "min_support": 2,
                "top_k": 20,
                "hpc_stage": "EXACT_TRAIN_ONLY_GSPAN_ONLY",
                "autodl_stages": [
                    "RHS_CHEMISTRY",
                    "FROZEN_GINE_STRICT_FLIP",
                    "CALIBRATION_ONLY_SELECTION",
                    "POST_FREEZE_HELDOUT_TEST",
                    "STANDARDIZED_EXPORT",
                    "INDEPENDENT_TERMINAL_VERIFY",
                ],
                "calibration_loaded_during_training": False,
                "test_loaded_during_training": False,
                "test_used_for_selection": False,
                "zero_rule_terminal_requires_complete_search_proof": True,
            },
            "input_paths": {key: str(value) for key, value in input_paths.items()},
            "fresh_output_required": True,
            "publisher_id": publisher_id,
            "terminal_locator": str(locator),
        },
        "task_spec_sha256",
    )
    publisher_spec = _self_hashed(
        {
            "schema_version": PUBLISHER_SPEC_SCHEMA,
            "status": "PREDEPLOYED_WAITING_T13_PASS",
            "publisher_id": publisher_id,
            "cell_id": f"{DATASET}/{METHOD}",
            "dataset": DATASET,
            "method": METHOD,
            "execution_commit": execution_commit,
            "terminal_root_locator": str(locator),
            "expected_terminal_root": str(science_root),
            "locator_schema": LOCATOR_SCHEMA,
            "publisher_lease_path": str(publisher_lease),
            "matrix_authority_root": str(authority),
            "matrix_writer": "EXISTING_FAST16_MATRIX_PUBLISHER_QUEUE_ONLY",
            "claim_enabled": True,
            "unique_claim_required": True,
            "hpc_may_write_locator": False,
            "hpc_may_write_matrix": False,
        },
        "task_spec_sha256",
    )
    manifest = _self_hashed(
        {
            "schema_version": SPEC_SET_SCHEMA,
            "status": "PASS",
            "created_at": created_at,
            "execution_commit": execution_commit,
            "spec_root": str(root),
            "specs": {
                "import": "t8_hpc_import_task_spec.json",
                "t13": "t13_from_hpc_task_spec.json",
                "publisher": "t8_publisher_task_spec.json",
            },
            "task_spec_sha256s": {
                "import": import_spec["task_spec_sha256"],
                "t13": t13_spec["task_spec_sha256"],
                "publisher": publisher_spec["task_spec_sha256"],
            },
            "matrix_authority_count": 1,
            "publisher_claim_count_for_cell": 1,
            "science_started": False,
        },
        "spec_set_sha256",
    )
    root.mkdir(parents=True)
    atomic_json(root / "t8_hpc_import_task_spec.json", import_spec)
    atomic_json(root / "t13_from_hpc_task_spec.json", t13_spec)
    atomic_json(root / "t8_publisher_task_spec.json", publisher_spec)
    atomic_json(root / "spec_set_manifest.json", manifest)
    return validate_spec_set(root, check_files=False)


def validate_spec_set(
    root_like: str | Path, *, check_files: bool = True
) -> dict[str, Any]:
    root = _absolute(root_like, label="spec root", exists=True)
    manifest = read_json(root / "spec_set_manifest.json", label="spec-set manifest")
    require_self_hash(manifest, "spec_set_sha256", "spec-set manifest")
    if (
        manifest.get("schema_version") != SPEC_SET_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("spec_root") != str(root)
        or manifest.get("matrix_authority_count") != 1
        or manifest.get("publisher_claim_count_for_cell") != 1
        or manifest.get("science_started") is not False
    ):
        raise T8HPCT13SpecError("spec-set manifest contract changed")
    names = manifest.get("specs")
    if names != {
        "import": "t8_hpc_import_task_spec.json",
        "t13": "t13_from_hpc_task_spec.json",
        "publisher": "t8_publisher_task_spec.json",
    }:
        raise T8HPCT13SpecError("spec-set file inventory changed")
    specs = {
        role: read_json(root / name, label=f"{role} task spec")
        for role, name in names.items()
    }
    expected_schemas = {
        "import": IMPORT_SPEC_SCHEMA,
        "t13": T13_SPEC_SCHEMA,
        "publisher": PUBLISHER_SPEC_SCHEMA,
    }
    for role, spec in specs.items():
        require_self_hash(spec, "task_spec_sha256", f"{role} task spec")
        if (
            spec.get("schema_version") != expected_schemas[role]
            or spec.get("task_spec_sha256")
            != manifest.get("task_spec_sha256s", {}).get(role)
        ):
            raise T8HPCT13SpecError(f"{role} task spec binding changed")
    import_spec, t13_spec, publisher = (
        specs["import"],
        specs["t13"],
        specs["publisher"],
    )
    science = t13_spec.get("science_contract")
    command = t13_spec.get("command")
    repository = Path(str(t13_spec.get("repo_root") or "/invalid"))
    required_command_pairs = {
        "--spec-root": str(root),
        "--hpc-import-root": t13_spec.get("required_import_root"),
        "--output-dir": t13_spec.get("output_root"),
        "--device": "cuda:0",
        "--epochs": "100",
        "--seed": "7",
    }
    command_pairs: dict[str, Any] = {}
    if type(command) is list:
        for index, token in enumerate(command[:-1]):
            if token in required_command_pairs:
                command_pairs[token] = command[index + 1]
    if (
        import_spec.get("status") != "PREDEPLOYED_WAITING_HPC_PACKAGE"
        or import_spec.get("ready_semantics") != "NO_READY_BEFORE_DEEP_IMPORT_PASS"
        or import_spec.get("expected_shard_count") != 16
        or import_spec.get("hpc_permissions")
        != {"matrix_write": False, "gine_inference": False, "calibration": False, "test": False}
        or t13_spec.get("status") != "PREDEPLOYED_WAITING_IMPORT_PASS"
        or t13_spec.get("predecessor_task_id") != import_spec.get("task_id")
        or t13_spec.get("owner_acquires_gpu_only_after_import_pass") is not True
        or type(command) is not list
        or len(command) < 10
        or command[1]
        != str(repository / "scripts/autodl/run_t13_from_hpc_import_v1.py")
        or command_pairs != required_command_pairs
        or "--t8-pass-root" in command
        or "matrix-authority" in " ".join(str(value).lower() for value in command)
        or type(science) is not dict
        or science.get("autodl_stages")
        != [
            "RHS_CHEMISTRY",
            "FROZEN_GINE_STRICT_FLIP",
            "CALIBRATION_ONLY_SELECTION",
            "POST_FREEZE_HELDOUT_TEST",
            "STANDARDIZED_EXPORT",
            "INDEPENDENT_TERMINAL_VERIFY",
        ]
        or science.get("test_used_for_selection") is not False
        or science.get("zero_rule_terminal_requires_complete_search_proof") is not True
        or publisher.get("status") != "PREDEPLOYED_WAITING_T13_PASS"
        or publisher.get("cell_id") != f"{DATASET}/{METHOD}"
        or publisher.get("claim_enabled") is not True
        or publisher.get("unique_claim_required") is not True
        or publisher.get("hpc_may_write_locator") is not False
        or publisher.get("hpc_may_write_matrix") is not False
        or publisher.get("matrix_writer")
        != "EXISTING_FAST16_MATRIX_PUBLISHER_QUEUE_ONLY"
        or publisher.get("publisher_id") != t13_spec.get("publisher_id")
        or publisher.get("terminal_root_locator") != t13_spec.get("terminal_locator")
        or publisher.get("expected_terminal_root") != t13_spec.get("output_root")
    ):
        raise T8HPCT13SpecError("successor task chain contract changed")
    if check_files:
        repository = _absolute(import_spec["repo_root"], label="repo root", exists=True)
        if _clean_commit(repository) != manifest.get("execution_commit"):
            raise T8HPCT13SpecError("execution checkout changed after spec sealing")
        for path in (
            repository / "scripts/autodl/run_t8_hpc_import_owner_v1.py",
            repository / "scripts/autodl/run_t13_from_hpc_import_v1.py",
            repository / "scripts/autodl/run_t13_from_hpc_owner_v1.py",
        ):
            if path.is_symlink() or not path.is_file():
                raise T8HPCT13SpecError(f"successor entrypoint is absent: {path}")
    return {"manifest": manifest, **specs}


def write_t13_release(
    *, spec_root: str | Path, import_root: str | Path, output: str | Path
) -> dict[str, Any]:
    specs = validate_spec_set(spec_root, check_files=True)
    imported = _absolute(import_root, label="import root", exists=True)
    import_manifest = read_json(imported / "import_manifest.json", label="import manifest")
    from src.baselines.globalgce_hpc_autodl_import import (
        validate_imported_hpc_result,
    )

    validate_imported_hpc_result(imported)
    if str(imported) != specs["t13"].get("required_import_root"):
        raise T8HPCT13SpecError("T13 release names another import root")
    release = _self_hashed(
        {
            "schema_version": RELEASE_SCHEMA,
            "status": "READY_WAITING_T13_GPU",
            "released_at": _utc_now(),
            "import_task_id": specs["import"]["task_id"],
            "t13_task_id": specs["t13"]["task_id"],
            "import_root": str(imported),
            "import_manifest_sha256": import_manifest["import_manifest_sha256"],
            "t13_task_spec_sha256": specs["t13"]["task_spec_sha256"],
            "gpu_index": specs["t13"]["gpu_index"],
            "gpu_uuid": specs["t13"]["gpu_uuid"],
            "output_root": specs["t13"]["output_root"],
            "publisher_id": specs["publisher"]["publisher_id"],
            "matrix_write_enabled": False,
        },
        "release_sha256",
    )
    destination = _absolute(output, label="release output")
    if destination.exists() or destination.is_symlink():
        existing = read_json(destination, label="existing T13 release")
        if existing != release:
            raise T8HPCT13SpecError("existing T13 release conflicts")
        return existing
    atomic_json(destination, release)
    return release


def validate_t13_release(
    *, spec_root: str | Path, release_path: str | Path
) -> dict[str, Any]:
    """Reopen the import-to-T13 release without treating it as cell readiness."""

    specs = validate_spec_set(spec_root, check_files=True)
    release = read_json(release_path, label="T13 release")
    require_self_hash(release, "release_sha256", "T13 release")
    imported = _absolute(release.get("import_root"), label="released import", exists=True)
    from src.baselines.globalgce_hpc_autodl_import import (
        validate_imported_hpc_result,
    )

    manifest = validate_imported_hpc_result(imported)
    expected = specs["t13"]
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("status") != "READY_WAITING_T13_GPU"
        or release.get("import_task_id") != specs["import"].get("task_id")
        or release.get("t13_task_id") != expected.get("task_id")
        or str(imported) != expected.get("required_import_root")
        or release.get("import_manifest_sha256")
        != manifest.get("import_manifest_sha256")
        or release.get("t13_task_spec_sha256") != expected.get("task_spec_sha256")
        or release.get("gpu_index") != expected.get("gpu_index")
        or release.get("gpu_uuid") != expected.get("gpu_uuid")
        or release.get("output_root") != expected.get("output_root")
        or release.get("publisher_id") != specs["publisher"].get("publisher_id")
        or release.get("matrix_write_enabled") is not False
    ):
        raise T8HPCT13SpecError("T13 release binding changed")
    return release


def publish_verified_t13_locator(
    *, spec_root: str | Path, terminal_root: str | Path
) -> dict[str, Any]:
    """Verify AutoDL T13 and expose its root to the sole matrix publisher.

    This function does not write the matrix.  It only writes the standard
    no-replace locator named by the unique publisher spec.
    """

    specs = validate_spec_set(spec_root, check_files=True)
    root = _absolute(terminal_root, label="T13 terminal root", exists=True)
    if str(root) != specs["publisher"].get("expected_terminal_root"):
        raise T8HPCT13SpecError("T13 terminal differs from publisher claim")
    from src.baselines.tastemolnet_globalgce_full import verify_t13_output

    lease_path = _absolute(
        specs["publisher"]["publisher_lease_path"], label="publisher lease"
    )
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise T8HPCT13SpecError(
                "canonical TasteMolNet/GlobalGCE publisher lease is busy"
            ) from exc
        audit = verify_t13_output(root)
        run_manifest = read_json(root / "run_manifest.json", label="T13 run manifest")
        if (
            audit.get("status") != "PASS"
            or audit.get("passed") is not True
            or run_manifest.get("status") != "PASS"
            or run_manifest.get("run_complete") is not True
            or run_manifest.get("upstream_kind") != "hpc_exact_gspan_import"
            or run_manifest.get("hpc_mining_only") is not True
            or run_manifest.get("rhs_chemistry_run_on_autodl") is not True
            or run_manifest.get("gine_inference_run_on_autodl") is not True
            or run_manifest.get("calibration_test_run_on_autodl") is not True
            or run_manifest.get("test_used_for_selection") is not False
        ):
            raise T8HPCT13SpecError("AutoDL T13 terminal is not publishable")
        locator = {
            "schema_version": LOCATOR_SCHEMA,
            "status": "READY",
            "dataset": DATASET,
            "method": METHOD,
            "terminal_root": str(root),
        }
        atomic_json_no_replace(specs["publisher"]["terminal_root_locator"], locator)
        return locator
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "IMPORT_SPEC_SCHEMA",
    "LOCATOR_SCHEMA",
    "PUBLISHER_SPEC_SCHEMA",
    "RELEASE_SCHEMA",
    "SPEC_SET_SCHEMA",
    "T13_SPEC_SCHEMA",
    "T8HPCT13SpecError",
    "atomic_json",
    "atomic_json_no_replace",
    "build_spec_set",
    "canonical_sha256",
    "publish_verified_t13_locator",
    "read_json",
    "require_self_hash",
    "validate_spec_set",
    "validate_t13_release",
    "write_t13_release",
]
