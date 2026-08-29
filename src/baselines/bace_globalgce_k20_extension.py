"""Fail-closed BACE GlobalGCE K20 raw-rule extension.

The paper selector remains exactly K=20.  This route only enlarges the fresh,
train-only official raw-rule pool.  It owns physical GPU2 for every science
child's complete lifetime and has no process-signal authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.baselines.bace_gnn_baseline_contracts import (
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_adapter import (
    BACE_GLOBALGCE_RULE_POOL_SHORTFALL,
    audit_bace_globalgce_train_contract,
    build_bace_frozen_gine_rule_pool,
    validate_bace_globalgce_terminal_artifacts,
)
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    validate_official_globalgce_root,
)
from src.baselines.globalgce_mutagenicity_adapter import PoolBuildConfig
from src.baselines.globalgce_resumable import validate_exact_top_k_proof_identity
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    sha256_file,
    stable_sha256,
    utc_now,
)
from src.utils.process_identity_v2 import (
    ProcessSnapshotV2,
    capture_process_snapshot,
    require_auto_termination_disabled,
)
from src.utils.tastemolnet_gine_pass_adoption_v1 import _git_identity
from src.oracles.gnn_oracle import REQUIRED_CHECKPOINT_FILES, verify_checkpoint_bundle


SCHEMA_VERSION = "bace_globalgce_k20_extension_v1"
CONTROLLER_SCHEMA = "bace_globalgce_k20_controller_v1"
ROUND_SCHEMA = "bace_globalgce_k20_round_v1"
RAW_ROUND_SCHEMA = "bace_globalgce_k20_raw_round_v1"
FINAL_K = 20
PHYSICAL_GPU_INDEX = 2
VISIBLE_DEVICE = "cuda:0"
EXPECTED_SOURCE_PARENT_COUNT = 360
EXPECTED_EPOCHS = 100
EXPECTED_MIN_FREQ = 7
HEARTBEAT_INTERVAL_SECONDS = 60
EXPECTED_SHORTFALL = BACE_GLOBALGCE_RULE_POOL_SHORTFALL
RELEASE_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
RUNTIME_ROOT = Path("/autodl-fs/data/counterfactual-subgraph-runtime")
OUTPUT_NAMESPACE = RUNTIME_ROOT / "outputs" / "bace_globalgce_k20"
GPU_LOCK_PATH = RUNTIME_ROOT / "locks" / "gpu-2.lock"
RAW_SHORTFALL_RECEIPT = "K20_RAW_ROUND.json"
RAW_SHORTFALL_EXIT_CODE = 20
RAW_SHORTFALL_MARKER = "[BACE_GLOBALGCE_K20_RAW_SHORTFALL]"
PASS_MARKER = "[BACE_GLOBALGCE_K20_PASS]"
LAUNCHED_MARKER = "[BACE_GLOBALGCE_K20_EXTENSION_LAUNCHED]"
BASE_POOL_PASS_MARKER = "[BACE_GLOBALGCE_FROZEN_GINE_RULE_POOL_PASS]"


class BACEGlobalGCEK20Error(RuntimeError):
    """The extension could not preserve its fixed scientific contract."""


@dataclass(frozen=True, slots=True)
class RoundSpec:
    round_index: int
    cumulative_seeds: tuple[int, ...]
    cumulative_raw_budget: int
    incremental_seed: int
    incremental_raw_budget: int


ROUND_PLAN: tuple[RoundSpec, ...] = (
    RoundSpec(1, (7,), 80, 7, 80),
    RoundSpec(2, (7, 17), 200, 17, 120),
    RoundSpec(3, (7, 17, 27), 500, 27, 300),
)


def validate_round_plan(plan: Sequence[RoundSpec] = ROUND_PLAN) -> None:
    expected = (
        (1, (7,), 80, 7, 80),
        (2, (7, 17), 200, 17, 120),
        (3, (7, 17, 27), 500, 27, 300),
    )
    observed = tuple(
        (
            item.round_index,
            item.cumulative_seeds,
            item.cumulative_raw_budget,
            item.incremental_seed,
            item.incremental_raw_budget,
        )
        for item in plan
    )
    if observed != expected or sum(item.incremental_raw_budget for item in plan) != 500:
        raise BACEGlobalGCEK20Error("K20 raw-rule extension schedule changed")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BACEGlobalGCEK20Error(f"{label} is unavailable or malformed") from exc
    if type(value) is not dict:
        raise BACEGlobalGCEK20Error(f"{label} is not one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if type(value) is not dict:
                    raise BACEGlobalGCEK20Error(
                        f"{label}:{line_number} is not one JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise BACEGlobalGCEK20Error(f"{label} is unavailable or malformed") from exc
    return rows


def _snapshot_matches(snapshot: ProcessSnapshotV2) -> None:
    observed = capture_process_snapshot(snapshot.pid)
    if not snapshot.same_runtime_identity(observed):
        raise BACEGlobalGCEK20Error(
            f"protected process identity changed: pid={snapshot.pid}"
        )


def _parse_process(value: str, *, gpu_index: int) -> ProcessSnapshotV2:
    pid_text, separator, ticks_text = value.partition(":")
    if not separator or not pid_text.isdigit() or not ticks_text.isdigit():
        raise BACEGlobalGCEK20Error(
            f"protected GPU{gpu_index} process must have PID:START_TICKS form"
        )
    pid, ticks = int(pid_text), int(ticks_text)
    if pid <= 1 or ticks <= 0:
        raise BACEGlobalGCEK20Error("protected process identity is invalid")
    snapshot = capture_process_snapshot(pid)
    if snapshot.pid_start_ticks != ticks:
        raise BACEGlobalGCEK20Error(
            f"protected GPU{gpu_index} process start ticks changed: pid={pid}"
        )
    return snapshot


def parse_protected_processes(
    *, gpu0: str, gpu3: str
) -> dict[int, ProcessSnapshotV2]:
    result = {
        0: _parse_process(gpu0, gpu_index=0),
        3: _parse_process(gpu3, gpu_index=3),
    }
    if result[0].pid == result[3].pid:
        raise BACEGlobalGCEK20Error("GPU0 and GPU3 protected PIDs must differ")
    return result


def _gpu_inventory() -> dict[int, dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BACEGlobalGCEK20Error("nvidia-smi GPU inventory is unavailable") from exc
    result: dict[int, dict[str, Any]] = {}
    for raw in output.splitlines():
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 3:
            raise BACEGlobalGCEK20Error("nvidia-smi GPU inventory is malformed")
        result[int(fields[0])] = {
            "uuid": fields[1],
            "memory_used_mib": int(fields[2]),
        }
    return result


def _gpu_compute_processes() -> dict[str, set[int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BACEGlobalGCEK20Error("nvidia-smi process inventory is unavailable") from exc
    result: dict[str, set[int]] = {}
    for raw in output.splitlines():
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise BACEGlobalGCEK20Error("nvidia-smi process inventory is malformed")
        result.setdefault(fields[0], set()).add(int(fields[1]))
    return result


def _command_option(command: Sequence[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _protected_command_has_exact_role(
    snapshot: ProcessSnapshotV2, *, gpu_index: int
) -> bool:
    command = list(snapshot.command)
    scripts = [Path(value).name for value in command if value.endswith(".py")]
    if gpu_index == 0:
        return (
            "run_bace_vrrw.py" in scripts
            and "--dataset-dir" in command
            and "--gnn-checkpoint" in command
            and any("gcfexplainer" in value.lower() for value in command)
        )
    if gpu_index == 3:
        return (
            "run_generation.py" in scripts
            and _command_option(command, "--route") == "project"
            and _command_option(command, "--dataset") == "bace"
            and _command_option(command, "--mode") == "full"
            and any("comrecgc" in value.lower() for value in command)
        )
    return False


def _verify_protected_gpu_roles(
    protected: Mapping[int, ProcessSnapshotV2],
    *,
    gpu_inventory: Mapping[int, Mapping[str, Any]],
) -> None:
    current_inventory = _gpu_inventory()
    compute = _gpu_compute_processes()
    for gpu_index in (0, 3):
        snapshot = protected.get(gpu_index)
        expected_gpu = gpu_inventory.get(gpu_index)
        current_gpu = current_inventory.get(gpu_index)
        if snapshot is None or expected_gpu is None or current_gpu is None:
            raise BACEGlobalGCEK20Error("protected GPU role inventory is incomplete")
        _snapshot_matches(snapshot)
        uuid = str(expected_gpu.get("uuid") or "")
        if not uuid or current_gpu.get("uuid") != uuid:
            raise BACEGlobalGCEK20Error(
                f"physical GPU{gpu_index} UUID changed during extension"
            )
        if snapshot.pid not in compute.get(uuid, set()):
            raise BACEGlobalGCEK20Error(
                f"protected PID is not running on physical GPU{gpu_index}"
            )
        if not _protected_command_has_exact_role(snapshot, gpu_index=gpu_index):
            raise BACEGlobalGCEK20Error(
                f"protected GPU{gpu_index} process command has the wrong task role"
            )


def _install_deferred_signal_mask() -> set[signal.Signals]:
    """Block controller stop signals before any process-local thread exists."""

    task_root = Path("/proc/self/task")
    if task_root.is_dir():
        task_ids = tuple(entry.name for entry in task_root.iterdir() if entry.name.isdigit())
        if task_ids != (str(os.getpid()),):
            raise BACEGlobalGCEK20Error(
                "K20 controller must install its signal mask as one OS thread"
            )
    elif threading.active_count() != 1:
        raise BACEGlobalGCEK20Error(
            "K20 controller must install its signal mask before Python threads"
        )
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, RELEASE_SIGNALS)
    if previous.intersection(RELEASE_SIGNALS):
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        raise BACEGlobalGCEK20Error(
            "K20 controller inherited an ambiguous blocked stop signal"
        )
    return set(previous)


def _require_deferred_signal_mask() -> None:
    current = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    if not RELEASE_SIGNALS.issubset(current):
        raise BACEGlobalGCEK20Error("K20 controller signal mask changed")


def _drain_deferred_signals(stop_requested: threading.Event) -> None:
    """Synchronously convert blocked signals into one durable stop request."""

    _require_deferred_signal_mask()
    while pending := signal.sigpending().intersection(RELEASE_SIGNALS):
        signal.sigwait(pending)
        stop_requested.set()


def _restore_signal_mask(previous: set[signal.Signals]) -> None:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def unblock_deferred_signals_for_science_child() -> None:
    """Restore normal signal delivery in a raw-round process after exec."""

    signal.pthread_sigmask(signal.SIG_UNBLOCK, RELEASE_SIGNALS)


@dataclass(slots=True)
class HeldGpuLease:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int]

    @classmethod
    def acquire(cls, path: Path) -> "HeldGpuLease":
        canonical = GPU_LOCK_PATH.resolve(strict=False)
        if path.resolve(strict=False) != canonical:
            raise BACEGlobalGCEK20Error("GPU2 lock path is not globally canonical")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = os.fstat(descriptor)
            named = os.stat(path, follow_symlinks=False)
            identity = (
                int(held.st_dev),
                int(held.st_ino),
                int(held.st_uid),
                int(held.st_nlink),
            )
            named_identity = (
                int(named.st_dev),
                int(named.st_ino),
                int(named.st_uid),
                int(named.st_nlink),
            )
            if identity != named_identity or held.st_nlink != 1:
                raise BACEGlobalGCEK20Error("GPU2 lock authority is aliased")
            return cls(path, descriptor, identity)
        except BaseException:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        held = os.fstat(self.descriptor)
        named = os.stat(self.path, follow_symlinks=False)
        current = (
            int(held.st_dev),
            int(held.st_ino),
            int(held.st_uid),
            int(held.st_nlink),
        )
        named_identity = (
            int(named.st_dev),
            int(named.st_ino),
            int(named.st_uid),
            int(named.st_nlink),
        )
        if current != self.identity or named_identity != self.identity:
            raise BACEGlobalGCEK20Error("GPU2 lock authority changed")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldGpuLease":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def build_round_command(
    *,
    python: Path,
    project_root: Path,
    config: Path,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gnn_checkpoint: Path,
    output_root: Path,
    spec: RoundSpec,
) -> list[str]:
    validate_round_plan()
    return [
        str(python),
        "-I",
        "-B",
        str(project_root / "scripts/autodl/run_bace_globalgce_k20_extension.py"),
        "--config",
        str(config),
        "--set",
        "inference.fallback_to_heuristic=false",
        "raw-round",
        "--gnn-checkpoint",
        str(gnn_checkpoint),
        "--source-manifest",
        str(source_manifest),
        "--native-train-csv",
        str(native_train_csv),
        "--official-root",
        str(official_root),
        "--output-dir",
        str(output_root),
        "--expected-parent-count",
        str(EXPECTED_SOURCE_PARENT_COUNT),
        "--seed",
        str(spec.incremental_seed),
        "--min-freq",
        str(EXPECTED_MIN_FREQ),
        "--epochs",
        str(EXPECTED_EPOCHS),
        "--top-k-native",
        str(spec.incremental_raw_budget),
        "--device",
        VISIBLE_DEVICE,
        "--no-resume",
        "--gspan-exact-top-k-pruning",
    ]


def validate_catalog_row(
    row: Mapping[str, Any], *, expected_checkpoint_hash: str
) -> tuple[dict[str, Any], str, str]:
    payload = dict(row)
    if (
        payload.get("action_kind") != "lhs_rhs_graph_transformation_rule"
        or payload.get("action_semantics")
        != "native_lhs_to_rhs_attachment_aware_v1"
        or payload.get("oracle_backend") != "gnn"
        or payload.get("classifier_family") != "gine"
        or payload.get("rf_oracle_used") is not False
        or payload.get("oracle_checkpoint_hash") != expected_checkpoint_hash
        or payload.get("source_split") != "train"
    ):
        raise BACEGlobalGCEK20Error("raw native rule provenance changed")
    rule = GlobalGCENativeRule.from_payload(payload)
    content_hash = rule.content_hash()
    if payload.get("candidate_id") != rule.rule_id:
        raise BACEGlobalGCEK20Error("raw native rule candidate ID changed")
    if payload.get("rule_content_hash") != content_hash:
        raise BACEGlobalGCEK20Error("raw native rule content hash changed")
    if payload.get("selector_chemistry") != rule.selector_chemistry():
        raise BACEGlobalGCEK20Error("raw native rule selector chemistry changed")
    semantic_payload = rule.to_payload()
    semantic_payload.pop("rule_id", None)
    semantic_payload.pop("native_rule_index", None)
    return payload, content_hash, stable_sha256(semantic_payload)


def merge_unique_rules(
    catalogs: Iterable[tuple[RoundSpec, Sequence[Mapping[str, Any]]]],
    *,
    expected_checkpoint_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    by_semantic_content: dict[str, str] = {}
    by_candidate: dict[str, str] = {}
    for spec, rows in catalogs:
        for raw_position, row in enumerate(rows, start=1):
            validated, content_hash, semantic_hash = validate_catalog_row(
                row, expected_checkpoint_hash=expected_checkpoint_hash
            )
            candidate_id = str(validated["candidate_id"])
            if candidate_id in by_candidate and by_candidate[candidate_id] != content_hash:
                raise BACEGlobalGCEK20Error(
                    "one candidate ID maps to multiple native rule contents"
                )
            by_candidate[candidate_id] = content_hash
            duplicate_of = by_semantic_content.get(semantic_hash)
            accepted = duplicate_of is None
            if accepted:
                by_semantic_content[semantic_hash] = candidate_id
                selected = dict(validated)
                selected["semantic_rule_content_hash"] = semantic_hash
                selected["extension_provenance"] = {
                    "round": spec.round_index,
                    "seed": spec.incremental_seed,
                    "raw_budget": spec.incremental_raw_budget,
                    "raw_position": raw_position,
                }
                unique.append(selected)
            audit.append(
                {
                    "candidate_id": candidate_id,
                    "rule_content_hash": content_hash,
                    "semantic_rule_content_hash": semantic_hash,
                    "round": spec.round_index,
                    "seed": spec.incremental_seed,
                    "raw_budget": spec.incremental_raw_budget,
                    "raw_position": raw_position,
                    "hard_native_validation_pass": True,
                    "accepted_unique": accepted,
                    "duplicate_of_candidate_id": duplicate_of,
                    "source_split": "train",
                }
            )
    return unique, audit


def _capture_science_contract(
    *,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gnn_checkpoint: Path,
) -> tuple[dict[str, Any], Path]:
    train_contract = audit_bace_globalgce_train_contract(
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
    )
    official_audit = validate_official_globalgce_root(official_root)
    checkpoint, card, _schema = validate_bace_frozen_gine(gnn_checkpoint)
    bundle_audit = verify_checkpoint_bundle(
        checkpoint, verify_hashes=True, require_taste_closure=False
    )
    provenance = oracle_provenance(card, checkpoint)
    if any((checkpoint / name).is_symlink() for name in REQUIRED_CHECKPOINT_FILES):
        raise BACEGlobalGCEK20Error("frozen BACE GINE bundle contains a symlink")
    checkpoint_files = {
        name: file_identity(checkpoint / name) for name in REQUIRED_CHECKPOINT_FILES
    }
    reopened_bundle_audit = verify_checkpoint_bundle(
        checkpoint, verify_hashes=True, require_taste_closure=False
    )
    reopened_checkpoint_files = {
        name: file_identity(checkpoint / name) for name in REQUIRED_CHECKPOINT_FILES
    }
    if (
        bundle_audit.get("hashes_verified", 0) < len(REQUIRED_CHECKPOINT_FILES) - 1
        or bundle_audit.get("model_card") != card
        or card.get("checkpoint_id") != checkpoint_files["model.pt"]["sha256"]
        or reopened_bundle_audit != bundle_audit
        or reopened_checkpoint_files != checkpoint_files
    ):
        raise BACEGlobalGCEK20Error("frozen BACE GINE bundle identity changed")
    contract = {
        "train_contract": train_contract.audit,
        "source_parent_ids": [
            parent.parent_id for parent in train_contract.source_parents
        ],
        "official_source_audit": official_audit,
        "oracle": provenance,
        "checkpoint_model": file_identity(checkpoint / "model.pt"),
        "checkpoint_bundle": {
            "required_files": list(REQUIRED_CHECKPOINT_FILES),
            "files": checkpoint_files,
            "hashes_verified": bundle_audit["hashes_verified"],
        },
    }
    if (
        len(contract["source_parent_ids"]) != EXPECTED_SOURCE_PARENT_COUNT
        or contract["train_contract"].get("calibration_loaded") is not False
        or contract["train_contract"].get("test_loaded") is not False
    ):
        raise BACEGlobalGCEK20Error("BACE train-only science contract changed")
    return contract, checkpoint


def _capture_execution_contract(
    *,
    project_root: Path,
    python: Path,
    config: Path,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gnn_checkpoint: Path,
) -> tuple[dict[str, Any], Path]:
    science, checkpoint = _capture_science_contract(
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
        official_root=official_root,
        gnn_checkpoint=gnn_checkpoint,
    )
    git = _git_identity(
        project_root,
        critical_paths=(
            "src/baselines/bace_globalgce_k20_extension.py",
            "src/baselines/globalgce_bace_adapter.py",
            "src/baselines/globalgce_bace_native_rules.py",
            "src/baselines/globalgce_mutagenicity_adapter.py",
            "src/baselines/globalgce_resumable.py",
            "scripts/autodl/run_bace_globalgce_k20_extension.py",
        ),
    )
    return (
        {
            "git": git,
            "python": file_identity(python),
            "config": file_identity(config),
            "science": science,
        },
        checkpoint,
    )


def _require_science_contract_unchanged(
    expected: Mapping[str, Any],
    *,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gnn_checkpoint: Path,
) -> Path:
    observed, checkpoint = _capture_science_contract(
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
        official_root=official_root,
        gnn_checkpoint=gnn_checkpoint,
    )
    if observed != dict(expected):
        raise BACEGlobalGCEK20Error("BACE K20 science inputs changed")
    return checkpoint


def _require_execution_contract_unchanged(
    expected: Mapping[str, Any],
    *,
    project_root: Path,
    python: Path,
    config: Path,
    source_manifest: Path,
    native_train_csv: Path,
    official_root: Path,
    gnn_checkpoint: Path,
) -> Path:
    observed, checkpoint = _capture_execution_contract(
        project_root=project_root,
        python=python,
        config=config,
        source_manifest=source_manifest,
        native_train_csv=native_train_csv,
        official_root=official_root,
        gnn_checkpoint=gnn_checkpoint,
    )
    if observed != dict(expected):
        raise BACEGlobalGCEK20Error("BACE K20 execution/input contract changed")
    return checkpoint


def _raw_config(*, seed: int, raw_budget: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "epochs": EXPECTED_EPOCHS,
        "top_k_native": raw_budget,
        "device": VISIBLE_DEVICE,
        "min_freq": EXPECTED_MIN_FREQ,
        "gspan_exact_top_k_pruning": True,
        "gspan_adoption_identity": None,
    }


def _validate_raw_manifest_binding(
    root: Path,
    *,
    science_contract: Mapping[str, Any],
    seed: int,
    raw_budget: int,
    complete: bool,
) -> dict[str, Any]:
    manifest = _load_json(root / "run_manifest.json", label="raw round manifest")
    train_contract = science_contract.get("train_contract")
    if not isinstance(train_contract, dict):
        raise BACEGlobalGCEK20Error("raw round expected train contract is malformed")
    expected_status = "PASS" if complete else "RUNNING"
    if (
        manifest.get("schema_version")
        != "bace_globalgce_frozen_gine_rule_pool_v1"
        or manifest.get("dataset") != "bace"
        or manifest.get("method_id") != "globalgce"
        or manifest.get("status") != expected_status
        or manifest.get("run_complete") is not complete
        or manifest.get("calibration_loaded") is not False
        or manifest.get("test_loaded") is not False
        or manifest.get("oracle_backend") != "gnn"
        or manifest.get("classifier_family") != "gine"
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("source_manifest") != train_contract.get("source_manifest")
        or manifest.get("native_train_csv") != train_contract.get("native_train_csv")
        or manifest.get("native_train_contract") != train_contract
        or manifest.get("official_source_audit")
        != science_contract.get("official_source_audit")
        or manifest.get("oracle") != science_contract.get("oracle")
        or manifest.get("source_parent_ids")
        != science_contract.get("source_parent_ids")
    ):
        raise BACEGlobalGCEK20Error("raw round manifest/input closure failed")
    config = manifest.get("config")
    expected_config = _raw_config(seed=seed, raw_budget=raw_budget)
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected_config.items()
    ):
        raise BACEGlobalGCEK20Error("raw round configuration binding changed")
    return manifest


def _validate_raw_catalog(
    root: Path,
    *,
    science_contract: Mapping[str, Any],
    raw_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_path = root / "native" / "native_rule_catalog.jsonl"
    rows = _load_jsonl(catalog_path, label="raw native rule catalog")
    oracle = science_contract.get("oracle")
    checkpoint_hash = (
        str(oracle.get("oracle_checkpoint_hash") or "")
        if isinstance(oracle, dict)
        else ""
    )
    if len(rows) > raw_budget or not checkpoint_hash:
        raise BACEGlobalGCEK20Error("raw native rule catalog exceeds its contract")
    for row in rows:
        validate_catalog_row(row, expected_checkpoint_hash=checkpoint_hash)
    training = _load_json(root / "training_summary.json", label="raw training summary")
    proof = validate_exact_top_k_proof_identity(
        training.get("gspan_exact_top_k_proof") or {}
    )
    if (
        training.get("classifier_parameters_frozen") is not True
        or training.get("oracle_backend") != "gnn"
        or training.get("classifier_family") != "gine"
        or training.get("rf_oracle_used") is not False
        or training.get("generation_input_split") != "train"
        or training.get("calibration_loaded") is not False
        or training.get("test_loaded") is not False
        or training.get("gspan_exact_top_k_pruning") is not True
        or training.get("gspan_mining_route") != "fresh_exact_top_k"
        or training.get("valid_native_rule_count") != len(rows)
        or training.get("native_rule_catalog_sha256") != sha256_file(catalog_path)
    ):
        raise BACEGlobalGCEK20Error("raw training/catalog evidence changed")
    return rows, proof


def validate_successful_raw_round(
    root: Path,
    *,
    science_contract: Mapping[str, Any],
    seed: int,
    raw_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    terminal = validate_bace_globalgce_terminal_artifacts(
        root, require_exact_top_k=True
    )
    manifest = _validate_raw_manifest_binding(
        root,
        science_contract=science_contract,
        seed=seed,
        raw_budget=raw_budget,
        complete=True,
    )
    rows, proof = _validate_raw_catalog(
        root, science_contract=science_contract, raw_budget=raw_budget
    )
    if (
        len(rows) < FINAL_K
        or int(manifest.get("candidate_count") or -1) != len(rows)
        or terminal.get("gspan_exact_top_k_proof") != proof
        or (root / "PASS").read_text(encoding="utf-8")
        != BASE_POOL_PASS_MARKER + "\n"
        or (root / RAW_SHORTFALL_RECEIPT).exists()
        or (root / "RAW_SHORTFALL").exists()
    ):
        raise BACEGlobalGCEK20Error("successful raw round terminal closure failed")
    return rows, terminal


def validate_shortfall_raw_round(
    root: Path,
    *,
    science_contract: Mapping[str, Any],
    seed: int,
    raw_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (root / "PASS").exists():
        raise BACEGlobalGCEK20Error("shortfall raw round also claims PASS")
    _validate_raw_manifest_binding(
        root,
        science_contract=science_contract,
        seed=seed,
        raw_budget=raw_budget,
        complete=False,
    )
    rows, proof = _validate_raw_catalog(
        root, science_contract=science_contract, raw_budget=raw_budget
    )
    if len(rows) >= FINAL_K:
        raise BACEGlobalGCEK20Error("shortfall receipt has twenty valid rules")
    receipt = _load_json(root / RAW_SHORTFALL_RECEIPT, label="raw shortfall receipt")
    payload = dict(receipt)
    payload_hash = payload.pop("receipt_payload_sha256", None)
    expected_artifacts = {
        "run_manifest": file_identity(root / "run_manifest.json"),
        "training_summary": file_identity(root / "training_summary.json"),
        "native_rule_catalog": file_identity(
            root / "native" / "native_rule_catalog.jsonl"
        ),
    }
    if (
        receipt.get("schema_version") != RAW_ROUND_SCHEMA
        or receipt.get("status") != "EXPECTED_SHORTFALL"
        or receipt.get("reason") != EXPECTED_SHORTFALL
        or receipt.get("exit_code") != RAW_SHORTFALL_EXIT_CODE
        or receipt.get("seed") != seed
        or receipt.get("raw_budget") != raw_budget
        or receipt.get("valid_native_rule_count") != len(rows)
        or receipt.get("science_contract") != dict(science_contract)
        or receipt.get("artifacts") != expected_artifacts
        or receipt.get("gspan_exact_top_k_proof") != proof
        or payload_hash != stable_sha256(payload)
        or (root / "RAW_SHORTFALL").read_text(encoding="utf-8")
        != RAW_SHORTFALL_MARKER + "\n"
    ):
        raise BACEGlobalGCEK20Error("raw shortfall receipt/hash closure failed")
    return rows, receipt


def run_raw_round(
    *,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    expected_parent_count: int,
    seed: int,
    min_freq: int,
    epochs: int,
    top_k_native: int,
    device: str,
    resume: bool,
    gspan_exact_top_k_pruning: bool,
) -> dict[str, Any]:
    """Run one fresh official round and seal only an exact expected shortfall."""

    allowed_rounds = {
        (spec.incremental_seed, spec.incremental_raw_budget) for spec in ROUND_PLAN
    }
    if (
        expected_parent_count != EXPECTED_SOURCE_PARENT_COUNT
        or min_freq != EXPECTED_MIN_FREQ
        or epochs != EXPECTED_EPOCHS
        or device != VISIBLE_DEVICE
        or resume is not False
        or gspan_exact_top_k_pruning is not True
        or (seed, top_k_native) not in allowed_rounds
    ):
        raise BACEGlobalGCEK20Error("raw round configuration changed")
    source = Path(source_manifest).resolve(strict=True)
    native = Path(native_train_csv).resolve(strict=True)
    official = Path(official_root).resolve(strict=True)
    checkpoint_input = Path(gnn_checkpoint).resolve(strict=True)
    root = Path(output_dir).resolve(strict=False)
    if OUTPUT_NAMESPACE.resolve(strict=False) not in root.parents or root.exists():
        raise BACEGlobalGCEK20Error("raw round output is not a fresh K20 child root")
    science, checkpoint = _capture_science_contract(
        source_manifest=source,
        native_train_csv=native,
        official_root=official,
        gnn_checkpoint=checkpoint_input,
    )
    shortfall: RuntimeError | None = None
    try:
        build_bace_frozen_gine_rule_pool(
            source_manifest=source,
            native_train_csv=native,
            official_root=official,
            gnn_checkpoint=checkpoint,
            output_dir=root,
            min_freq=EXPECTED_MIN_FREQ,
            config=PoolBuildConfig(
                expected_parent_count=EXPECTED_SOURCE_PARENT_COUNT,
                seed=seed,
                epochs=EXPECTED_EPOCHS,
                top_k_native=top_k_native,
                device=VISIBLE_DEVICE,
                resume=False,
                forbid_calibration_test=True,
                gspan_exact_top_k_pruning=True,
            ),
        )
    except RuntimeError as exc:
        if str(exc) != EXPECTED_SHORTFALL:
            raise
        shortfall = exc
    _require_science_contract_unchanged(
        science,
        source_manifest=source,
        native_train_csv=native,
        official_root=official,
        gnn_checkpoint=checkpoint_input,
    )
    if shortfall is None:
        rows, terminal = validate_successful_raw_round(
            root,
            science_contract=science,
            seed=seed,
            raw_budget=top_k_native,
        )
        return {
            "status": "TERMINAL_ARTIFACTS_VERIFIED",
            "seed": seed,
            "raw_budget": top_k_native,
            "valid_native_rule_count": len(rows),
            "terminal": terminal,
        }

    _validate_raw_manifest_binding(
        root,
        science_contract=science,
        seed=seed,
        raw_budget=top_k_native,
        complete=False,
    )
    rows, proof = _validate_raw_catalog(
        root, science_contract=science, raw_budget=top_k_native
    )
    if len(rows) >= FINAL_K:
        raise BACEGlobalGCEK20Error("expected shortfall has twenty valid rules")
    receipt = {
        "schema_version": RAW_ROUND_SCHEMA,
        "status": "EXPECTED_SHORTFALL",
        "reason": EXPECTED_SHORTFALL,
        "exit_code": RAW_SHORTFALL_EXIT_CODE,
        "seed": seed,
        "raw_budget": top_k_native,
        "valid_native_rule_count": len(rows),
        "science_contract": science,
        "gspan_exact_top_k_proof": proof,
        "artifacts": {
            "run_manifest": file_identity(root / "run_manifest.json"),
            "training_summary": file_identity(root / "training_summary.json"),
            "native_rule_catalog": file_identity(
                root / "native" / "native_rule_catalog.jsonl"
            ),
        },
        "completed_at": utc_now(),
    }
    receipt["receipt_payload_sha256"] = stable_sha256(receipt)
    atomic_json(root / RAW_SHORTFALL_RECEIPT, receipt)
    # This marker is the final filesystem write in an expected-shortfall root.
    atomic_marker(root / "RAW_SHORTFALL", RAW_SHORTFALL_MARKER)
    return receipt


def _write_heartbeat(
    root: Path,
    *,
    sequence: int,
    controller_id: str,
    controller: ProcessSnapshotV2,
    child: ProcessSnapshotV2 | None,
    state: str,
    round_spec: RoundSpec | None,
) -> None:
    heartbeat = {
        "schema_version": CONTROLLER_SCHEMA,
        "controller_id": controller_id,
        "sequence": sequence,
        "state": state,
        "controller_process": controller.to_dict(),
        "science_child_process": child.to_dict() if child is not None else None,
        "round": round_spec.round_index if round_spec else None,
        "cumulative_raw_budget": (
            round_spec.cumulative_raw_budget if round_spec else 0
        ),
        "observed_at": utc_now(),
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    atomic_json(root / "heartbeats" / f"{sequence:020d}.json", heartbeat)
    atomic_json(root / "state.json", heartbeat)


def _wait_naturally(
    child: subprocess.Popen[Any],
    *,
    root: Path,
    sequence: int,
    controller_id: str,
    controller: ProcessSnapshotV2,
    child_snapshot: ProcessSnapshotV2,
    spec: RoundSpec,
    protected: Mapping[int, ProcessSnapshotV2],
    gpu_inventory: Mapping[int, Mapping[str, Any]],
    lease: HeldGpuLease,
    stop_requested: threading.Event,
    heartbeat_interval_seconds: int,
) -> tuple[int, int]:
    pending_error: BaseException | None = None
    gpu2_uuid = str(gpu_inventory.get(PHYSICAL_GPU_INDEX, {}).get("uuid") or "")
    child_observed_on_gpu2 = False
    try:
        while child.poll() is None:
            _drain_deferred_signals(stop_requested)
            lease.verify()
            _snapshot_matches(controller)
            _verify_protected_gpu_roles(protected, gpu_inventory=gpu_inventory)
            gpu2_compute = _gpu_compute_processes().get(gpu2_uuid, set())
            foreign_gpu2 = gpu2_compute - {child_snapshot.pid}
            if foreign_gpu2:
                raise BACEGlobalGCEK20Error(
                    f"physical GPU2 gained foreign compute PIDs: {sorted(foreign_gpu2)}"
                )
            child_observed_on_gpu2 = child_observed_on_gpu2 or (
                child_snapshot.pid in gpu2_compute
            )
            _write_heartbeat(
                root,
                sequence=sequence,
                controller_id=controller_id,
                controller=controller,
                child=child_snapshot,
                state=(
                    "STOP_REQUESTED_WAITING_FOR_SCIENCE_CHILD"
                    if stop_requested.is_set()
                    else "SCIENCE_CHILD_RUNNING"
                ),
                round_spec=spec,
            )
            sequence += 1
            deadline = time.monotonic() + heartbeat_interval_seconds
            while child.poll() is None and time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                _drain_deferred_signals(stop_requested)
    except BaseException as exc:
        pending_error = exc
    if pending_error is not None:
        # The lock cannot be dropped while the child is alive.  There is no
        # termination API: even repeated interrupts are deferred naturally.
        while child.poll() is None:
            try:
                time.sleep(1.0)
            except BaseException:
                continue
        child.wait()
        raise pending_error
    post_exit_gpu2 = _gpu_compute_processes().get(gpu2_uuid, set())
    if post_exit_gpu2:
        raise BACEGlobalGCEK20Error(
            f"physical GPU2 is not empty after science exit: {sorted(post_exit_gpu2)}"
        )
    if not child_observed_on_gpu2:
        raise BACEGlobalGCEK20Error(
            "science child was never observed on physical GPU2"
        )
    return int(child.wait()), sequence


def _publish_release_candidate(
    *,
    root: Path,
    controller_id: str,
    controller: ProcessSnapshotV2,
    selected: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    round_receipts: Sequence[Mapping[str, Any]],
    physical_gpu_uuid: str,
    sequence: int,
    revalidate: Callable[[], None],
    stop_requested: threading.Event,
) -> dict[str, Any]:
    if len(selected) != FINAL_K:
        raise BACEGlobalGCEK20Error("release candidate is not exactly K20")
    checkpoint_hash = str(provenance.get("oracle_checkpoint_hash") or "")
    semantic_hashes: set[str] = set()
    candidate_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for expected_rank, raw in enumerate(selected, start=1):
        row, _content_hash, semantic_hash = validate_catalog_row(
            raw, expected_checkpoint_hash=checkpoint_hash
        )
        if (
            row.get("rank") != expected_rank
            or row.get("extension_rank") != expected_rank
            or row.get("semantic_rule_content_hash") != semantic_hash
            or str(row.get("candidate_id")) in candidate_ids
            or semantic_hash in semantic_hashes
        ):
            raise BACEGlobalGCEK20Error("release candidate rank/semantic uniqueness failed")
        candidate_ids.add(str(row["candidate_id"]))
        semantic_hashes.add(semantic_hash)
        normalized.append(row)

    atomic_jsonl(root / "candidate_universe.jsonl", normalized)
    atomic_jsonl(root / "candidate_filter_audit.jsonl", audit)
    atomic_json(root / "oracle_provenance.json", provenance)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_CANDIDATE",
        "run_complete": False,
        "dataset": "bace",
        "method": "GlobalGCE",
        "final_k": FINAL_K,
        "selected_candidate_count": len(normalized),
        "cumulative_valid_unique_rules": sum(
            1 for row in audit if row.get("accepted_unique") is True
        ),
        "round_count": len(round_receipts),
        "cumulative_raw_budget_consumed": round_receipts[-1][
            "cumulative_raw_budget"
        ],
        "seeds_consumed": round_receipts[-1]["cumulative_seeds"],
        "stop_rule": "first_round_with_at_least_20_unique_valid_native_rules",
        "hard_validation": "GlobalGCENativeRule.from_payload_plus_hash_recompute",
        "deduplication": "canonical_semantic_rule_content_excluding_id_and_index",
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": checkpoint_hash,
        "classifier_checkpoint_unchanged": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "sealed_at": utc_now(),
    }
    atomic_json(root / "summary.json", summary)
    artifact_names = (
        "candidate_universe.jsonl",
        "candidate_filter_audit.jsonl",
        "oracle_provenance.json",
        "summary.json",
    )
    manifest = {
        **summary,
        "controller_id": controller_id,
        "controller_process": controller.to_dict(),
        "controller_receipt": file_identity(root / "controller_receipt.json"),
        "rounds": [dict(row) for row in round_receipts],
        "artifacts": {name: file_identity(root / name) for name in artifact_names},
        "execution_contract": dict(execution_contract),
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "physical_gpu_uuid": physical_gpu_uuid,
        "auto_terminate_uncontrolled_children": False,
        "signal_authority": False,
    }
    manifest["manifest_payload_sha256"] = stable_sha256(manifest)
    atomic_json(root / "run_manifest.json", manifest)
    _write_heartbeat(
        root,
        sequence=sequence,
        controller_id=controller_id,
        controller=controller,
        child=None,
        state="RELEASE_CANDIDATE_SEALED",
        round_spec=ROUND_PLAN[len(round_receipts) - 1],
    )

    # First full live check occurs after candidate sealing and before a
    # verifier/gate can authorize the marker.
    revalidate()
    verification = {
        "schema_version": SCHEMA_VERSION,
        "status": "RELEASE",
        "decision": "RELEASE",
        "exact_k20": True,
        "semantic_unique": True,
        "hard_native_validation": True,
        "train_only": True,
        "classifier_checkpoint_unchanged": True,
        "execution_contract": dict(execution_contract),
        "run_manifest": file_identity(root / "run_manifest.json"),
        "candidate_universe": file_identity(root / "candidate_universe.jsonl"),
        "verified_at": utc_now(),
    }
    atomic_json(root / "verification.json", verification)
    release_gate = {
        "schema_version": SCHEMA_VERSION,
        "status": "RELEASE",
        "decision": "RELEASE",
        "final_marker": PASS_MARKER,
        "artifacts": {
            "run_manifest": file_identity(root / "run_manifest.json"),
            "summary": file_identity(root / "summary.json"),
            "candidate_universe": file_identity(root / "candidate_universe.jsonl"),
            "candidate_filter_audit": file_identity(
                root / "candidate_filter_audit.jsonl"
            ),
            "oracle_provenance": file_identity(root / "oracle_provenance.json"),
            "verification": file_identity(root / "verification.json"),
        },
        "released_at": utc_now(),
    }
    release_gate["gate_payload_sha256"] = stable_sha256(release_gate)
    atomic_json(root / "release_gate.json", release_gate)

    persisted_gate = _load_json(root / "release_gate.json", label="release gate")
    gate_payload = dict(persisted_gate)
    gate_hash = gate_payload.pop("gate_payload_sha256", None)
    if gate_hash != stable_sha256(gate_payload):
        raise BACEGlobalGCEK20Error("release gate payload hash changed")
    for name, identity in persisted_gate["artifacts"].items():
        if file_identity(Path(str(identity["path"]))) != identity:
            raise BACEGlobalGCEK20Error(f"release artifact changed: {name}")
    reopened = _load_jsonl(root / "candidate_universe.jsonl", label="sealed K20")
    if reopened != normalized or (root / "_RUN_COMPLETE.json").exists():
        raise BACEGlobalGCEK20Error("sealed K20 publication changed")
    persisted_manifest = _load_json(root / "run_manifest.json", label="run manifest")
    manifest_payload = dict(persisted_manifest)
    manifest_hash = manifest_payload.pop("manifest_payload_sha256", None)
    if (
        persisted_manifest.get("status") != "SEALED_CANDIDATE"
        or persisted_manifest.get("run_complete") is not False
        or manifest_hash != stable_sha256(manifest_payload)
    ):
        raise BACEGlobalGCEK20Error("sealed run manifest closure failed")

    # The mask was installed while the controller had one OS thread, so every
    # subsequently created thread inherits it.  Synchronously draining here is
    # the linearization point between a deferred stop and the PASS commit.
    _require_deferred_signal_mask()
    revalidate()
    _drain_deferred_signals(stop_requested)
    if stop_requested.is_set():
        raise BACEGlobalGCEK20Error(
            "deferred controller stop requested before PASS commit"
        )
    atomic_marker(root / "PASS", PASS_MARKER)
    return manifest


def run_extension(
    *,
    controller_id: str,
    project_root: str | Path,
    python: str | Path,
    config: str | Path,
    output_root: str | Path,
    source_manifest: str | Path,
    native_train_csv: str | Path,
    official_root: str | Path,
    gnn_checkpoint: str | Path,
    protected_gpu0_process: str,
    protected_gpu3_process: str,
    heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Own physical GPU2, run the bounded schedule, and publish exactly K20."""

    require_auto_termination_disabled()
    validate_round_plan()
    if not controller_id or heartbeat_interval_seconds != HEARTBEAT_INTERVAL_SECONDS:
        raise BACEGlobalGCEK20Error("production controller identity/heartbeat changed")
    project = Path(project_root).resolve(strict=True)
    executable = Path(python).resolve(strict=True)
    config_path = Path(config).resolve(strict=True)
    source = Path(source_manifest).resolve(strict=True)
    native = Path(native_train_csv).resolve(strict=True)
    official = Path(official_root).resolve(strict=True)
    checkpoint_input = Path(gnn_checkpoint).resolve(strict=True)
    destination = Path(output_root).resolve(strict=False)
    namespace = OUTPUT_NAMESPACE.resolve(strict=False)
    if config_path != project / "configs/hpc.yaml":
        raise BACEGlobalGCEK20Error("K20 requires this checkout's configs/hpc.yaml")
    if destination.parent != namespace or destination == namespace:
        raise BACEGlobalGCEK20Error(
            "K20 output must be a direct child of the fixed runtime namespace"
        )
    if RUNTIME_ROOT.resolve(strict=True) not in destination.parents:
        raise BACEGlobalGCEK20Error("K20 runtime root changed")

    protected = parse_protected_processes(
        gpu0=protected_gpu0_process,
        gpu3=protected_gpu3_process,
    )
    stop_requested = threading.Event()
    previous_signal_mask = _install_deferred_signal_mask()
    root: Path | None = None
    result: dict[str, Any] | None = None
    try:
        with HeldGpuLease.acquire(GPU_LOCK_PATH) as lease:
            inventory = _gpu_inventory()
            gpu2 = inventory.get(PHYSICAL_GPU_INDEX)
            if gpu2 is None or not str(gpu2.get("uuid") or ""):
                raise BACEGlobalGCEK20Error("physical GPU2 inventory is absent")
            gpu2_uuid = str(gpu2["uuid"])
            if (
                int(gpu2.get("memory_used_mib") or 0) > 256
                or _gpu_compute_processes().get(gpu2_uuid)
            ):
                raise BACEGlobalGCEK20Error("physical GPU2 is not idle at launch")
            _verify_protected_gpu_roles(protected, gpu_inventory=inventory)
            execution_contract, checkpoint = _capture_execution_contract(
                project_root=project,
                python=executable,
                config=config_path,
                source_manifest=source,
                native_train_csv=native,
                official_root=official,
                gnn_checkpoint=checkpoint_input,
            )
            if stop_requested.is_set():
                raise BACEGlobalGCEK20Error("controller stop requested before launch")

            root = fresh_output_dir(destination)
            root.chmod(0o700)
            (root / "heartbeats").mkdir(mode=0o700)
            (root / "rounds").mkdir(mode=0o700)
            controller = capture_process_snapshot(os.getpid())
            receipt = {
                "schema_version": CONTROLLER_SCHEMA,
                "controller_id": controller_id,
                "controller_process": controller.to_dict(),
                "execution_contract": execution_contract,
                "round_plan": [asdict(item) for item in ROUND_PLAN],
                "final_k": FINAL_K,
                "physical_gpu_index": PHYSICAL_GPU_INDEX,
                "physical_gpu_uuid": gpu2_uuid,
                "gpu_lock": str(GPU_LOCK_PATH),
                "protected_gpu_roles": {
                    str(index): snapshot.to_dict()
                    for index, snapshot in sorted(protected.items())
                },
                "calibration_loaded": False,
                "test_loaded": False,
                "auto_terminate_uncontrolled_children": False,
                "signal_authority": False,
                "created_at": utc_now(),
            }
            atomic_json(root / "controller_receipt.json", receipt)
            sequence = 1
            catalogs: list[tuple[RoundSpec, list[dict[str, Any]]]] = []
            round_receipts: list[dict[str, Any]] = []
            unique: list[dict[str, Any]] = []
            audit: list[dict[str, Any]] = []
            science_contract = execution_contract["science"]
            provenance = science_contract["oracle"]
            checkpoint_hash = str(provenance["oracle_checkpoint_hash"])

            def revalidate() -> None:
                _drain_deferred_signals(stop_requested)
                lease.verify()
                _snapshot_matches(controller)
                current_inventory = _gpu_inventory()
                current_gpu2 = current_inventory.get(PHYSICAL_GPU_INDEX)
                if current_gpu2 is None or current_gpu2.get("uuid") != gpu2_uuid:
                    raise BACEGlobalGCEK20Error("physical GPU2 UUID changed")
                if _gpu_compute_processes().get(gpu2_uuid, set()):
                    raise BACEGlobalGCEK20Error(
                        "physical GPU2 is not empty outside a science-child lifetime"
                    )
                _verify_protected_gpu_roles(protected, gpu_inventory=inventory)
                _require_execution_contract_unchanged(
                    execution_contract,
                    project_root=project,
                    python=executable,
                    config=config_path,
                    source_manifest=source,
                    native_train_csv=native,
                    official_root=official,
                    gnn_checkpoint=checkpoint_input,
                )
                _drain_deferred_signals(stop_requested)
                if stop_requested.is_set():
                    raise BACEGlobalGCEK20Error(
                        "deferred controller stop requested; no new child or PASS"
                    )

            try:
                for spec in ROUND_PLAN:
                    revalidate()
                    if _gpu_compute_processes().get(gpu2_uuid):
                        raise BACEGlobalGCEK20Error(
                            "physical GPU2 gained a foreign compute process"
                        )
                    round_root = (
                        root
                        / "rounds"
                        / f"round-{spec.round_index}-seed-{spec.incremental_seed}"
                    )
                    command = build_round_command(
                        python=executable,
                        project_root=project,
                        config=config_path,
                        source_manifest=source,
                        native_train_csv=native,
                        official_root=official,
                        gnn_checkpoint=checkpoint,
                        output_root=round_root,
                        spec=spec,
                    )
                    log_path = root / "rounds" / f"round-{spec.round_index}.log"
                    environment = {
                        **os.environ,
                        "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0",
                        "CUDA_VISIBLE_DEVICES": str(PHYSICAL_GPU_INDEX),
                        "AUTODL_PHYSICAL_GPU_INDEX": str(PHYSICAL_GPU_INDEX),
                        "AUTODL_PHYSICAL_GPU_UUID": gpu2_uuid,
                        "PYTHONPATH": str(project),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                    }
                    with log_path.open("xb") as log_handle:
                        child = subprocess.Popen(
                            command,
                            cwd=project,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                        )
                        try:
                            child_snapshot = capture_process_snapshot(child.pid)
                            print(
                                json.dumps(
                                    {
                                        "marker": LAUNCHED_MARKER,
                                        "controller_id": controller_id,
                                        "science_child": child_snapshot.to_dict(),
                                        "round": spec.round_index,
                                        "seed": spec.incremental_seed,
                                        "incremental_raw_budget": (
                                            spec.incremental_raw_budget
                                        ),
                                        "cumulative_raw_budget": (
                                            spec.cumulative_raw_budget
                                        ),
                                        "root": str(round_root),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            return_code, sequence = _wait_naturally(
                                child,
                                root=root,
                                sequence=sequence,
                                controller_id=controller_id,
                                controller=controller,
                                child_snapshot=child_snapshot,
                                spec=spec,
                                protected=protected,
                                gpu_inventory=inventory,
                                lease=lease,
                                stop_requested=stop_requested,
                                heartbeat_interval_seconds=(
                                    heartbeat_interval_seconds
                                ),
                            )
                        except BaseException:
                            while child.poll() is None:
                                try:
                                    time.sleep(1.0)
                                except BaseException:
                                    continue
                            child.wait()
                            raise
                    if stop_requested.is_set():
                        raise BACEGlobalGCEK20Error(
                            "deferred controller stop completed after child exit"
                        )
                    revalidate()
                    if return_code == 0:
                        rows, child_evidence = validate_successful_raw_round(
                            round_root,
                            science_contract=science_contract,
                            seed=spec.incremental_seed,
                            raw_budget=spec.incremental_raw_budget,
                        )
                        child_status = "TERMINAL_SUCCESS"
                        child_marker = file_identity(round_root / "PASS")
                    elif return_code == RAW_SHORTFALL_EXIT_CODE:
                        rows, child_evidence = validate_shortfall_raw_round(
                            round_root,
                            science_contract=science_contract,
                            seed=spec.incremental_seed,
                            raw_budget=spec.incremental_raw_budget,
                        )
                        child_status = "EXPECTED_SHORTFALL"
                        child_marker = file_identity(round_root / "RAW_SHORTFALL")
                    else:
                        raise BACEGlobalGCEK20Error(
                            f"round {spec.round_index} science child exited "
                            f"unexpected code {return_code}"
                        )
                    catalog_path = round_root / "native" / "native_rule_catalog.jsonl"
                    catalogs.append((spec, rows))
                    unique, audit = merge_unique_rules(
                        catalogs, expected_checkpoint_hash=checkpoint_hash
                    )
                    round_receipt = {
                        "schema_version": ROUND_SCHEMA,
                        "round": spec.round_index,
                        "seed": spec.incremental_seed,
                        "incremental_raw_budget": spec.incremental_raw_budget,
                        "cumulative_seeds": list(spec.cumulative_seeds),
                        "cumulative_raw_budget": spec.cumulative_raw_budget,
                        "science_child": child_snapshot.to_dict(),
                        "science_child_observed_on_physical_gpu2": True,
                        "science_exit_code": return_code,
                        "science_status": child_status,
                        "science_evidence": child_evidence,
                        "science_marker": child_marker,
                        "catalog": file_identity(catalog_path),
                        "valid_catalog_rows": len(rows),
                        "cumulative_valid_unique_rules": len(unique),
                        "execution_contract_sha256": stable_sha256(
                            execution_contract
                        ),
                        "checkpoint_unchanged": True,
                        "calibration_loaded": False,
                        "test_loaded": False,
                        "completed_at": utc_now(),
                    }
                    atomic_json(
                        root / "rounds" / f"round-{spec.round_index}.json",
                        round_receipt,
                    )
                    round_receipts.append(round_receipt)
                    if len(unique) >= FINAL_K:
                        break

                if len(unique) < FINAL_K:
                    raise BACEGlobalGCEK20Error(
                        f"raw budget 500 yielded only {len(unique)} "
                        "semantic-unique valid native rules"
                    )
                selected: list[dict[str, Any]] = []
                for rank, row in enumerate(unique[:FINAL_K], start=1):
                    item = dict(row)
                    item["rank"] = rank
                    item["extension_rank"] = rank
                    selected.append(item)
                result = _publish_release_candidate(
                    root=root,
                    controller_id=controller_id,
                    controller=controller,
                    selected=selected,
                    audit=audit,
                    provenance=provenance,
                    execution_contract=execution_contract,
                    round_receipts=round_receipts,
                    physical_gpu_uuid=gpu2_uuid,
                    sequence=sequence,
                    revalidate=revalidate,
                    stop_requested=stop_requested,
                )
            except BaseException:
                # A live child has already been waited naturally before this
                # handler can run.  Never write failure state over a PASS root.
                if root is not None and not (root / "PASS").exists():
                    atomic_json(
                        root / "FAILED.json",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "status": "FAILED",
                            "controller_id": controller_id,
                            "observed_at": utc_now(),
                            "auto_terminate_uncontrolled_children": False,
                            "signal_sent": False,
                        },
                    )
                raise
    finally:
        _drain_deferred_signals(stop_requested)
        _restore_signal_mask(previous_signal_mask)
    if result is None:
        raise BACEGlobalGCEK20Error("K20 controller returned without a result")
    print(PASS_MARKER, flush=True)
    return result


__all__ = [
    "BACEGlobalGCEK20Error",
    "FINAL_K",
    "GPU_LOCK_PATH",
    "LAUNCHED_MARKER",
    "PASS_MARKER",
    "PHYSICAL_GPU_INDEX",
    "RAW_SHORTFALL_EXIT_CODE",
    "RAW_SHORTFALL_MARKER",
    "ROUND_PLAN",
    "RoundSpec",
    "build_round_command",
    "merge_unique_rules",
    "parse_protected_processes",
    "run_extension",
    "run_raw_round",
    "unblock_deferred_signals_for_science_child",
    "validate_catalog_row",
    "validate_round_plan",
    "validate_shortfall_raw_round",
    "validate_successful_raw_round",
]
