"""Read-only recovery of completed TasteMolNet T8 target branches.

This module is intentionally dataset specific.  It never resumes or mutates a
source branch.  It validates and retains both completed 25-epoch branch trees,
copies their immutable bytes into one fresh private state tree, replays native
rule application on the bounded train-only cohort, and emits the exact deadline
terminal consumed by the existing managed-v2 T8 adoption path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from scripts.autodl import run_tastemolnet_t8_deadline as deadline
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    apply_rule_to_parent,
)
from src.baselines.globalgce_mutagenicity_adapter import (
    OFFICIAL_AFFINE_EDGE_HARD_DECODE,
    OfficialGlobalGCEMutagenicityGenerator,
    _cohort_resume_identity,
    _stable_split,
)
from src.baselines.globalgce_resumable import (
    validate_globalgce_epoch_checkpoint_identity,
)
from src.baselines.tastemolnet_globalgce_smoke import (
    BRANCH_SCHEMA,
    CHECKPOINT_SEAL_SCHEMA,
    DATASET,
    GINE_PAYLOAD_FILES,
    METHOD,
    NUM_CLASSES,
    OFFICIAL_GLOBALGCE_COMMIT,
    PASS_MARKER,
    SCIENCE_SCHEMA,
    SOURCE_LABEL,
    STAGE,
    TARGET_BRANCHES,
    ZERO_CANDIDATE_RECOVERY_EPOCHS,
    FrozenTasteGINEScorer,
    TasteGlobalGCESmokeConfig,
    TasteGlobalGCESmokeError,
    _HeldBranchDirectory,
    _canonical_rule_action,
    _canonical_sha256,
    _deduplicate_generated_candidates,
    _validate_official_startup_identity,
    load_taste_train_cohort,
    merge_branch_rule_catalogs,
    run_resumed_target_branch,
    select_bounded_sweet_parents,
    validate_candidates_with_original_gine,
    validate_science_summary,
)
from src.utils.retained_output_directory import (
    FreshOutputDirectory,
    RetainedOutputTree,
    prepare_terminal_output,
)


SALVAGE_SCHEMA = "tastemolnet_t8_branch_salvage_v1"
BRANCH_SEAL_SCHEMA = "tastemolnet_t8_read_only_branch_seal_v1"
RERUN_SCHEMA = "tastemolnet_t8_single_branch_rerun_request_v1"
SALVAGE_MARKER = "[TASTE_T8_SALVAGE_PASS]"
REQUIRED_BRANCH_FILES = frozenset(
    {
        "globalgce_model.pt",
        "globalgce_rules.pt",
        "training_core_summary.json",
        "native_rule_catalog.jsonl",
        "native_rule_rejections.jsonl",
        "official_api_signature.json",
        "python_module_provenance.json",
        "globalgce_training_checkpoints/training_checkpoint.pt",
        "globalgce_training_checkpoints/training_heartbeat.json",
        "sealed-planned-checkpoint/training_checkpoint.pt",
        "sealed-planned-checkpoint/training_heartbeat.json",
    }
)


def _json_bytes(value: Any) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 salvage cannot read {label}") from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 salvage {label} is not one object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 salvage cannot read {label}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TasteGlobalGCESmokeError(
                f"T8 salvage {label} row {number} is invalid JSON"
            ) from exc
        if type(row) is not dict:
            raise TasteGlobalGCESmokeError(
                f"T8 salvage {label} row {number} is not an object"
            )
        rows.append(row)
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        payload = _json_bytes(dict(value))
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short T8 salvage receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _absolute_existing_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGlobalGCESmokeError(f"T8 salvage {label} must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise TasteGlobalGCESmokeError(
            f"T8 salvage {label} must be one real directory"
        )
    return resolved


def _open_tree(path: Path) -> tuple[int, RetainedOutputTree]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return descriptor, RetainedOutputTree.capture(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _writer_fds(roots: Sequence[Path], *, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Return live write-capable descriptors resolving inside source roots."""

    if not proc_root.is_dir():
        raise TasteGlobalGCESmokeError(
            "T8 salvage requires procfs to prove that source branches have no writer"
        )
    normalized = tuple(str(root) + os.sep for root in roots)
    writers: list[dict[str, Any]] = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                flags_text = (process / "fdinfo" / descriptor.name).read_text(
                    encoding="utf-8"
                )
            except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
                continue
            if target.endswith(" (deleted)"):
                target = target[: -len(" (deleted)")]
            if not any(target == str(root) or target.startswith(prefix) for root, prefix in zip(roots, normalized, strict=True)):
                continue
            flags_line = next(
                (line for line in flags_text.splitlines() if line.startswith("flags:")),
                "",
            )
            try:
                flags = int(flags_line.split()[1], 8)
            except (IndexError, ValueError):
                raise TasteGlobalGCESmokeError(
                    "T8 salvage could not determine one source descriptor mode"
                )
            if flags & os.O_ACCMODE != os.O_RDONLY:
                writers.append(
                    {"pid": int(process.name), "fd": int(descriptor.name), "path": target}
                )
    return writers


def write_single_branch_rerun_request(
    path: Path,
    failures: Mapping[int, str],
) -> dict[str, Any]:
    invalid = sorted(failures)
    if not invalid or any(target not in TARGET_BRANCHES for target in invalid):
        raise TasteGlobalGCESmokeError("T8 rerun request target set is invalid")
    receipt = {
        "schema_version": RERUN_SCHEMA,
        "status": "RERUN_REQUIRED",
        "invalid_target_branches": invalid,
        "valid_target_branches_preserved": [
            target for target in TARGET_BRANCHES if target not in invalid
        ],
        "rerun_both_branches": len(invalid) == len(TARGET_BRANCHES),
        "source_artifacts_mutated": False,
        "reasons": {str(target): failures[target] for target in invalid},
    }
    _atomic_json(path, receipt)
    return receipt


def _content_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    files = inventory.get("files")
    if type(files) is not dict:
        raise TasteGlobalGCESmokeError("T8 branch inventory is malformed")
    return {
        "files": {
            relative: {
                "bytes": evidence.get("bytes"),
                "sha256": evidence.get("sha256"),
            }
            for relative, evidence in sorted(files.items())
        }
    }


def _feature_schema_hash(identity: Mapping[str, Any]) -> str:
    oracle = identity.get("oracle_identity")
    inventory = oracle.get("inventory") if type(oracle) is dict else None
    if type(inventory) is not list:
        raise TasteGlobalGCESmokeError("T8 branch GINE inventory is absent")
    matches = [
        row.get("sha256")
        for row in inventory
        if type(row) is dict and row.get("name") == "feature_schema.json"
    ]
    if len(matches) != 1 or type(matches[0]) is not str:
        raise TasteGlobalGCESmokeError("T8 branch feature schema identity is absent")
    return matches[0]


def _reload_epoch_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    expected_next_epoch: int,
) -> dict[str, Any]:
    """Independently deserialize and validate one retained epoch checkpoint."""

    import torch

    # The retained schema is compatible with PyTorch's restricted loader.
    # Never execute a general pickle merely to inspect a recovery source.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise TasteGlobalGCESmokeError("T8 retained epoch checkpoint is not a mapping")
    try:
        validated = validate_globalgce_epoch_checkpoint_identity(payload, identity)
    except (TypeError, ValueError) as exc:
        raise TasteGlobalGCESmokeError(
            "T8 retained epoch checkpoint identity failed independent reload"
        ) from exc
    if (
        payload.get("next_epoch") != expected_next_epoch
        or not isinstance(payload.get("model_state"), Mapping)
        or not isinstance(payload.get("optimizer_state"), Mapping)
        or not isinstance(payload.get("scheduler_state"), Mapping)
        or payload.get("torch_rng_state") is None
        or payload.get("numpy_rng_state") is None
        or payload.get("python_rng_state") is None
    ):
        raise TasteGlobalGCESmokeError(
            "T8 retained epoch checkpoint lacks complete resumable state"
        )
    return validated


@dataclass(slots=True)
class ValidatedBranch:
    target: int
    root: Path
    root_fd: int
    tree: RetainedOutputTree
    catalog: list[dict[str, Any]]
    evidence: dict[str, Any]
    seal: dict[str, Any]

    def close(self) -> None:
        self.tree.close()
        os.close(self.root_fd)


class T8BranchScientificFailure(TasteGlobalGCESmokeError):
    """A smoke failure attributable to an exact subset of target branches."""

    def __init__(self, message: str, *, invalid_targets: Sequence[int]) -> None:
        super().__init__(message)
        self.invalid_targets = tuple(sorted(set(invalid_targets)))


def validate_source_branch(
    *,
    root: Path,
    target: int,
    checkpoint_id: str,
    feature_schema_sha256: str,
    proc_root: Path = Path("/proc"),
) -> ValidatedBranch:
    if target not in TARGET_BRANCHES:
        raise TasteGlobalGCESmokeError("T8 salvage target must be 0 or 2")
    root = _absolute_existing_directory(root, label=f"target-{target} root")
    writers = _writer_fds([root], proc_root=proc_root)
    if writers:
        raise TasteGlobalGCESmokeError(
            f"T8 target-{target} source still has write-capable descriptors"
        )
    descriptor, tree = _open_tree(root)
    try:
        inventory = tree.revalidate()
        missing = sorted(REQUIRED_BRANCH_FILES - set(inventory["files"]))
        if missing:
            raise TasteGlobalGCESmokeError(
                f"T8 target-{target} branch is incomplete: {missing}"
            )
        core = _read_json(root / "training_core_summary.json", label="training core")
        terminal = _read_json(
            root / "globalgce_training_checkpoints/training_heartbeat.json",
            label="terminal heartbeat",
        )
        planned = _read_json(
            root / "sealed-planned-checkpoint/training_heartbeat.json",
            label="planned heartbeat",
        )
        identity = core.get("training_resume_identity")
        oracle = identity.get("oracle_identity") if type(identity) is dict else None
        training_config = identity.get("training_config") if type(identity) is dict else None
        if (
            core.get("dataset_name") != "TasteMolNet"
            or core.get("num_classes") != NUM_CLASSES
            or core.get("source_label") != SOURCE_LABEL
            or core.get("target_label") != target
            or core.get("prediction_backend") != "frozen_gine_differentiable_bridge"
            or core.get("rf_oracle_used") is not False
            or core.get("trained_once") is not True
            or core.get("rule_selection_performed_once") is not True
            or type(identity) is not dict
            or identity.get("dataset") != "TasteMolNet"
            or identity.get("num_classes") != NUM_CLASSES
            or identity.get("source_label") != SOURCE_LABEL
            or identity.get("target_label") != target
            or type(oracle) is not dict
            or oracle.get("backend") != "frozen_gine"
            or oracle.get("checkpoint_id") != checkpoint_id
            or _feature_schema_hash(identity) != feature_schema_sha256
            or type(training_config) is not dict
            or training_config.get("epochs") != ZERO_CANDIDATE_RECOVERY_EPOCHS
            or training_config.get("gspan_exact_top_k_pruning") is not True
            or core.get("gspan_exact_top_k_pruning") is not True
            or core.get("gnn_checkpoint_sha256") != checkpoint_id
            or core.get("globalgce_model_checkpoint_sha256")
            != _sha256_file(root / "globalgce_model.pt")
            or core.get("rules_checkpoint_sha256")
            != _sha256_file(root / "globalgce_rules.pt")
        ):
            raise TasteGlobalGCESmokeError(
                f"T8 target-{target} training/GINE/feature identity changed"
            )
        planned_checkpoint = root / "sealed-planned-checkpoint/training_checkpoint.pt"
        terminal_checkpoint = root / "globalgce_training_checkpoints/training_checkpoint.pt"
        planned_sha = _sha256_file(planned_checkpoint)
        resume_identity_sha = str(core.get("training_resume_identity_sha256") or "")
        if (
            terminal.get("stage") != "complete"
            or terminal.get("next_epoch") != ZERO_CANDIDATE_RECOVERY_EPOCHS + 1
            or planned.get("stage") != "training"
            or planned.get("next_epoch") != 1
            or planned.get("resume_identity_sha256") != resume_identity_sha
            or terminal.get("resume_identity_sha256") != resume_identity_sha
        ):
            raise TasteGlobalGCESmokeError(
                f"T8 target-{target} checkpoint/resume closure changed"
            )
        _reload_epoch_checkpoint(
            planned_checkpoint,
            identity=identity,
            expected_next_epoch=1,
        )
        _reload_epoch_checkpoint(
            terminal_checkpoint,
            identity=identity,
            expected_next_epoch=ZERO_CANDIDATE_RECOVERY_EPOCHS + 1,
        )
        catalog = _read_jsonl(root / "native_rule_catalog.jsonl", label="rule catalog")
        if not catalog:
            raise TasteGlobalGCESmokeError(f"T8 target-{target} has no native rules")
        rule_hashes: set[str] = set()
        for row in catalog:
            GlobalGCENativeRule.from_payload(row)
            action = _canonical_rule_action(row)
            if action["rule_hash"] in rule_hashes:
                raise TasteGlobalGCESmokeError(
                    f"T8 target-{target} repeats one canonical rule"
                )
            rule_hashes.add(action["rule_hash"])
            if (
                row.get("source_split") != "train"
                or row.get("oracle_backend") != "gnn"
                or row.get("classifier_family") != "gine"
                or row.get("rf_oracle_used") is not False
                or row.get("oracle_checkpoint_hash") != checkpoint_id
                or row.get("edge_score_contract")
                != "pinned_official_unbounded_affine_class_scores"
                or row.get("edge_score_hard_decode")
                != OFFICIAL_AFFINE_EDGE_HARD_DECODE
            ):
                raise TasteGlobalGCESmokeError(
                    f"T8 target-{target} native rule scientific contract changed"
                )
        startup = _validate_official_startup_identity(
            completed_tree=tree,
            api_relative="official_api_signature.json",
            provenance_relative="python_module_provenance.json",
            # The retained training core deliberately does not duplicate
            # startup evidence.  Reconstruct only the five binding fields
            # from the independently retained documents and their bytes.
            training_summary={
                "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
                "official_api_signature_sha256": inventory["files"]
                ["official_api_signature.json"]["sha256"],
                "python_module_provenance_sha256": inventory["files"]
                ["python_module_provenance.json"]["sha256"],
                "isolated_python": True,
                "no_user_site": True,
            },
        )
        sealed_files = {
            "training_checkpoint.pt": {
                "bytes": planned_checkpoint.stat().st_size,
                "sha256": planned_sha,
            },
            "training_heartbeat.json": {
                "bytes": (root / "sealed-planned-checkpoint/training_heartbeat.json").stat().st_size,
                "sha256": _sha256_file(
                    root / "sealed-planned-checkpoint/training_heartbeat.json"
                ),
            },
        }
        sealed_inventory_sha = _canonical_sha256(sealed_files)
        checkpoint_stat = planned_checkpoint.stat()
        heartbeat_stat = (
            root / "sealed-planned-checkpoint/training_heartbeat.json"
        ).stat()
        evidence = {
            "schema_version": BRANCH_SCHEMA,
            **startup,
            "target_label": target,
            "source_label": SOURCE_LABEL,
            "num_classes": NUM_CLASSES,
            "planned_checkpoint_stop_observed": True,
            "planned_checkpoint_next_epoch": 1,
            "planned_checkpoint_sha256": planned_sha,
            "resume_checkpoint_adopted": True,
            "resume_checkpoint_sha256": planned_sha,
            "resume_identity_sha256": resume_identity_sha,
            "model_state_restored": True,
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
            "terminal_next_epoch": terminal["next_epoch"],
            "terminal_training_checkpoint_sha256": _sha256_file(terminal_checkpoint),
            "terminal_training_core_sha256": _sha256_file(
                root / "training_core_summary.json"
            ),
            "terminal_model_checkpoint_sha256": _sha256_file(
                root / "globalgce_model.pt"
            ),
            "terminal_rule_checkpoint_sha256": _sha256_file(
                root / "globalgce_rules.pt"
            ),
            "native_rule_catalog_sha256": _sha256_file(
                root / "native_rule_catalog.jsonl"
            ),
            "valid_native_rule_count": len(catalog),
            "native_rule_edge_score_contract": (
                "pinned_official_unbounded_affine_class_scores"
            ),
            "native_rule_edge_score_hard_decode": OFFICIAL_AFFINE_EDGE_HARD_DECODE,
            "raw_generated_count": 1,
            "train_only": True,
            "external_validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "rf_oracle_used": False,
            "checkpoint_seal_schema_version": CHECKPOINT_SEAL_SCHEMA,
            "checkpoint_seal_pass": True,
            "checkpoint_writer_unwound": True,
            "checkpoint_durable_flush": True,
            "checkpoint_sealed_directory": "sealed-planned-checkpoint",
            "checkpoint_sealed_inventory_sha256": sealed_inventory_sha,
            "checkpoint_resume_copy_inventory_sha256": sealed_inventory_sha,
            "checkpoint_no_follow_identity_verified": True,
            "checkpoint_independent_reopen_verified": True,
            "checkpoint_callback_ctime_settled": False,
            "checkpoint_callback_ctime_ns": int(checkpoint_stat.st_ctime_ns),
            "checkpoint_sealed_ctime_ns": int(checkpoint_stat.st_ctime_ns),
            "heartbeat_callback_ctime_ns": int(heartbeat_stat.st_ctime_ns),
            "heartbeat_sealed_ctime_ns": int(heartbeat_stat.st_ctime_ns),
        }
        seal = {
            "schema_version": BRANCH_SEAL_SCHEMA,
            "status": "PASS",
            "source_root": str(root),
            "target_label": target,
            "source_read_only": True,
            "active_writer_count": 0,
            "inventory_sha256": inventory["inventory_sha256"],
            "content_inventory": _content_inventory(inventory),
            "model_sha256": evidence["terminal_model_checkpoint_sha256"],
            "rules_sha256": evidence["terminal_rule_checkpoint_sha256"],
            "catalog_sha256": evidence["native_rule_catalog_sha256"],
            "oracle_checkpoint_hash": checkpoint_id,
            "feature_schema_sha256": feature_schema_sha256,
            "training_resume_identity_sha256": resume_identity_sha,
            "native_train_cohort": identity["native_train_cohort"],
            "source_train_cohort": identity["source_train_cohort"],
            "official_source_identity_sha256": identity[
                "official_source_identity"
            ]["identity_sha256"],
            "train_only": True,
            "test_loaded": False,
            "exact_top_k_proof_verified": True,
        }
        tree.revalidate()
        return ValidatedBranch(target, root, descriptor, tree, catalog, evidence, seal)
    except BaseException:
        tree.close()
        os.close(descriptor)
        raise


def materialize_smoke_candidates(
    catalogs: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    parents: Sequence[Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Apply retained native rules; never train or alter a branch checkpoint."""

    if set(catalogs) != set(TARGET_BRANCHES):
        raise TasteGlobalGCESmokeError("T8 salvage requires both branch catalogs")
    records: dict[int, list[dict[str, Any]]] = {0: [], 2: []}
    application_failures = {0: 0, 2: 0}
    for target in TARGET_BRANCHES:
        for raw_rule in catalogs[target]:
            rule = GlobalGCENativeRule.from_payload(raw_rule)
            for parent in parents:
                try:
                    applications = apply_rule_to_parent(parent.smiles, rule)
                except Exception:
                    application_failures[target] += 1
                    continue
                for application in applications:
                    if application.get("valid") is not True:
                        continue
                    records[target].append(
                        {
                            "raw_smiles": str(application["canonical_smiles"]),
                            "source_parent_id": parent.parent_id,
                            "source_parent_smiles": parent.smiles,
                            "source_split": "train",
                            "generator_method": METHOD,
                            "native_conversion_ok": True,
                            "native_codec_decoded": True,
                        }
                    )
    if any(not records[target] for target in TARGET_BRANCHES):
        missing = [target for target in TARGET_BRANCHES if not records[target]]
        raise T8BranchScientificFailure(
            f"T8 salvage native rule application produced no candidates for {missing}",
            invalid_targets=missing,
        )
    return records, {
        "application_engine": "official_attachment_aware_lhs_to_rhs",
        "target_0_application_failures": application_failures[0],
        "target_2_application_failures": application_failures[2],
        "source_branch_mutated": False,
    }


def _copy_validated_branches(
    branches: Mapping[int, ValidatedBranch],
    *,
    state_root: Path,
) -> tuple[int, RetainedOutputTree]:
    if state_root.exists() or state_root.is_symlink():
        raise TasteGlobalGCESmokeError("T8 salvage private state root must be fresh")
    state_root.mkdir(parents=True, mode=0o700)
    for target in TARGET_BRANCHES:
        shutil.copytree(
            branches[target].root,
            state_root / f"target-{target}",
            symlinks=False,
            copy_function=shutil.copy2,
        )
    _atomic_json(
        state_root / "salvage_receipt.json",
        {
            "schema_version": SALVAGE_SCHEMA,
            "status": "SOURCE_BRANCHES_COPIED_AND_SEALED",
            "source_artifacts_mutated": False,
            "branches": {str(target): branches[target].seal for target in TARGET_BRANCHES},
        },
    )
    descriptor, tree = _open_tree(state_root)
    try:
        copied = tree.revalidate()
        for target in TARGET_BRANCHES:
            source = branches[target].tree.revalidate()
            prefix = f"target-{target}/"
            observed = {
                relative[len(prefix) :]: {
                    "bytes": evidence["bytes"],
                    "sha256": evidence["sha256"],
                }
                for relative, evidence in copied["files"].items()
                if relative.startswith(prefix)
            }
            expected = {
                relative: {
                    "bytes": evidence["bytes"],
                    "sha256": evidence["sha256"],
                }
                for relative, evidence in source["files"].items()
            }
            if observed != expected:
                raise TasteGlobalGCESmokeError(
                    f"T8 target-{target} sealed copy differs from its source bytes"
                )
        return descriptor, tree
    except BaseException:
        tree.close()
        os.close(descriptor)
        raise


def _strict_flip_counts_by_branch(
    candidates: Sequence[Mapping[str, Any]],
    *,
    scorer: Any,
) -> dict[int, int]:
    values: list[str] = []
    for row in candidates:
        values.extend((str(row["parent"]), str(row["candidate"])))
    predictions = scorer.score_smiles(values)
    if len(predictions) != len(values):
        raise TasteGlobalGCESmokeError("T8 salvage diagnostic prediction count changed")
    counts = {0: 0, 2: 0}
    for index, row in enumerate(candidates):
        before = predictions[2 * index]
        after = predictions[2 * index + 1]
        if (
            before.get("checkpoint_id") == scorer.checkpoint_id
            and after.get("checkpoint_id") == scorer.checkpoint_id
            and before.get("predicted_label") == SOURCE_LABEL
            and after.get("predicted_label") in TARGET_BRANCHES
        ):
            for target in row["target_branches"]:
                counts[int(target)] += 1
    return counts


def run_salvage(
    *,
    config: Path,
    attempt_id: str,
    recovery_source_attempt_id: str,
    target_roots: Mapping[int, Path],
    t3_output: Path,
    t4_output: Path,
    gnn_checkpoint: Path,
    train_csv: Path,
    official_root: Path,
    state_root: Path,
    output_root: Path,
    rerun_request: Path,
    device: str,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    if set(target_roots) != set(TARGET_BRANCHES):
        raise TasteGlobalGCESmokeError("T8 salvage requires target roots 0 and 2")
    preflight, payloads, train_payload = deadline.deadline_preflight(
        SimpleNamespace(
            config=config,
            set=["inference.fallback_to_heuristic=false"],
            attempt_id=attempt_id,
            zero_candidate_recovery=True,
            recovery_source_attempt_id=recovery_source_attempt_id,
            t3_output=t3_output,
            t4_output=t4_output,
            gnn_checkpoint=gnn_checkpoint,
            train_csv=train_csv,
            official_root=official_root,
        )
    )
    checkpoint_id = str(preflight["checkpoint_id"])
    feature_schema_sha = _sha256_file(gnn_checkpoint / "feature_schema.json")
    branches: dict[int, ValidatedBranch] = {}
    failures: dict[int, str] = {}
    for target in TARGET_BRANCHES:
        try:
            branches[target] = validate_source_branch(
                root=target_roots[target],
                target=target,
                checkpoint_id=checkpoint_id,
                feature_schema_sha256=feature_schema_sha,
                proc_root=proc_root,
            )
        except Exception as exc:
            failures[target] = f"{type(exc).__name__}:{exc}"
    if failures:
        for branch in branches.values():
            branch.close()
        write_single_branch_rerun_request(rerun_request, failures)
        raise TasteGlobalGCESmokeError(
            "T8 salvage rejected source branch(es): " + ",".join(map(str, sorted(failures)))
        )
    state_fd = -1
    state_tree: RetainedOutputTree | None = None
    prepared = None
    try:
        scorer = FrozenTasteGINEScorer(
            payloads,
            device=device,
            batch_size=TasteGlobalGCESmokeConfig().oracle_batch_size,
        )
        cohort = load_taste_train_cohort(
            train_payload,
            expected_row_count=int(preflight["train_rows"]),
            expected_label_counts=preflight["train_label_counts"],
        )
        science_config = TasteGlobalGCESmokeConfig(
            epochs=ZERO_CANDIDATE_RECOVERY_EPOCHS
        )
        parents, selection = select_bounded_sweet_parents(
            cohort,
            scorer=scorer,
            config=science_config,
        )
        source_train_idx, source_val_idx = _stable_split(
            parents,
            seed=science_config.seed,
        )
        expected_source_cohort = _cohort_resume_identity(
            parents,
            train_idx=source_train_idx,
            val_idx=source_val_idx,
        )
        if any(
            branches[target].seal["source_train_cohort"]
            != expected_source_cohort
            for target in TARGET_BRANCHES
        ):
            raise TasteGlobalGCESmokeError(
                "T8 salvage retained branch source cohort differs from frozen train selection"
            )
        if (
            branches[0].seal["native_train_cohort"]
            != branches[2].seal["native_train_cohort"]
            or branches[0].seal["official_source_identity_sha256"]
            != branches[2].seal["official_source_identity_sha256"]
        ):
            raise TasteGlobalGCESmokeError(
                "T8 salvage target branches used different train/official identities"
            )
        source_trees = {target: branches[target].tree for target in TARGET_BRANCHES}
        _merged_rules, rule_merge = merge_branch_rule_catalogs(
            branch_trees=source_trees,
            checkpoint_id=checkpoint_id,
        )
        try:
            generated, materialization = materialize_smoke_candidates(
                {target: branches[target].catalog for target in TARGET_BRANCHES},
                parents=parents,
            )
        except T8BranchScientificFailure as exc:
            write_single_branch_rerun_request(
                rerun_request,
                {target: str(exc) for target in exc.invalid_targets},
            )
            raise
        candidates, candidate_merge = _deduplicate_generated_candidates(
            generated,
            selected_parents=parents,
        )
        try:
            strict = validate_candidates_with_original_gine(
                candidates,
                scorer=scorer,
                checkpoint_id=checkpoint_id,
                minimum_strict_flips_per_branch=1,
            )
        except TasteGlobalGCESmokeError as exc:
            counts = _strict_flip_counts_by_branch(candidates, scorer=scorer)
            invalid = [target for target in TARGET_BRANCHES if counts[target] < 1]
            if invalid:
                write_single_branch_rerun_request(
                    rerun_request,
                    {target: f"strict_flip_count={counts[target]}:{exc}" for target in invalid},
                )
            raise
        for target in TARGET_BRANCHES:
            branches[target].evidence["raw_generated_count"] = len(generated[target])
        state_fd, state_tree = _copy_validated_branches(
            branches,
            state_root=state_root,
        )
        state_inventory = state_tree.revalidate()
        science = {
            "schema_version": SCIENCE_SCHEMA,
            "stage": STAGE,
            "dataset": DATASET,
            "method": METHOD,
            "status": "SCIENCE_PASS_PENDING_TERMINAL_COMMIT",
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "target_branches": list(TARGET_BRANCHES),
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "oracle_checkpoint_hash": checkpoint_id,
            "temperature_hex": float(scorer.temperature).hex(),
            "config": science_config.to_dict(),
            "train_boundary": {
                "train_loaded": True,
                "train_row_count": cohort.train_row_count,
                "train_label_counts": dict(cohort.label_counts),
                "external_validation_loaded": False,
                "calibration_loaded": False,
                "test_loaded": False,
                "data_reprepared": False,
                "data_redistributed": False,
            },
            "selection": selection,
            "branches": {
                "0": branches[0].evidence,
                "2": branches[2].evidence,
            },
            "rule_merge": rule_merge,
            "candidate_merge": candidate_merge,
            "strict_flip_validation": strict,
            "private_state": {
                "inventory_sha256": state_inventory["inventory_sha256"],
                "file_count": len(state_inventory["files"]),
                "aggregate_only_terminal": True,
                "private_rows_serialized_to_terminal": False,
            },
            "native_action_preserved": True,
            "binary_classifier_trained": False,
            "rf_oracle_used": False,
            "gnn_retrained": False,
            "gnn_ablation_started": False,
            "dataset_redistributed": False,
            "per_example_terminal_payload": False,
        }
        validate_science_summary(science)
        output = FreshOutputDirectory.create(output_root)
        manifest = {
            **preflight,
            "status": "PASS",
            "science_sha256": _sha256_bytes(_json_bytes(science)),
            "strict_flip_count": strict["strict_flip_count"],
            "destination_distribution": strict["destination_distribution"],
            "canonical_rule_merge_complete": True,
            "canonical_candidate_dedup_complete": True,
            "untargeted_strict_flip_complete": True,
        }
        gate = {
            "schema_version": deadline.SCHEMA,
            "status": "PASS",
            "marker": PASS_MARKER,
            "attempt_id": attempt_id,
            "checkpoint_id": checkpoint_id,
            "target_branches": [0, 2],
            "strict_flip_count": strict["strict_flip_count"],
            "destination_distribution": strict["destination_distribution"],
            "rf_oracle_used": False,
            "test_loaded": False,
            "zero_candidate_recovery": preflight["zero_candidate_recovery"],
        }
        output.write_new("science.json", _json_bytes(science))
        output.write_new("manifest.json", _json_bytes(manifest))
        output.write_new("gate.json", _json_bytes(gate))
        prepared = prepare_terminal_output(
            output,
            marker_name="PASS",
            marker_payload=(PASS_MARKER + "\n").encode("utf-8"),
        )

        def _closure() -> None:
            if _writer_fds(
                [branches[target].root for target in TARGET_BRANCHES],
                proc_root=proc_root,
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 source branch acquired a writer during salvage"
                )
            for branch in branches.values():
                branch.tree.revalidate()
            state_tree.revalidate()

        prepared.commit(retained_input_closure=_closure)
        return {
            "schema_version": SALVAGE_SCHEMA,
            "status": "PASS",
            "marker": SALVAGE_MARKER,
            "t8_marker": PASS_MARKER,
            "attempt_id": attempt_id,
            "source_attempt_id": recovery_source_attempt_id,
            "output_root": str(output_root),
            "state_root": str(state_root),
            "source_artifacts_mutated": False,
            "source_branch_seals": {
                str(target): branches[target].seal for target in TARGET_BRANCHES
            },
            "rule_merge": rule_merge,
            "candidate_merge": candidate_merge,
            "strict_flip_validation": strict,
            "materialization": materialization,
            "single_branch_rerun_policy": True,
        }
    finally:
        if prepared is not None:
            prepared.close()
        if state_tree is not None:
            state_tree.close()
        if state_fd >= 0:
            os.close(state_fd)
        for branch in branches.values():
            branch.close()


def run_single_branch_recovery(
    *,
    config: Path,
    attempt_id: str,
    recovery_source_attempt_id: str,
    target: int,
    t3_output: Path,
    t4_output: Path,
    gnn_checkpoint: Path,
    train_csv: Path,
    official_root: Path,
    state_root_path: Path,
    gspan_scratch_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run one bounded fresh branch named by a prior salvage receipt."""

    if target not in TARGET_BRANCHES:
        raise TasteGlobalGCESmokeError("T8 recovery target must be exactly 0 or 2")
    preflight, payloads, train_payload = deadline.deadline_preflight(
        SimpleNamespace(
            config=config,
            set=["inference.fallback_to_heuristic=false"],
            attempt_id=attempt_id,
            zero_candidate_recovery=True,
            recovery_source_attempt_id=recovery_source_attempt_id,
            t3_output=t3_output,
            t4_output=t4_output,
            gnn_checkpoint=gnn_checkpoint,
            train_csv=train_csv,
            official_root=official_root,
        )
    )
    for path, label in (
        (state_root_path, "state root"),
        (gspan_scratch_root, "gSpan scratch root"),
    ):
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise TasteGlobalGCESmokeError(f"T8 branch recovery {label} must be absolute")
        if path.exists() or path.is_symlink():
            raise TasteGlobalGCESmokeError(f"T8 branch recovery {label} must be fresh")
        path.parent.resolve(strict=True)
    gspan_scratch_root.mkdir(mode=0o700)
    science_config = TasteGlobalGCESmokeConfig(
        epochs=ZERO_CANDIDATE_RECOVERY_EPOCHS
    )
    scorer = FrozenTasteGINEScorer(
        payloads,
        device=device,
        batch_size=science_config.oracle_batch_size,
    )
    cohort = load_taste_train_cohort(
        train_payload,
        expected_row_count=int(preflight["train_rows"]),
        expected_label_counts=preflight["train_label_counts"],
    )
    parents, selection = select_bounded_sweet_parents(
        cohort,
        scorer=scorer,
        config=science_config,
    )
    state_root = FreshOutputDirectory.create(state_root_path)
    branch = _HeldBranchDirectory.create(state_root, target_label=target)
    branch_tree: RetainedOutputTree | None = None
    receipt_leaf = None
    try:
        generator = OfficialGlobalGCEMutagenicityGenerator(
            preflight["official_root"],
            native_train_csv=preflight["train_csv"],
            dataset_name="TasteMolNet",
            min_freq=science_config.min_freq,
            frozen_gine_checkpoint=preflight["checkpoint_dir"],
            source_label=SOURCE_LABEL,
            target_label=target,
            num_classes=NUM_CLASSES,
            frozen_gine_payloads=payloads,
            native_train_payload=train_payload,
            official_source_authority=preflight["official_runtime_source_authority"],
            require_isolated_imports=True,
            gspan_scratch_root=gspan_scratch_root,
        )
        result, evidence, branch_tree = run_resumed_target_branch(
            target_label=target,
            generator=generator,
            parents=parents,
            branch=branch,
            config=science_config,
        )
        if not result.records:
            raise TasteGlobalGCESmokeError(
                f"T8 recovered target-{target} still produced no native candidates"
            )
        receipt = {
            "schema_version": "tastemolnet_t8_single_branch_recovery_v1",
            "status": "PASS",
            "attempt_id": attempt_id,
            "recovery_source_attempt_id": recovery_source_attempt_id,
            "target_label": target,
            "source_label": SOURCE_LABEL,
            "state_root": str(state_root_path),
            "branch_root": str(state_root_path / f"target-{target}"),
            "source_selection_sha256": selection["selected_cohort_sha256"],
            "oracle_checkpoint_hash": scorer.checkpoint_id,
            "raw_generated_count": len(result.records),
            "branch_evidence": evidence,
            "other_target_rerun": False,
            "test_loaded": False,
            "calibration_loaded": False,
            "gnn_ablation_started": False,
        }
        receipt_leaf = state_root.write_new(
            "single_branch_recovery.json",
            _json_bytes(receipt),
        )
        branch_tree.revalidate()
        state_root.revalidate()
        return receipt
    finally:
        if receipt_leaf is not None:
            receipt_leaf.close()
        if branch_tree is not None:
            branch_tree.close()
        branch.close()
        state_root.close()


__all__ = [
    "BRANCH_SEAL_SCHEMA",
    "RERUN_SCHEMA",
    "SALVAGE_MARKER",
    "SALVAGE_SCHEMA",
    "ValidatedBranch",
    "T8BranchScientificFailure",
    "materialize_smoke_candidates",
    "run_salvage",
    "run_single_branch_recovery",
    "validate_source_branch",
    "write_single_branch_rerun_request",
]
