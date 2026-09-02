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
from typing import Any, Mapping, NoReturn, Sequence

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
    run_resumed_target_branch,
    select_bounded_sweet_parents,
    validate_candidates_with_original_gine,
    validate_science_summary,
)
from src.baselines.tastemolnet_multiclass_adapters import (
    merge_globalgce_target_branches,
)
from src.utils.retained_output_directory import (
    FreshOutputDirectory,
    RetainedOutputTree,
    prepare_terminal_output,
)


SALVAGE_SCHEMA = "tastemolnet_t8_branch_salvage_v1"
BRANCH_SEAL_SCHEMA = "tastemolnet_t8_read_only_branch_seal_v1"
RERUN_SCHEMA = "tastemolnet_t8_single_branch_rerun_request_v1"
RHS_CHEMISTRY_PREFLIGHT_SCHEMA = (
    "tastemolnet_t8_rhs_standalone_chemistry_preflight_v1"
)
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


class T8FinalizationError(TasteGlobalGCESmokeError):
    """Typed failure for the bounded T8 salvage/finalization path."""

    def __init__(
        self,
        *,
        code: str,
        field: str,
        expected: Any,
        actual: Any,
        source_manifest: str,
        source_artifact: str,
        stage: str,
    ) -> None:
        self.code = code
        self.field = field
        self.expected = expected
        self.actual = actual
        self.source_manifest = source_manifest
        self.source_artifact = source_artifact
        self.stage = stage
        super().__init__(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "source_manifest": self.source_manifest,
            "source_artifact": self.source_artifact,
            "stage": self.stage,
        }


def _raise_finalization_error(
    *,
    code: str,
    field: str,
    expected: Any,
    actual: Any,
    source_manifest: str,
    source_artifact: Path,
    stage: str,
) -> NoReturn:
    raise T8FinalizationError(
        code=code,
        field=field,
        expected=expected,
        actual=actual,
        source_manifest=source_manifest,
        source_artifact=str(source_artifact),
        stage=stage,
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


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def read_single_branch_rerun_target(path: Path) -> int:
    """Return the sole rerun target or fail with field-level evidence.

    This replaces the shell finalizer's untyped condition.  A malformed or
    multi-target receipt is a typed terminal blocker; it must never silently
    select one branch or trigger an unbounded two-branch replay.
    """

    source_artifact = Path(path)
    source_manifest = RERUN_SCHEMA
    stage = "T8_SINGLE_BRANCH_RERUN_SELECTION"
    try:
        payload = json.loads(source_artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _raise_finalization_error(
            code="T8_RERUN_RECEIPT_UNREADABLE",
            field="rerun_request",
            expected="one readable JSON object",
            actual=f"{type(exc).__name__}:{exc}",
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    if type(payload) is not dict:
        _raise_finalization_error(
            code="T8_RERUN_RECEIPT_NOT_OBJECT",
            field="rerun_request",
            expected="object",
            actual=type(payload).__name__,
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    checks = (
        (
            "schema_version",
            RERUN_SCHEMA,
            payload.get("schema_version"),
            "T8_RERUN_SCHEMA_MISMATCH",
        ),
        ("status", "RERUN_REQUIRED", payload.get("status"), "T8_RERUN_STATUS_MISMATCH"),
    )
    for field, expected, actual, code in checks:
        if actual != expected:
            _raise_finalization_error(
                code=code,
                field=field,
                expected=expected,
                actual=actual,
                source_manifest=source_manifest,
                source_artifact=source_artifact,
                stage=stage,
            )
    targets = payload.get("invalid_target_branches")
    if type(targets) is not list:
        _raise_finalization_error(
            code="T8_RERUN_TARGETS_NOT_LIST",
            field="invalid_target_branches",
            expected="list[int]",
            actual=type(targets).__name__,
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    if len(targets) != 1:
        _raise_finalization_error(
            code="T8_RERUN_NOT_SINGLE_BRANCH",
            field="invalid_target_branches",
            expected="exactly one target branch",
            actual=targets,
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    target = targets[0]
    if type(target) is not int or target not in TARGET_BRANCHES:
        _raise_finalization_error(
            code="T8_RERUN_TARGET_INVALID",
            field="invalid_target_branches[0]",
            expected=list(TARGET_BRANCHES),
            actual=target,
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    reasons = payload.get("reasons")
    if type(reasons) is not dict or type(reasons.get(str(target))) is not str:
        _raise_finalization_error(
            code="T8_RERUN_REASON_MISSING",
            field=f"reasons.{target}",
            expected="nonempty string",
            actual=None if type(reasons) is not dict else reasons.get(str(target)),
            source_manifest=source_manifest,
            source_artifact=source_artifact,
            stage=stage,
        )
    return target


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


def _hard_label(values: Any) -> int:
    return int(values.argmax(-1).item())


def _edge_position(left: int, right: int) -> int:
    high, low = max(int(left), int(right)), min(int(left), int(right))
    if high == low:
        raise TasteGlobalGCESmokeError("T8 RHS preflight rejects self-edge slots")
    return (high - 1) * high // 2 + low


def _raw_rule_identity(
    raw_rule: Mapping[str, Any],
    *,
    target: int,
    catalog_row_index: int,
) -> dict[str, Any]:
    source = (
        raw_rule.get("rule")
        if isinstance(raw_rule.get("rule"), Mapping)
        else raw_rule
    )
    return {
        "target_label": target,
        "catalog_row_index": catalog_row_index,
        "candidate_id": raw_rule.get("candidate_id"),
        "native_rule_index": source.get("native_rule_index"),
        "raw_rule_sha256": _canonical_sha256(raw_rule),
    }


def _rhs_rule_chemistry_evidence(
    raw_rule: Mapping[str, Any],
    *,
    target: int,
    catalog_row_index: int,
    source_artifact: Path,
) -> tuple[GlobalGCENativeRule | None, dict[str, Any]]:
    """Validate one standalone RHS without applying it to a parent graph."""

    identity = _raw_rule_identity(
        raw_rule,
        target=target,
        catalog_row_index=catalog_row_index,
    )
    evidence: dict[str, Any] = {
        **identity,
        "source_artifact": str(source_artifact),
        "status": "FAILED",
        "rhs_internal_bond_no_edge_consistent": False,
        "rhs_standalone_rdkit_sanitized": False,
        "errors": [],
    }
    try:
        rule = GlobalGCENativeRule.from_payload(raw_rule)
    except Exception as exc:
        evidence["errors"] = [
            {
                "code": "T8_RHS_RULE_DECODE_FAILED",
                "field": "rule",
                "expected": "one valid pinned native LHS/RHS rule",
                "actual": f"{type(exc).__name__}:{exc}",
            }
        ]
        return None, evidence

    active = tuple(
        index
        for index in range(rule.maximum_nodes)
        if _hard_label(rule.rhs_feature[index]) > 0
    )
    active_set = set(active)
    errors: list[dict[str, Any]] = []
    pair_evidence: list[dict[str, Any]] = []
    for left in range(rule.maximum_nodes):
        for right in range(left + 1, rule.maximum_nodes):
            edge_position = _edge_position(left, right)
            adjacency_present = bool(
                float(rule.rhs_adjacency[left, right].item()) > 0.5
            )
            bond_label = _hard_label(rule.rhs_edge_attr[edge_position])
            both_active = left in active_set and right in active_set
            pair = {
                "left": left,
                "right": right,
                "edge_position": edge_position,
                "both_nodes_active": both_active,
                "adjacency_present": adjacency_present,
                "bond_label": bond_label,
                "bond_name": (
                    rule.bond_names[bond_label]
                    if 0 <= bond_label < len(rule.bond_names)
                    else None
                ),
            }
            pair_evidence.append(pair)
            if not both_active and (adjacency_present or bond_label != 0):
                errors.append(
                    {
                        "code": "T8_RHS_INACTIVE_NODE_EDGE_STATE",
                        "field": f"rhs_pair[{left},{right}]",
                        "expected": {
                            "adjacency_present": False,
                            "bond_label": 0,
                        },
                        "actual": {
                            "adjacency_present": adjacency_present,
                            "bond_label": bond_label,
                        },
                    }
                )
            elif both_active and adjacency_present != (bond_label != 0):
                errors.append(
                    {
                        "code": "T8_RHS_BOND_NO_EDGE_MISMATCH",
                        "field": f"rhs_pair[{left},{right}]",
                        "expected": (
                            "adjacency present iff decoded bond label is not no_edge"
                        ),
                        "actual": {
                            "adjacency_present": adjacency_present,
                            "bond_label": bond_label,
                            "bond_name": pair["bond_name"],
                        },
                    }
                )
    evidence.update(
        {
            "rhs_active_node_indices": list(active),
            "rhs_active_node_count": len(active),
            "rhs_internal_pairs": pair_evidence,
            "rhs_internal_bond_no_edge_consistent": not errors,
        }
    )
    if not active:
        errors.append(
            {
                "code": "T8_RHS_EMPTY",
                "field": "rhs_feature",
                "expected": ">=1 active atom",
                "actual": 0,
            }
        )
    if errors:
        evidence["errors"] = errors
        return None, evidence

    try:
        from rdkit import Chem

        editable = Chem.RWMol()
        old_to_new: dict[int, int] = {}
        for old_index in active:
            atom_label = _hard_label(rule.rhs_feature[old_index])
            if not 0 < atom_label <= len(rule.atom_symbols):
                raise ValueError(f"unknown RHS atom label {atom_label}")
            old_to_new[old_index] = int(
                editable.AddAtom(Chem.Atom(rule.atom_symbols[atom_label - 1]))
            )
        bond_types = {
            "single": Chem.BondType.SINGLE,
            "double": Chem.BondType.DOUBLE,
            "triple": Chem.BondType.TRIPLE,
            "aromatic": Chem.BondType.AROMATIC,
        }
        for position, left in enumerate(active):
            for right in active[position + 1 :]:
                if float(rule.rhs_adjacency[left, right].item()) <= 0.5:
                    continue
                bond_label = _hard_label(
                    rule.rhs_edge_attr[_edge_position(left, right)]
                )
                bond_name = rule.bond_names[bond_label].strip().lower()
                if bond_name not in bond_types:
                    raise ValueError(f"unsupported RHS bond label {bond_name!r}")
                editable.AddBond(
                    old_to_new[left],
                    old_to_new[right],
                    bond_types[bond_name],
                )
        molecule = editable.GetMol()
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        if not canonical:
            raise ValueError("standalone sanitized RHS has empty canonical SMILES")
    except Exception as exc:
        evidence["errors"] = [
            {
                "code": "T8_RHS_STANDALONE_SANITIZATION_FAILED",
                "field": "rhs_standalone_molecule",
                "expected": "RDKit SanitizeMol PASS with nonempty canonical SMILES",
                "actual": f"{type(exc).__name__}:{exc}",
            }
        ]
        return None, evidence

    evidence.update(
        {
            "status": "PASS",
            "rhs_standalone_rdkit_sanitized": True,
            "rhs_standalone_canonical_smiles": canonical,
            "rhs_standalone_component_count": len(Chem.GetMolFrags(molecule)),
            "errors": [],
        }
    )
    return rule, evidence


def preflight_rhs_rule_catalogs(
    catalogs: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    source_artifacts: Mapping[int, Path],
    artifact_path: Path,
) -> tuple[dict[int, list[Mapping[str, Any]]], dict[str, Any]]:
    """Filter only independently sane RHS rules and persist field-level evidence."""

    if set(catalogs) != set(TARGET_BRANCHES) or set(source_artifacts) != set(
        TARGET_BRANCHES
    ):
        raise TasteGlobalGCESmokeError(
            "T8 RHS preflight requires both target catalogs and source artifacts"
        )
    approved: dict[int, list[Mapping[str, Any]]] = {0: [], 2: []}
    rule_evidence: dict[str, list[dict[str, Any]]] = {"0": [], "2": []}
    for target in TARGET_BRANCHES:
        for index, raw_rule in enumerate(catalogs[target]):
            rule, evidence = _rhs_rule_chemistry_evidence(
                raw_rule,
                target=target,
                catalog_row_index=index,
                source_artifact=source_artifacts[target],
            )
            rule_evidence[str(target)].append(evidence)
            if rule is not None:
                approved[target].append(raw_rule)
    approved_counts = {
        str(target): len(approved[target]) for target in TARGET_BRANCHES
    }
    rejected_counts = {
        str(target): len(catalogs[target]) - len(approved[target])
        for target in TARGET_BRANCHES
    }
    invalid_targets = [
        target for target in TARGET_BRANCHES if not approved[target]
    ]
    audit = {
        "schema_version": RHS_CHEMISTRY_PREFLIGHT_SCHEMA,
        "status": "BLOCKED" if invalid_targets else "PASS",
        "stage": "T8_RHS_STANDALONE_CHEMISTRY_PREFLIGHT",
        "source_artifacts_mutated": False,
        "native_rule_application_started": False,
        "gine_candidate_validation_started": False,
        "approved_rule_counts": approved_counts,
        "rejected_rule_counts": rejected_counts,
        "invalid_target_branches": invalid_targets,
        "rules": rule_evidence,
    }
    _atomic_json(artifact_path, audit)
    if invalid_targets:
        raise T8FinalizationError(
            code="T8_RHS_PREFLIGHT_NO_USABLE_RULES",
            field="branches.approved_rule_counts",
            expected={str(target): ">=1" for target in TARGET_BRANCHES},
            actual=approved_counts,
            source_manifest=RHS_CHEMISTRY_PREFLIGHT_SCHEMA,
            source_artifact=str(artifact_path),
            stage="T8_RHS_STANDALONE_CHEMISTRY_PREFLIGHT",
        )
    return approved, audit


def _merge_preflight_approved_rules(
    catalogs: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    checkpoint_id: str,
    preflight: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branches: dict[int, list[dict[str, Any]]] = {}
    branch_hashes: dict[int, set[str]] = {}
    for target in TARGET_BRANCHES:
        rows: list[dict[str, Any]] = []
        hashes: set[str] = set()
        for raw in catalogs[target]:
            action = _canonical_rule_action(raw)
            if action["rule_hash"] in hashes:
                raise TasteGlobalGCESmokeError(
                    "T8 RHS-approved catalog repeats one canonical rule"
                )
            hashes.add(action["rule_hash"])
            rows.append(
                {
                    **action,
                    "action_kind": "lhs_rhs_graph_transformation_rule",
                    "target_label": target,
                    "source_label": SOURCE_LABEL,
                    "data_split_used": "train",
                    "calibration_loaded": False,
                    "test_loaded": False,
                    "rf_oracle_used": False,
                    "oracle_backend": "gnn",
                    "oracle_checkpoint_hash": checkpoint_id,
                }
            )
        if not rows:
            raise TasteGlobalGCESmokeError(
                f"T8 target-{target} has no RHS-approved native rules"
            )
        branches[target] = rows
        branch_hashes[target] = hashes
    merged = merge_globalgce_target_branches(
        branches,
        oracle_checkpoint_hash=checkpoint_id,
    )
    if not merged or len({row["rule_hash"] for row in merged}) != len(merged):
        raise TasteGlobalGCESmokeError(
            "T8 RHS-approved canonical rule merge is empty or duplicated"
        )
    overlap = branch_hashes[0] & branch_hashes[2]
    return merged, {
        "merge_stage": (
            "after_rhs_standalone_chemistry_preflight_before_calibration"
        ),
        "dedup_identity": (
            "sha256(canonical_lhs,rhs,official_attachment_map,action_kind)"
        ),
        "target_0_rule_count": len(branches[0]),
        "target_2_rule_count": len(branches[2]),
        "premerge_rule_count": len(branches[0]) + len(branches[2]),
        "cross_branch_duplicate_count": len(overlap),
        "merged_unique_rule_count": len(merged),
        "hash_collision_or_action_mismatch": False,
        "canonical_dedup_complete": True,
        "rhs_chemistry_preflight_schema": RHS_CHEMISTRY_PREFLIGHT_SCHEMA,
        "rhs_chemistry_preflight_status": preflight.get("status"),
        "rhs_chemistry_approved_rule_counts": preflight.get(
            "approved_rule_counts"
        ),
        "rhs_chemistry_rejected_rule_counts": preflight.get(
            "rejected_rule_counts"
        ),
        "merged_rule_set_sha256": _canonical_sha256(
            [
                {
                    "rule_hash": row["rule_hash"],
                    "target_branches": row["target_branches"],
                }
                for row in merged
            ]
        ),
    }


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
            or not _is_sha256(oracle.get("identity_sha256"))
            or type(oracle.get("temperature_hex")) is not str
            or not _is_sha256(oracle.get("temperature_scaling_sha256"))
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
            "dataset": DATASET,
            "source_label": SOURCE_LABEL,
            "target_label": target,
            "num_classes": NUM_CLASSES,
            "generation_split": "train",
            "source_read_only": True,
            "active_writer_count": 0,
            "inventory_sha256": inventory["inventory_sha256"],
            "content_inventory": _content_inventory(inventory),
            "model_sha256": evidence["terminal_model_checkpoint_sha256"],
            "rules_sha256": evidence["terminal_rule_checkpoint_sha256"],
            "catalog_sha256": evidence["native_rule_catalog_sha256"],
            "oracle_checkpoint_hash": checkpoint_id,
            "oracle_identity_sha256": oracle["identity_sha256"],
            "temperature_hex": oracle["temperature_hex"],
            "temperature_scaling_sha256": oracle[
                "temperature_scaling_sha256"
            ],
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
    audit_documents: Mapping[str, Mapping[str, Any]],
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
    expected_audit_names = {
        "target0_adoption_receipt.json",
        "target2_adoption_receipt.json",
        "branch_inventory.json",
        "merged_rules.json",
        "canonical_dedup.json",
        "strict_flip_smoke.json",
        "terminal.json",
        "final_audit.json",
    }
    if set(audit_documents) != expected_audit_names:
        raise TasteGlobalGCESmokeError(
            "T8 salvage audit-document closure is incomplete"
        )
    for name in sorted(expected_audit_names):
        _atomic_json(state_root / name, audit_documents[name])
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
    rhs_preflight_path = rerun_request.with_name(
        "rhs-standalone-chemistry-preflight.json"
    )
    try:
        approved_catalogs, rhs_preflight = preflight_rhs_rule_catalogs(
            {target: branches[target].catalog for target in TARGET_BRANCHES},
            source_artifacts={
                target: branches[target].root / "native_rule_catalog.jsonl"
                for target in TARGET_BRANCHES
            },
            artifact_path=rhs_preflight_path,
        )
    except T8FinalizationError as exc:
        invalid_targets = json.loads(
            rhs_preflight_path.read_text(encoding="utf-8")
        ).get("invalid_target_branches", [])
        write_single_branch_rerun_request(
            rerun_request,
            {
                int(target): (
                    f"{exc.code}:field={exc.field}:"
                    f"evidence={rhs_preflight_path}"
                )
                for target in invalid_targets
            },
        )
        for branch in branches.values():
            branch.close()
        raise
    state_fd = -1
    state_tree: RetainedOutputTree | None = None
    prepared = None
    try:
        scorer = FrozenTasteGINEScorer(
            payloads,
            device=device,
            batch_size=TasteGlobalGCESmokeConfig().oracle_batch_size,
        )
        expected_temperature_hex = float(scorer.temperature).hex()
        if any(
            branches[target].seal["oracle_checkpoint_hash"] != checkpoint_id
            or branches[target].seal["temperature_hex"]
            != expected_temperature_hex
            or branches[target].seal["feature_schema_sha256"]
            != feature_schema_sha
            for target in TARGET_BRANCHES
        ):
            raise TasteGlobalGCESmokeError(
                "T8 salvage branches differ from the calibrated GINE identity"
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
        merged_rules, rule_merge = _merge_preflight_approved_rules(
            approved_catalogs,
            checkpoint_id=checkpoint_id,
            preflight=rhs_preflight,
        )
        if rule_merge.get("merged_unique_rule_count", 0) < 1:
            _raise_finalization_error(
                code="T8_SMOKE_INSUFFICIENT_UNIQUE_RULES",
                field="rule_merge.merged_unique_rule_count",
                expected=">=1",
                actual=rule_merge.get("merged_unique_rule_count"),
                source_manifest=SALVAGE_SCHEMA,
                source_artifact=state_root,
                stage="T8_CANONICAL_RULE_MERGE",
            )
        try:
            generated, materialization = materialize_smoke_candidates(
                approved_catalogs,
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
                # The authorized salvage smoke is untargeted after merging:
                # both target artifacts must be valid, but one real flip in
                # either destination branch closes the smoke science gate.
                minimum_strict_flips_per_branch=0,
                minimum_strict_flips_total=1,
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
        audit_documents: dict[str, Mapping[str, Any]] = {
            "target0_adoption_receipt.json": {
                **branches[0].seal,
                "adoption_status": "PASS",
                "source_artifact_mutated": False,
            },
            "target2_adoption_receipt.json": {
                **branches[2].seal,
                "adoption_status": "PASS",
                "source_artifact_mutated": False,
            },
            "branch_inventory.json": {
                "schema_version": "tastemolnet_t8_branch_inventory_v1",
                "status": "PASS",
                "rhs_chemistry_preflight": rhs_preflight,
                "branches": {
                    str(target): branches[target].seal
                    for target in TARGET_BRANCHES
                },
            },
            "merged_rules.json": {
                "schema_version": "tastemolnet_t8_merged_rules_v1",
                "status": "PASS",
                "merged_unique_rule_count": len(merged_rules),
                "rules": merged_rules,
            },
            "canonical_dedup.json": {
                "schema_version": "tastemolnet_t8_canonical_dedup_v1",
                "status": "PASS",
                "rule_merge": rule_merge,
                "candidate_merge": candidate_merge,
            },
            "strict_flip_smoke.json": {
                "schema_version": "tastemolnet_t8_strict_flip_smoke_v1",
                "status": "PASS",
                "minimum_required_total": 1,
                "minimum_required_per_branch": 0,
                "result": strict,
            },
            "terminal.json": {
                "schema_version": "tastemolnet_t8_salvage_terminal_v1",
                "state": "PASS",
                "dataset": DATASET,
                "method": METHOD,
                "source_label": SOURCE_LABEL,
                "target_branches": list(TARGET_BRANCHES),
                "merged_valid_unique_rules": len(merged_rules),
                "strict_flip_count": strict["strict_flip_count"],
                "test_loaded": False,
            },
            "final_audit.json": {
                "schema_version": "tastemolnet_t8_salvage_final_audit_v1",
                "passed": True,
                "same_calibrated_gine": True,
                "same_temperature": True,
                "same_dataset_split": True,
                "both_target_artifacts_verified": True,
                "canonical_merge_dedup_complete": True,
                "real_untargeted_strict_flip_observed": True,
                "source_artifacts_mutated": False,
                "test_loaded": False,
                "gnn_ablation_started": False,
                "rhs_standalone_chemistry_preflight": True,
                "rhs_preflight_artifact": str(rhs_preflight_path),
            },
        }
        state_fd, state_tree = _copy_validated_branches(
            branches,
            state_root=state_root,
            audit_documents=audit_documents,
        )
        state_inventory = state_tree.revalidate()
        salvage_config = science_config.to_dict()
        salvage_config["minimum_strict_flips_per_branch"] = 0
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
            "config": salvage_config,
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
            "rhs_standalone_chemistry_preflight": rhs_preflight,
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
        validate_science_summary(science, minimum_strict_flips_per_branch=0)
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
            "rhs_chemistry_preflight": rhs_preflight,
            "rhs_chemistry_preflight_artifact": str(rhs_preflight_path),
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
    "T8FinalizationError",
    "ValidatedBranch",
    "T8BranchScientificFailure",
    "materialize_smoke_candidates",
    "run_salvage",
    "run_single_branch_recovery",
    "read_single_branch_rerun_target",
    "validate_source_branch",
    "write_single_branch_rerun_request",
]
