"""Typed TasteMolNet research/reporting and no-redistribution policy.

The upstream repository has no explicit data-licence statement.  This module
keeps that observation separate from the project owner's scoped authorization
for private research computation and aggregate paper reporting.  It never
converts that authorization into an upstream licence conclusion or a dataset
redistribution permission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping


POLICY_V1_SCHEMA = "tastemolnet_research_reporting_policy_v1"
POLICY_SCHEMA = "tastemolnet_research_reporting_policy_v2"
POLICY_V1_ID = "tastemolnet-research-reporting-no-redistribution-20260825"
POLICY_ID = "tastemolnet-research-reporting-no-redistribution-v2-20260825"
PENDING_STATE = "PENDING_ROOT_ACTIVATION"
ACTIVE_STATE = "ACTIVE_SCOPED_AUTHORIZATION"
PENDING_STATUS = "POLICY_READY_EXECUTION_DISABLED"
ACTIVE_STATUS = "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
UPSTREAM_TERMS_STATUS = "NOT_EXPLICITLY_STATED"
UPSTREAM_REPOSITORY = "https://github.com/MujeebOnawole/Taste_Prediction_RGCN"
UPSTREAM_COMMIT = "16af8ead8a17b6bd3941d9eb5879c5be75c14114"
UPSTREAM_DATA_FILE = "processed_data/taste_scaffold_split.csv"
SOURCE_CSV_SHA256 = "b7308b3277fd07ed6af4b861c0d2ce2d843f92cc81a9e5e4efd65cf4040a291b"
ACTIVE_AUDIT_MARKER = "TASTE_RESEARCH_AND_PAPER_REPORTING_AUTHORIZED"
POLICY_V2_AUDIT_MARKER = "TASTE_RESEARCH_POLICY_V2_PASS"
NO_REDISTRIBUTION_MARKER = "TASTE_NO_DATA_REDISTRIBUTION_GUARD_PASS"
PENDING_AUDIT_MARKER = "TASTEMOLNET_POLICY_READY_EXECUTION_DISABLED"

_HEX_64 = frozenset("0123456789abcdef")


class TasteResearchPolicyError(RuntimeError):
    """The scoped policy or its execution boundary failed closed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TasteResearchPolicyError(f"{field} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise TasteResearchPolicyError(
            f"{field} keys changed: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _hex(value: Any, *, field: str) -> str:
    if type(value) is not str:
        raise TasteResearchPolicyError(f"{field} must be one native JSON string")
    result = value.lower()
    if len(result) != 64 or any(character not in _HEX_64 for character in result):
        raise TasteResearchPolicyError(f"{field} must be one lowercase SHA-256")
    return result


def _exact_native_equal(actual: Any, expected: Any, *, field: str) -> None:
    """Compare JSON-like authority without Python's bool/int coercions."""

    if type(actual) is not type(expected):
        raise TasteResearchPolicyError(f"{field} changed native JSON type")
    if type(expected) is dict:
        if any(type(key) is not str for key in actual):
            raise TasteResearchPolicyError(f"{field} contains a non-string key")
        if set(actual) != set(expected):
            raise TasteResearchPolicyError(f"{field} keys changed")
        for key in expected:
            _exact_native_equal(actual[key], expected[key], field=f"{field}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise TasteResearchPolicyError(f"{field} length changed")
        for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
            _exact_native_equal(observed, wanted, field=f"{field}[{index}]")
        return
    if type(expected) is float and not math.isfinite(actual):
        raise TasteResearchPolicyError(f"{field} must be finite")
    if actual != expected:
        raise TasteResearchPolicyError(f"{field} changed value")


def _native_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TasteResearchPolicyError(f"{field} must be one native JSON integer")
    return value


def _read_physical(path: str | Path) -> tuple[Path, bytes]:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = source.resolve(strict=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteResearchPolicyError("policy must be one physical regular file")
        parts: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(source, follow_symlinks=False)
        identity = lambda value: (  # noqa: E731 - compact immutable projection.
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise TasteResearchPolicyError("policy changed while it was read")
        return source.resolve(strict=True), b"".join(parts)
    finally:
        os.close(descriptor)


def _validate(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema_version")
    version = payload.get("policy_version")
    if type(version) is not int or version not in {1, 2}:
        raise TasteResearchPolicyError("policy_version must be native JSON integer 1 or 2")
    if (schema, version) not in {(POLICY_V1_SCHEMA, 1), (POLICY_SCHEMA, 2)}:
        raise TasteResearchPolicyError("policy schema/version changed")
    top_level_keys = {
        "schema_version",
        "policy_id",
        "effective_date",
        "authorization_basis",
        "authorization_state",
        "dataset",
        "policy_version",
        "authorization_source",
        "authorization_date",
        "research_compute_allowed",
        "paper_result_reporting_allowed",
        "upstream_license_status",
        "upstream_license_claimed_resolved",
        "raw_data_redistribution_allowed",
        "cleaned_dataset_redistribution_allowed",
        "full_smiles_label_table_release_allowed",
        "reconstructable_dataset_artifact_allowed",
        "preprocessing_code_release_allowed",
        "configuration_release_allowed",
        "aggregated_metrics_release_allowed",
        "figure_release_allowed",
        "trained_model_release_allowed",
        "required_citations",
        "fixed_upstream_commit",
        "fixed_csv_sha256",
        "dataset_identity",
        "permissions",
        "data_handling",
        "execution",
        "publication",
        "hash_contract",
    }
    if version == 2:
        top_level_keys.add("data_redistribution_allowed")
    _exact_keys(
        payload,
        top_level_keys,
        field="policy",
    )
    state = str(payload.get("authorization_state") or "")
    if state not in {PENDING_STATE, ACTIVE_STATE}:
        raise TasteResearchPolicyError("authorization_state is invalid")
    expected_basis = (
        "forwarded_user_instruction_pending_root_activation"
        if state == PENDING_STATE
        else "explicit_user_instruction"
    )
    if payload.get("authorization_basis") != expected_basis:
        raise TasteResearchPolicyError("authorization basis/state mismatch")

    active = state == ACTIVE_STATE
    expected_top_level = {
        "policy_id": POLICY_ID if version == 2 else POLICY_V1_ID,
        "effective_date": "2026-08-25",
        "dataset": "tastemolnet",
        "policy_version": version,
        "authorization_source": (
            "user_project_owner_instruction"
            if active
            else "PENDING_ROOT_ACTIVATION"
        ),
        "authorization_date": "2026-08-25",
        "research_compute_allowed": active,
        "paper_result_reporting_allowed": active,
        "upstream_license_status": UPSTREAM_TERMS_STATUS,
        "upstream_license_claimed_resolved": False,
        "raw_data_redistribution_allowed": False,
        "cleaned_dataset_redistribution_allowed": False,
        "full_smiles_label_table_release_allowed": False,
        "reconstructable_dataset_artifact_allowed": False,
        "preprocessing_code_release_allowed": True,
        "configuration_release_allowed": True,
        "aggregated_metrics_release_allowed": active,
        "figure_release_allowed": active,
        "trained_model_release_allowed": "review_required" if active else False,
        "required_citations": [
            "TasteMolNet paper",
            "https://github.com/MujeebOnawole/Taste_Prediction_RGCN/tree/"
            + UPSTREAM_COMMIT
        ],
        "fixed_upstream_commit": UPSTREAM_COMMIT,
        "fixed_csv_sha256": SOURCE_CSV_SHA256,
    }
    if version == 2:
        expected_top_level["data_redistribution_allowed"] = False
    boolean_fields = {
        "research_compute_allowed",
        "paper_result_reporting_allowed",
        "upstream_license_claimed_resolved",
        "raw_data_redistribution_allowed",
        "cleaned_dataset_redistribution_allowed",
        "full_smiles_label_table_release_allowed",
        "reconstructable_dataset_artifact_allowed",
        "preprocessing_code_release_allowed",
        "configuration_release_allowed",
        "aggregated_metrics_release_allowed",
        "figure_release_allowed",
    }
    if version == 2:
        boolean_fields.add("data_redistribution_allowed")
    if any(type(payload.get(key)) is not bool for key in boolean_fields):
        raise TasteResearchPolicyError("top-level policy booleans changed type")
    for key, expected in expected_top_level.items():
        _exact_native_equal(payload.get(key), expected, field=f"policy.{key}")

    dataset = _mapping(payload.get("dataset_identity"), field="dataset_identity")
    expected_dataset = {
        "id": "tastemolnet",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_data_file": UPSTREAM_DATA_FILE,
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "prepared_output_manifest_sha256": dataset.get(
            "prepared_output_manifest_sha256"
        ),
        "split_manifest_sha256": dataset.get("split_manifest_sha256"),
        "upstream_terms_status": UPSTREAM_TERMS_STATUS,
        "prepared_rows": 13421,
        "split_rows": {
            "train": 9437,
            "validation": 1328,
            "calibration": 1328,
            "test": 1328,
        },
    }
    _exact_native_equal(dataset, expected_dataset, field="dataset_identity")
    _hex(
        dataset.get("prepared_output_manifest_sha256"),
        field="dataset_identity.prepared_output_manifest_sha256",
    )
    _hex(
        dataset.get("split_manifest_sha256"),
        field="dataset_identity.split_manifest_sha256",
    )

    permissions = _mapping(payload.get("permissions"), field="permissions")
    expected_permission = "PENDING_ROOT_ACTIVATION" if state == PENDING_STATE else "ALLOWED"
    expected_aggregate = (
        "ALLOWED_ONLY_AFTER_ACTIVATION_AND_PUBLIC_ARTIFACT_AUDIT"
        if state == PENDING_STATE
        else "ALLOWED_AFTER_PUBLIC_ARTIFACT_AUDIT"
    )
    _exact_native_equal(permissions, {
        "research_execution": expected_permission,
        "paper_reporting": expected_permission,
        "dataset_redistribution": "FORBIDDEN",
        "raw_dataset_publication": "FORBIDDEN",
        "processed_dataset_publication": "FORBIDDEN",
        "molecule_level_publication": "FORBIDDEN",
        "aggregate_publication": expected_aggregate,
        "model_artifact_publication": "INTERNAL_ONLY",
    }, field="permissions")

    handling = _mapping(payload.get("data_handling"), field="data_handling")
    _exact_native_equal(handling, {
        "reuse_existing_prepared_data_only": True,
        "reuse_existing_graph_cache_only": True,
        "data_preparation_allowed": False,
        "network_download_allowed": False,
        "source_copy_allowed": False,
        "public_artifact_audit_required": True,
    }, field="data_handling")

    execution = _mapping(payload.get("execution"), field="execution")
    _exact_keys(
        execution,
        {
            "phase",
            "platform",
            "hpc_execution_allowed",
            "run_tastemolnet",
            "gpu_index",
            "gpu_lock_mode",
            "fresh_output_required",
            "classifier",
            "split_access",
        },
        field="execution",
    )
    expected_run = 0 if state == PENDING_STATE else 1
    run = execution.get("run_tastemolnet")
    if (
        type(run) is not int
        or run != expected_run
        or execution.get("phase") != "TASTEMOLNET_GINE_FULL_RESEARCH_V1"
        or execution.get("platform") != "autodl"
        or execution.get("hpc_execution_allowed") is not False
        or type(execution.get("gpu_index")) is not int
        or execution.get("gpu_index") != (1 if version == 2 else 2)
        or execution.get("gpu_lock_mode") != "exclusive"
        or execution.get("fresh_output_required") is not True
    ):
        raise TasteResearchPolicyError("execution boundary changed")
    classifier = _mapping(execution.get("classifier"), field="classifier")
    _exact_native_equal(classifier, {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "num_classes": 3,
        "source_label": 1,
    }, field="execution.classifier")
    _exact_native_equal(
        _mapping(execution.get("split_access"), field="split_access"),
        {
        "train_loaded": True,
        "validation_loaded": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "test_metadata_hash_only": True,
        },
        field="execution.split_access",
    )

    publication = _mapping(payload.get("publication"), field="publication")
    _exact_keys(
        publication,
        {"allowed_categories", "forbidden_categories", "marker"},
        field="publication",
    )
    _exact_native_equal(publication.get("allowed_categories"), [
        "aggregate_metrics",
        "aggregate_tables",
        "aggregate_figures",
        "method_configuration",
        "provenance_hashes",
    ], field="publication.allowed_categories")
    _exact_native_equal(publication.get("forbidden_categories"), [
        "source_csv",
        "prepared_split_rows",
        "graph_cache_payloads",
        "molecule_identifiers",
        "smiles_or_molecular_records",
        "per_example_predictions",
    ], field="publication.forbidden_categories")
    if publication.get("marker") != (
        "TASTEMOLNET_PUBLIC_ARTIFACT_NO_DATA_REDISTRIBUTION_AUDIT"
    ):
        raise TasteResearchPolicyError("publication marker changed")
    hash_contract = _mapping(payload.get("hash_contract"), field="hash_contract")
    _exact_native_equal(hash_contract, {
        "policy_file_hash": "sha256_raw_bytes",
        "policy_semantic_hash": "canonical_json_sha256_v1",
        "runtime_must_bind": [
            "policy_file_sha256",
            "policy_canonical_sha256",
            "provenance_manifest_sha256",
            "prepared_output_manifest_sha256",
            "split_manifest_sha256",
            "graph_cache_manifest_sha256",
        ],
    }, field="hash_contract")


@dataclass(frozen=True, slots=True)
class TasteResearchPolicy:
    path: Path
    file_sha256: str
    canonical_sha256: str
    payload: Mapping[str, Any]

    @property
    def authorization_state(self) -> str:
        return str(self.payload["authorization_state"])

    @property
    def active(self) -> bool:
        return self.authorization_state == ACTIVE_STATE

    @property
    def version(self) -> int:
        return int(self.payload["policy_version"])

    @property
    def status(self) -> str:
        """The unresolved upstream-terms status; never an authorization claim."""

        return UPSTREAM_TERMS_STATUS

    @property
    def authorization_status(self) -> str:
        return ACTIVE_STATUS if self.active else PENDING_STATUS

    def evidence(self) -> dict[str, Any]:
        evidence = {
            "schema_version": str(self.payload["schema_version"]),
            "policy_id": str(self.payload["policy_id"]),
            "policy_path": str(self.path),
            "policy_file_sha256": self.file_sha256,
            "policy_canonical_sha256": self.canonical_sha256,
            "authorization_state": self.authorization_state,
            "status": self.status,
            "authorization_status": self.authorization_status,
            "upstream_terms_status": UPSTREAM_TERMS_STATUS,
            "research_execution_allowed": self.active,
            "paper_reporting_allowed": self.active,
            "dataset_redistribution_allowed": False,
            "dataset_redistributed": False,
            "upstream_license_not_explicit": True,
            "paper_results_reporting_allowed_by_project_policy": self.active,
            "license_conclusion": "NOT_GRANTED_OR_INFERRED",
        }
        if self.version == 2:
            evidence.update(
                {
                    "policy_version": 2,
                    "research_compute_allowed": self.active,
                    "paper_result_reporting_allowed": self.active,
                    "data_redistribution_allowed": False,
                    "main_route_state": (
                        "READY_FOR_MAIN_ROUTE" if self.active else PENDING_STATUS
                    ),
                }
            )
        return evidence

    def require_active(self) -> None:
        if not self.active:
            raise TasteResearchPolicyError("TASTEMOLNET_POLICY_NOT_ACTIVATED")

    def require_main_route(self) -> None:
        self.require_active()
        if self.version != 2 or self.payload.get("schema_version") != POLICY_SCHEMA:
            raise TasteResearchPolicyError("TASTEMOLNET_POLICY_V2_REQUIRED")


def _read_json_physical(path: Path) -> dict[str, Any]:
    source, data = _read_physical(path)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteResearchPolicyError(f"invalid JSON authority: {source}") from exc
    if not isinstance(payload, dict):
        raise TasteResearchPolicyError(f"JSON authority must be one object: {source}")
    return payload


def _physical_directory(path: str | Path, *, field: str) -> Path:
    value = Path(path).expanduser().resolve(strict=True)
    info = os.lstat(value)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise TasteResearchPolicyError(f"{field} must be one physical directory")
    return value


def _inventory_files(root: Path, *, excluded: set[str]) -> set[str]:
    result: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in directories:
            info = os.lstat(base / name)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteResearchPolicyError("private authority contains symlink/special directory")
        for name in files:
            path = base / name
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TasteResearchPolicyError("private authority contains symlink/special file")
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                result.add(relative)
    return result


@dataclass(frozen=True, slots=True)
class TasteLocalDataAuthority:
    prepared_root: Path
    graph_cache_root: Path
    provenance_manifest_sha256: str
    prepared_output_manifest_sha256: str
    split_manifest_sha256: str
    graph_cache_manifest_sha256: str
    source_csv_sha256: str
    prepared_rows: int
    split_rows: Mapping[str, int]
    graph_cache_rows: int

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": "tastemolnet_existing_private_data_authority_v1",
            "prepared_root": str(self.prepared_root),
            "graph_cache_root": str(self.graph_cache_root),
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "prepared_output_manifest_sha256": self.prepared_output_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "graph_cache_manifest_sha256": self.graph_cache_manifest_sha256,
            "source_csv_sha256": self.source_csv_sha256,
            "prepared_rows": self.prepared_rows,
            "split_rows": dict(self.split_rows),
            "graph_cache_rows": self.graph_cache_rows,
            "data_reprepared": False,
            "graph_cache_rebuilt": False,
            "cache_payloads_deserialized_by_audit": False,
            "test_rows_deserialized_by_audit": False,
        }


def validate_tastemolnet_local_authority(
    policy: TasteResearchPolicy,
    *,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
) -> TasteLocalDataAuthority:
    """Reopen existing prepared/cache artifacts without preparing or loading rows."""

    prepared = _physical_directory(prepared_root, field="prepared_root")
    cache = _physical_directory(graph_cache_root, field="graph_cache_root")
    policy_dataset = _mapping(
        policy.payload.get("dataset_identity"), field="dataset_identity"
    )
    provenance_path = prepared / "provenance_manifest.json"
    output_manifest_path = prepared / "output_manifest.json"
    split_manifest_path = prepared / "splits/split_manifest.json"
    split_statistics_path = prepared / "splits/split_statistics.json"
    marker_path = prepared / "LICENSE_REVIEW_REQUIRED"
    for required in (
        provenance_path,
        output_manifest_path,
        split_manifest_path,
        split_statistics_path,
        marker_path,
    ):
        _read_physical(required)

    provenance = _read_json_physical(provenance_path)
    if (
        provenance.get("dataset") != "tastemolnet"
        or provenance.get("upstream_commit") != UPSTREAM_COMMIT
        or provenance.get("source_csv_sha256") != SOURCE_CSV_SHA256
        or provenance.get("download_performed") is not False
        or provenance.get("raw_data_copied_into_output") is not False
        or provenance.get("raw_data_commit_allowed") is not False
        or provenance.get("license_status") != "LICENSE_REVIEW_REQUIRED"
    ):
        raise TasteResearchPolicyError("prepared provenance authority changed")

    output_manifest = _read_json_physical(output_manifest_path)
    files = output_manifest.get("files")
    if (
        type(output_manifest.get("schema_version")) is not int
        or output_manifest.get("schema_version") != 1
        or not isinstance(files, Mapping)
        or output_manifest.get("manifest_digest") != stable_json_sha256(files)
    ):
        raise TasteResearchPolicyError("prepared output manifest changed")
    if sha256_file(output_manifest_path) != policy_dataset.get(
        "prepared_output_manifest_sha256"
    ):
        raise TasteResearchPolicyError("prepared output manifest authority changed")
    current = _inventory_files(prepared, excluded={"output_manifest.json"})
    if set(files) != current:
        raise TasteResearchPolicyError("prepared output inventory changed")
    for relative, identity in files.items():
        if (
            not isinstance(identity, Mapping)
            or _hex(identity.get("sha256"), field=f"prepared:{relative}")
            != sha256_file(prepared / relative)
            or type(identity.get("bytes")) is not int
            or identity.get("bytes") != (prepared / relative).stat().st_size
        ):
            raise TasteResearchPolicyError(f"prepared file identity changed: {relative}")

    split_manifest = _read_json_physical(split_manifest_path)
    if sha256_file(split_manifest_path) != policy_dataset.get("split_manifest_sha256"):
        raise TasteResearchPolicyError("prepared split manifest authority changed")
    if (
        split_manifest.get("dataset") != "tastemolnet"
        or type(split_manifest.get("num_classes")) is not int
        or split_manifest.get("num_classes") != 3
        or type(split_manifest.get("source_label")) is not int
        or split_manifest.get("source_label") != 1
        or split_manifest.get("label_map")
        != {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
        or split_manifest.get("scaffold_overlap_gate_passed") is not True
        or split_manifest.get("all_classes_present_per_split") is not True
    ):
        raise TasteResearchPolicyError("prepared split contract changed")
    statistics = _read_json_physical(split_statistics_path)
    prepared_rows = _native_int(
        statistics.get("total_clean_rows"), field="split_statistics.total_clean_rows"
    )
    raw_splits = statistics.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise TasteResearchPolicyError("prepared split statistics changed")
    split_rows = {}
    for split in ("train", "validation", "calibration", "test"):
        row = _mapping(raw_splits.get(split), field=f"split:{split}")
        split_rows[split] = _native_int(row.get("rows"), field=f"split:{split}.rows")
    if prepared_rows != policy_dataset.get("prepared_rows") or split_rows != policy_dataset.get("split_rows"):
        raise TasteResearchPolicyError("prepared row counts changed")

    cache_manifest_path = cache / "manifest.json"
    cache_manifest = _read_json_physical(cache_manifest_path)
    splits = cache_manifest.get("splits")
    if (
        cache_manifest.get("schema_version") != "molecular_graph_cache_manifest_v1"
        or cache_manifest.get("dataset") != "tastemolnet"
        or type(cache_manifest.get("num_classes")) is not int
        or cache_manifest.get("num_classes") != 3
        or cache_manifest.get("split_order")
        != ["train", "validation", "calibration", "test"]
        or not isinstance(splits, Mapping)
        or set(splits) != {"train", "validation", "calibration", "test"}
    ):
        raise TasteResearchPolicyError("graph-cache manifest changed")
    cache_files = {"manifest.json"}
    graph_cache_rows = 0
    for split in ("train", "validation", "calibration", "test"):
        identity = _mapping(splits[split], field=f"cache:{split}")
        filename = str(identity.get("cache_file") or "")
        if filename != f"{split}.pt":
            raise TasteResearchPolicyError(f"graph-cache {split} filename changed")
        expected_hash = _hex(identity.get("cache_sha256"), field=f"cache:{split}:sha256")
        if sha256_file(cache / filename) != expected_hash:
            raise TasteResearchPolicyError(f"graph-cache {split} hash changed")
        if (
            type(identity.get("num_classes")) is not int
            or identity.get("num_classes") != 3
            or identity.get("safe_load_verified") is not True
        ):
            raise TasteResearchPolicyError(f"graph-cache {split} semantic contract changed")
        if identity.get("source_csv_sha256") != sha256_file(prepared / "splits" / f"{split}.csv"):
            raise TasteResearchPolicyError(f"graph-cache {split} source hash changed")
        count = _native_int(identity.get("graph_count"), field=f"cache:{split}.graph_count")
        if count != split_rows[split]:
            raise TasteResearchPolicyError(f"graph-cache {split} row count changed")
        graph_cache_rows += count
        cache_files.add(filename)
    if _inventory_files(cache, excluded=set()) != cache_files:
        raise TasteResearchPolicyError("graph-cache physical inventory changed")
    if (
        type(cache_manifest.get("total_graph_count")) is not int
        or graph_cache_rows != prepared_rows
        or cache_manifest.get("total_graph_count") != prepared_rows
    ):
        raise TasteResearchPolicyError("graph-cache total row count changed")
    return TasteLocalDataAuthority(
        prepared_root=prepared,
        graph_cache_root=cache,
        provenance_manifest_sha256=sha256_file(provenance_path),
        prepared_output_manifest_sha256=sha256_file(output_manifest_path),
        split_manifest_sha256=sha256_file(split_manifest_path),
        graph_cache_manifest_sha256=sha256_file(cache_manifest_path),
        source_csv_sha256=SOURCE_CSV_SHA256,
        prepared_rows=prepared_rows,
        split_rows=split_rows,
        graph_cache_rows=graph_cache_rows,
    )


@dataclass(frozen=True, slots=True)
class TastePolicyReceipt:
    path: Path
    sha256: str
    payload: Mapping[str, Any]


def validate_tastemolnet_policy_receipt(
    path: str | Path,
    *,
    policy: TasteResearchPolicy,
    authority: TasteLocalDataAuthority,
    require_active: bool,
    require_policy_version: int | None = None,
) -> TastePolicyReceipt:
    """Reopen a fresh policy-audit receipt and all of its typed authority."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_physical(source)
    expected_keys = {
        "schema_version",
        "created_at",
        "dataset",
        "status",
        "authorization_state",
        "authorization_status",
        "policy",
        "private_data_authority",
        "run_tastemolnet",
        "heavy_route_authorized",
        "paper_reporting_authorized",
        "dataset_redistribution_authorized",
        "upstream_terms_status",
        "license_conclusion",
        "hpc_execution_authorized",
        "data_reprepared",
        "graph_cache_rebuilt",
        "terminal_marker",
        "no_redistribution_marker",
    }
    _exact_keys(payload, expected_keys, field="policy_receipt")
    expected_marker = (
        POLICY_V2_AUDIT_MARKER
        if policy.active and policy.version == 2
        else ACTIVE_AUDIT_MARKER
        if policy.active
        else PENDING_AUDIT_MARKER
    )
    expected_run = 1 if policy.active else 0
    expected_receipt_schema = (
        "tastemolnet_research_reporting_policy_receipt_v2"
        if policy.version == 2
        else "tastemolnet_research_reporting_policy_receipt_v1"
    )
    if (
        payload.get("schema_version") != expected_receipt_schema
        or not isinstance(payload.get("created_at"), str)
        or not payload.get("created_at")
        or payload.get("dataset") != "tastemolnet"
        or payload.get("status") != policy.status
        or payload.get("authorization_state") != policy.authorization_state
        or payload.get("authorization_status") != policy.authorization_status
        or type(payload.get("run_tastemolnet")) is not int
        or payload.get("run_tastemolnet") != expected_run
        or payload.get("heavy_route_authorized") is not policy.active
        or payload.get("paper_reporting_authorized") is not policy.active
        or payload.get("dataset_redistribution_authorized") is not False
        or payload.get("upstream_terms_status") != UPSTREAM_TERMS_STATUS
        or payload.get("license_conclusion") != "NOT_GRANTED_OR_INFERRED"
        or payload.get("hpc_execution_authorized") is not False
        or payload.get("data_reprepared") is not False
        or payload.get("graph_cache_rebuilt") is not False
        or payload.get("terminal_marker") != expected_marker
        or payload.get("no_redistribution_marker")
        != (NO_REDISTRIBUTION_MARKER if policy.active else None)
    ):
        raise TasteResearchPolicyError("typed Taste policy receipt changed")
    _exact_native_equal(payload.get("policy"), policy.evidence(), field="policy_receipt.policy")
    _exact_native_equal(
        payload.get("private_data_authority"),
        authority.evidence(),
        field="policy_receipt.private_data_authority",
    )
    marker_path = source.parent / expected_marker
    marker_source, marker_data = _read_physical(marker_path)
    if marker_data != (expected_marker + "\n").encode("utf-8"):
        raise TasteResearchPolicyError("Taste policy terminal marker changed")
    expected_inventory = {
        source.name,
        "tastemolnet_policy_audit.md",
        marker_source.name,
    }
    if policy.active:
        guard_source, guard_data = _read_physical(
            source.parent / NO_REDISTRIBUTION_MARKER
        )
        if guard_data != (NO_REDISTRIBUTION_MARKER + "\n").encode("utf-8"):
            raise TasteResearchPolicyError("Taste no-redistribution marker changed")
        expected_inventory.add(guard_source.name)
    if _inventory_files(source.parent, excluded=set()) != expected_inventory:
        raise TasteResearchPolicyError("Taste policy audit output inventory changed")
    if require_active:
        policy.require_active()
    if require_policy_version is not None and policy.version != require_policy_version:
        raise TasteResearchPolicyError(
            f"Taste policy version {require_policy_version} is required"
        )
    return TastePolicyReceipt(path=source, sha256=sha256_file(source), payload=payload)


def parse_tastemolnet_research_policy(
    data: bytes,
    *,
    source: str | Path,
    expected_file_sha256: str | None = None,
) -> TasteResearchPolicy:
    """Validate policy bytes already read through a caller-held authority.

    Downstream stages use this entry point after opening the exact tracked
    policy root-to-leaf with retained descriptors.  Keeping parsing separate
    from pathname lookup prevents a second, path-based reopen from weakening
    that stronger authority.
    """

    source_path = Path(source)
    file_sha256 = hashlib.sha256(data).hexdigest()
    if expected_file_sha256 is not None and file_sha256 != _hex(
        expected_file_sha256, field="expected_file_sha256"
    ):
        raise TasteResearchPolicyError("policy file SHA-256 changed")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(data.decode("utf-8"))
    except Exception as exc:
        raise TasteResearchPolicyError("invalid Taste policy YAML") from exc
    if not isinstance(payload, dict):
        raise TasteResearchPolicyError("Taste policy must contain one mapping")
    _validate(payload)
    return TasteResearchPolicy(
        path=source_path,
        file_sha256=file_sha256,
        canonical_sha256=stable_json_sha256(payload),
        payload=payload,
    )


def load_tastemolnet_research_policy(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> TasteResearchPolicy:
    source, data = _read_physical(path)
    return parse_tastemolnet_research_policy(
        data,
        source=source,
        expected_file_sha256=expected_file_sha256,
    )


__all__ = [
    "ACTIVE_STATE",
    "ACTIVE_AUDIT_MARKER",
    "ACTIVE_STATUS",
    "PENDING_STATE",
    "PENDING_AUDIT_MARKER",
    "PENDING_STATUS",
    "POLICY_V2_AUDIT_MARKER",
    "POLICY_SCHEMA",
    "POLICY_V1_SCHEMA",
    "NO_REDISTRIBUTION_MARKER",
    "SOURCE_CSV_SHA256",
    "UPSTREAM_COMMIT",
    "UPSTREAM_TERMS_STATUS",
    "TasteResearchPolicy",
    "TasteResearchPolicyError",
    "TasteLocalDataAuthority",
    "TastePolicyReceipt",
    "load_tastemolnet_research_policy",
    "parse_tastemolnet_research_policy",
    "validate_tastemolnet_local_authority",
    "validate_tastemolnet_policy_receipt",
    "sha256_file",
    "stable_json_sha256",
]
