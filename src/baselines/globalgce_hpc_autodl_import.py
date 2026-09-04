"""AutoDL-side import of the exact TasteMolNet GlobalGCE HPC bundle.

The HPC result is deliberately only a train-side gSpan intermediate.  This
module therefore verifies and imports transport/scientific evidence, but it
does not call a GNN, open calibration or test data, or publish a matrix cell.
Those operations belong to the separately sealed T13 successor.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from src.baselines import globalgce_hpc_exact as exact
from src.baselines.globalgce_hpc_hierarchical import (
    ARRAY_ADOPTION_SCHEMA,
    EVIDENCE_MANIFEST_SCHEMA,
    FINAL_VERIFICATION_SCHEMA,
    GROUP_PLAN_SCHEMA,
    PACKAGE_READY_SCHEMA,
)
from src.baselines.globalgce_hpc_storage_safe import (
    STORAGE_SAFE_RECEIPT_SCHEMA,
    stream_verify_storage_safe_bundle,
)


RELAY_READY_SCHEMA = "t8_hpc_package_ready_v1"
IMPORT_MANIFEST_SCHEMA = "t8_hpc_autodl_import_manifest_v1"
IMPORT_VERIFICATION_SCHEMA = "t8_hpc_autodl_import_verification_v1"
MINING_ADOPTION_SCHEMA = "globalgce_hpc_t8_mining_adoption_v1"
IMPORT_PASS = b"[T8_HPC_IMPORT_PASS]\n"
EXPECTED_DATASET = "tastemolnet"
EXPECTED_METHOD = "globalgce"
EXPECTED_ROUTE = "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD"
EXPECTED_SHARD_COUNT = 16
MAX_SMALL_MEMBER_BYTES = 64 * 1024 * 1024
COPY_BLOCK_BYTES = 4 * 1024 * 1024


class T8HPCAutoDLImportError(RuntimeError):
    """The relayed bundle is absent, mutable, or not the exact T8 result."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise T8HPCAutoDLImportError(f"reserved self-hash field: {field}")
    result[field] = _canonical_sha(result)
    return result


def _require_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    claimed = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not _is_sha256(claimed) or _canonical_sha(unsigned) != claimed:
        raise T8HPCAutoDLImportError(f"{label} self-hash mismatch")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _physical_dir(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise T8HPCAutoDLImportError(f"{label} must be an absolute physical directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise T8HPCAutoDLImportError(f"{label} is absent: {path}") from exc
    if not resolved.is_dir():
        raise T8HPCAutoDLImportError(f"{label} is not a directory: {resolved}")
    return resolved


def _fresh_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise T8HPCAutoDLImportError(f"{label} must be an absolute physical path")
    path = path.resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise T8HPCAutoDLImportError(f"{label} must be fresh: {path}")
    return path


def _physical_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise T8HPCAutoDLImportError(f"{label} is absent, empty, or indirect: {path}")
    return path


def _json(path: Path, *, label: str) -> dict[str, Any]:
    _physical_file(path, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise T8HPCAutoDLImportError(f"invalid {label}: {path}") from exc
    if type(value) is not dict:
        raise T8HPCAutoDLImportError(f"{label} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(dict(payload)) + b"\n")


def _dfs_graph(row: Mapping[str, Any]) -> Any:
    """Reconstruct the exact labelled undirected graph encoded by one DFS code."""

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - production dependency
        raise T8HPCAutoDLImportError("networkx is required for T13 adoption") from exc
    code = exact.dfs_code_from_json(row.get("dfs_code"))
    if row.get("dfs_code_sha256") != exact.dfs_code_sha256(code):
        raise T8HPCAutoDLImportError("selected DFS identity changed")
    graph = nx.Graph()
    labels: dict[int, Any] = {}
    for edge in code:
        left_label, edge_label, right_label = edge.vevlb
        for node, label in ((edge.frm, left_label), (edge.to, right_label)):
            # Official gSpan uses -1 for a label already fixed earlier in the
            # DFS code.  Any other value is an exact node label.
            if label == -1:
                if node not in labels:
                    raise T8HPCAutoDLImportError(
                        "DFS code uses a vacant label before defining the vertex"
                    )
            elif node in labels and labels[node] != label:
                raise T8HPCAutoDLImportError("DFS code changes a vertex label")
            elif label != -1:
                labels[node] = label
                graph.add_node(node, label=label)
        if edge.frm == edge.to or graph.has_edge(edge.frm, edge.to):
            raise T8HPCAutoDLImportError("DFS code contains a loop/duplicate edge")
        graph.add_edge(edge.frm, edge.to, label=edge_label)
    if sorted(graph.nodes) != list(range(len(graph.nodes))):
        raise T8HPCAutoDLImportError("DFS graph vertex IDs are not consecutive")
    if len(labels) != len(graph.nodes) or not nx.is_connected(graph):
        raise T8HPCAutoDLImportError("DFS graph is unlabeled or disconnected")
    return graph


def _graph_semantics(graph: Any) -> dict[str, Any]:
    return {
        "nodes": [
            {"node": node, "label": attributes.get("label")}
            for node, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {"left": left, "right": right, "label": attributes.get("label")}
            for left, right, attributes in graph.edges(data=True)
        ],
    }


def _writable_fds_below(root: Path, *, proc_root: str | Path = "/proc") -> list[dict[str, Any]]:
    proc = Path(proc_root)
    if not proc.is_dir():
        return []
    holders: list[dict[str, Any]] = []
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = Path(os.readlink(descriptor)).resolve(strict=False)
                target.relative_to(root)
                flags_line = next(
                    line
                    for line in (process / "fdinfo" / descriptor.name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
            except (OSError, StopIteration, ValueError):
                continue
            if flags & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}:
                holders.append(
                    {
                        "pid": int(process.name),
                        "fd": int(descriptor.name),
                        "path": str(target),
                    }
                )
    return holders


def _small_tar_json_members(
    archive: Path,
    *,
    required: Sequence[str],
    first_member: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    with tarfile.open(archive, "r|gz") as tar:
        for index, member in enumerate(tar):
            if index == 0 and member.name != first_member:
                raise T8HPCAutoDLImportError(f"{first_member} is not first")
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.name in seen
            ):
                raise T8HPCAutoDLImportError("archive contains an unsafe member")
            seen.add(member.name)
            stream = tar.extractfile(member)
            if stream is None:
                raise T8HPCAutoDLImportError(f"archive member is unreadable: {member.name}")
            digest = hashlib.sha256()
            size = 0
            capture = member.name in required
            chunks: list[bytes] = []
            for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
                size += len(block)
                digest.update(block)
                if capture:
                    if size > MAX_SMALL_MEMBER_BYTES:
                        raise T8HPCAutoDLImportError(
                            f"required metadata member is too large: {member.name}"
                        )
                    chunks.append(block)
            if size != member.size:
                raise T8HPCAutoDLImportError(f"archive member size changed: {member.name}")
            inventory.append(
                {"name": member.name, "bytes": size, "sha256": digest.hexdigest()}
            )
            if capture:
                try:
                    value = json.loads(b"".join(chunks))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise T8HPCAutoDLImportError(
                        f"archive metadata is invalid JSON: {member.name}"
                    ) from exc
                if type(value) is not dict:
                    raise T8HPCAutoDLImportError(
                        f"archive metadata is not an object: {member.name}"
                    )
                payloads[member.name] = value
    missing = sorted(set(required) - set(payloads))
    if missing:
        raise T8HPCAutoDLImportError(f"archive metadata is incomplete: {missing}")
    return payloads, inventory


def _validate_hierarchical_evidence(
    root: Path,
    *,
    result_receipt: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = root / "hierarchical_evidence_manifest.json"
    archive_path = root / "t8_hierarchical_evidence.tar.gz"
    receipt = _json(receipt_path, label="hierarchical evidence receipt")
    _require_self_hash(receipt, "receipt_sha256", "hierarchical evidence receipt")
    if (
        receipt.get("schema_version") != EVIDENCE_MANIFEST_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("matrix_write_enabled") is not False
        or receipt.get("archive_name") != archive_path.name
        or receipt.get("archive_sha256") != _sha256(archive_path)
        or receipt.get("archive_sha256") != ready.get("evidence_archive_sha256")
        or receipt.get("receipt_sha256") != ready.get("evidence_receipt_sha256")
    ):
        raise T8HPCAutoDLImportError("hierarchical evidence receipt binding failed")
    required = (
        "EVIDENCE_MANIFEST.json",
        "array_adoption_manifest.json",
        "group_plan.json",
        "final/hierarchical_verification.json",
    )
    documents, inventory = _small_tar_json_members(
        archive_path, required=required, first_member="EVIDENCE_MANIFEST.json"
    )
    evidence = documents["EVIDENCE_MANIFEST.json"]
    adoption = documents["array_adoption_manifest.json"]
    plan = documents["group_plan.json"]
    final = documents["final/hierarchical_verification.json"]
    _require_self_hash(evidence, "evidence_manifest_sha256", "evidence manifest")
    _require_self_hash(adoption, "array_adoption_sha256", "array adoption")
    _require_self_hash(plan, "group_plan_sha256", "group plan")
    _require_self_hash(final, "verification_sha256", "hierarchical verification")
    members = {row["name"]: row for row in inventory[1:]}
    declared = evidence.get("files")
    if type(declared) is not list or declared != [
        members[name] for name in sorted(members)
    ]:
        raise T8HPCAutoDLImportError("hierarchical evidence member inventory changed")
    shards = adoption.get("shards")
    if (
        evidence.get("schema_version") != EVIDENCE_MANIFEST_SCHEMA
        or evidence.get("status") != "PASS"
        or evidence.get("successful_shards_rerun") is not False
        or evidence.get("matrix_write_enabled") is not False
        or evidence.get("storage_safe_receipt_sha256")
        != result_receipt.get("receipt_sha256")
        or evidence.get("merge_result_sha256") != ready.get("merge_result_sha256")
        or receipt.get("evidence_manifest_sha256")
        != evidence.get("evidence_manifest_sha256")
        or adoption.get("schema_version") != ARRAY_ADOPTION_SCHEMA
        or adoption.get("status") != "PASS"
        or adoption.get("shard_count") != EXPECTED_SHARD_COUNT
        or adoption.get("passed_shard_count") != EXPECTED_SHARD_COUNT
        or adoption.get("successful_shards_rerun") is not False
        or adoption.get("matrix_write_enabled") is not False
        or type(shards) is not list
        or [row.get("shard_index") for row in shards]
        != list(range(EXPECTED_SHARD_COUNT))
        or any(not _is_sha256(row.get("result_sha256")) for row in shards)
        or len({row.get("result_sha256") for row in shards})
        != EXPECTED_SHARD_COUNT
        or plan.get("schema_version") != GROUP_PLAN_SCHEMA
        or plan.get("status") != "PASS"
        or plan.get("partition_disjoint") is not True
        or plan.get("partition_complete") is not True
        or plan.get("official_global_order_preserved") is not True
        or plan.get("successful_shards_rerun") is not False
        or plan.get("matrix_write_enabled") is not False
        or final.get("schema_version") != FINAL_VERIFICATION_SCHEMA
        or final.get("status") != "PASS"
        or final.get("event_identity_unique") is not True
        or final.get("pattern_identity_unique") is not True
        or final.get("matrix_write_enabled") is not False
        or final.get("group_plan_sha256") != plan.get("group_plan_sha256")
        or final.get("merge_result_sha256") != ready.get("merge_result_sha256")
    ):
        raise T8HPCAutoDLImportError("hierarchical exactness/array binding failed")
    return {
        "evidence_archive_sha256": receipt["archive_sha256"],
        "evidence_receipt_sha256": receipt["receipt_sha256"],
        "evidence_manifest_sha256": evidence["evidence_manifest_sha256"],
        "array_adoption_sha256": adoption["array_adoption_sha256"],
        "group_plan_sha256": plan["group_plan_sha256"],
        "final_verification_sha256": final["verification_sha256"],
        "shard_count": EXPECTED_SHARD_COUNT,
        "shard_result_sha256s": [str(row["result_sha256"]) for row in shards],
        "successful_shards_rerun": False,
    }


def validate_relayed_hpc_package(
    package_root: str | Path,
    *,
    expected_execution_commit: str,
    expected_scientific_input_sha256: str,
    expected_partition_manifest_sha256: str,
    expected_official_globalgce_commit: str = exact.OFFICIAL_GLOBALGCE_COMMIT,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Deeply verify a relayed package without extracting its large streams."""

    if not _is_commit(expected_execution_commit):
        raise T8HPCAutoDLImportError("expected execution commit is malformed")
    for value, label in (
        (expected_scientific_input_sha256, "scientific input SHA"),
        (expected_partition_manifest_sha256, "partition manifest SHA"),
    ):
        if not _is_sha256(value):
            raise T8HPCAutoDLImportError(f"expected {label} is malformed")
    if not _is_commit(expected_official_globalgce_commit):
        raise T8HPCAutoDLImportError("expected official GlobalGCE commit is malformed")
    root = _physical_dir(package_root, label="relay package root")
    required_names = {
        "HPC_PACKAGE_READY.json",
        "HIERARCHICAL_PACKAGE_READY.json",
        "hierarchical_evidence_manifest.json",
        "result_manifest.json",
        "t8_exact_result_bundle.tar.gz",
        "t8_hierarchical_evidence.tar.gz",
    }
    observed_names = {path.name for path in root.iterdir()}
    if not required_names <= observed_names or any(
        name.endswith((".partial", ".incomplete")) for name in observed_names
    ):
        raise T8HPCAutoDLImportError("relay package is incomplete")
    holders = _writable_fds_below(root, proc_root=proc_root)
    if holders:
        raise T8HPCAutoDLImportError(f"relay package still has writable FDs: {holders}")

    relay = _json(root / "HPC_PACKAGE_READY.json", label="relay ready marker")
    ready = _json(
        root / "HIERARCHICAL_PACKAGE_READY.json",
        label="hierarchical package marker",
    )
    _require_self_hash(ready, "package_ready_sha256", "hierarchical package marker")
    receipt_path = root / "result_manifest.json"
    receipt = _json(receipt_path, label="result receipt")
    _require_self_hash(receipt, "receipt_sha256", "result receipt")
    archive = _physical_file(
        root / "t8_exact_result_bundle.tar.gz", label="result archive"
    )
    archive_sha = _sha256(archive)
    if (
        relay.get("schema_version") != RELAY_READY_SCHEMA
        or relay.get("state") != "HPC_PACKAGE_READY"
        or relay.get("matrix_write_enabled") is not False
        or relay.get("archive_sha256") != archive_sha
        or relay.get("hierarchical_evidence_sha256")
        != ready.get("evidence_archive_sha256")
        or ready.get("schema_version") != PACKAGE_READY_SCHEMA
        or ready.get("status") != "PASS"
        or ready.get("matrix_write_enabled") is not False
        or ready.get("result_archive_sha256") != archive_sha
        or ready.get("result_receipt_sha256") != receipt.get("receipt_sha256")
        or receipt.get("schema_version") != STORAGE_SAFE_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("matrix_write_enabled") is not False
    ):
        raise T8HPCAutoDLImportError("outer relay/package binding failed")
    stream = stream_verify_storage_safe_bundle(archive, receipt_path=receipt_path)
    if (
        stream.get("status") != "PASS"
        or stream.get("receipt_verified") is not True
        or stream.get("matrix_write_enabled") is not False
        or stream.get("scientific_input_sha256")
        != expected_scientific_input_sha256
        or stream.get("partition_manifest_sha256")
        != expected_partition_manifest_sha256
        or stream.get("merge_result_sha256") != ready.get("merge_result_sha256")
    ):
        raise T8HPCAutoDLImportError("streaming result verification identity failed")
    result_documents, _result_inventory = _small_tar_json_members(
        archive,
        required=(
            "RESULT_MANIFEST.json",
            "partition_manifest.json",
            "merge/stable_top_k.json",
        ),
        first_member="RESULT_MANIFEST.json",
    )
    inner = result_documents["RESULT_MANIFEST.json"]
    partition = result_documents["partition_manifest.json"]
    stable_top_k = result_documents["merge/stable_top_k.json"]
    _require_self_hash(inner, "bundle_content_sha256", "inner result manifest")
    _require_self_hash(partition, "manifest_sha256", "partition manifest")
    provenance = partition.get("provenance")
    official = partition.get("official_gspan")
    proof = partition.get("completeness_proof")
    configuration = partition.get("configuration")
    if type(provenance) is not dict or type(official) is not dict or type(proof) is not dict:
        raise T8HPCAutoDLImportError("partition provenance/completeness is malformed")
    _require_self_hash(provenance, "provenance_sha256", "partition provenance")
    _require_self_hash(proof, "proof_sha256", "partition completeness proof")
    selected = stable_top_k.get("selected")
    if (
        partition.get("scope") != "FULL_ROOT_UNIVERSE"
        or partition.get("manifest_sha256") != expected_partition_manifest_sha256
        or partition.get("scientific_input_sha256")
        != expected_scientific_input_sha256
        or partition.get("matrix_write_enabled") is not False
        or provenance.get("execution_commit") != expected_execution_commit
        or provenance.get("route_kind") != EXPECTED_ROUTE
        or provenance.get("dataset") != EXPECTED_DATASET
        or provenance.get("method") != EXPECTED_METHOD
        or provenance.get("split_scope") != "train_only"
        or provenance.get("source_label") != 1
        or provenance.get("target_branches") != [0, 2]
        or provenance.get("calibration_loaded") is not False
        or provenance.get("test_loaded") is not False
        or provenance.get("matrix_write_enabled") is not False
        or official.get("commit") != expected_official_globalgce_commit
        or proof.get("disjoint") is not True
        or proof.get("complete") is not True
        or proof.get("partition_count") != len(partition.get("partitions") or [])
        or type(configuration) is not dict
        or configuration.get("min_support") != 2
        or configuration.get("min_vertices") != 3
        or configuration.get("max_vertices") != 20
        or configuration.get("top_k") != 20
        or stable_top_k.get("schema_version")
        != "globalgce_hpc_exact_stable_top_k_v1"
        or stable_top_k.get("top_k") != 20
        or type(selected) is not list
        or stable_top_k.get("selected_count") != len(selected)
        or stable_top_k.get("selected_sha256") != _canonical_sha(selected)
    ):
        raise T8HPCAutoDLImportError("scientific route/partition/top-K binding failed")
    hierarchy = _validate_hierarchical_evidence(
        root, result_receipt=receipt, ready=ready
    )
    if hierarchy["shard_count"] != EXPECTED_SHARD_COUNT:
        raise T8HPCAutoDLImportError("array does not contain all 16 exact shards")
    return _self_hashed(
        {
            "schema_version": IMPORT_VERIFICATION_SCHEMA,
            "status": "PASS",
            "package_root": str(root),
            "archive_sha256": archive_sha,
            "result_receipt_sha256": receipt["receipt_sha256"],
            "bundle_content_sha256": inner["bundle_content_sha256"],
            "merge_result_sha256": stream["merge_result_sha256"],
            "scientific_input_sha256": stream["scientific_input_sha256"],
            "partition_manifest_sha256": stream["partition_manifest_sha256"],
            "execution_commit": provenance["execution_commit"],
            "source_commit": provenance["source_commit"],
            "official_globalgce_commit": official["commit"],
            "split_scope": "train_only",
            "source_label": 1,
            "target_branches": [0, 2],
            "event_count": stream["event_count"],
            "pattern_count": stream["pattern_count"],
            "rejection_count": stream["rejection_count"],
            "selected_pattern_count": len(selected),
            "selected_patterns_sha256": stable_top_k["selected_sha256"],
            "hierarchy": hierarchy,
            "no_active_writer": True,
            "hpc_gine_inference_used": False,
            "hpc_calibration_or_test_used": False,
            "matrix_write_enabled": False,
        },
        "verification_sha256",
    )


def import_relayed_hpc_package(
    package_root: str | Path,
    output_root: str | Path,
    *,
    expected_execution_commit: str,
    expected_scientific_input_sha256: str,
    expected_partition_manifest_sha256: str,
    expected_official_globalgce_commit: str = exact.OFFICIAL_GLOBALGCE_COMMIT,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Publish a compact fresh AutoDL import after complete verification."""

    source = _physical_dir(package_root, label="relay package root")
    destination = _fresh_path(output_root, label="AutoDL import root")
    verification = validate_relayed_hpc_package(
        source,
        expected_execution_commit=expected_execution_commit,
        expected_scientific_input_sha256=expected_scientific_input_sha256,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_official_globalgce_commit=expected_official_globalgce_commit,
        proc_root=proc_root,
    )
    documents, _inventory = _small_tar_json_members(
        source / "t8_exact_result_bundle.tar.gz",
        required=("merge/stable_top_k.json",),
        first_member="RESULT_MANIFEST.json",
    )
    selected = documents["merge/stable_top_k.json"]
    staging = destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.incomplete"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(mode=0o700)
    try:
        selected_path = staging / "selected_patterns.json"
        _atomic_json(selected_path, selected)
        selected_rows = selected.get("selected")
        if type(selected_rows) is not list:
            raise T8HPCAutoDLImportError("selected-pattern collection is malformed")
        graphs = [_dfs_graph(row) for row in selected_rows]
        supports = [int(row["support"]) for row in selected_rows]
        if supports != sorted(supports, reverse=True):
            raise T8HPCAutoDLImportError("selected patterns are not support ordered")
        adopted_payload = {
            "schema_version": MINING_ADOPTION_SCHEMA,
            "graphs": graphs,
            "supports": supports,
            "selected_patterns_sha256": selected["selected_sha256"],
        }
        adopted_bytes = pickle.dumps(adopted_payload, protocol=5)
        adopted_path = staging / "selected_top20.pkl"
        _atomic_bytes(adopted_path, adopted_bytes)
        semantic_sha = _canonical_sha(
            [
                {
                    "rank": rank,
                    "support": supports[rank - 1],
                    "graph": _graph_semantics(graph),
                }
                for rank, graph in enumerate(graphs, start=1)
            ]
        )
        adoption_proof = _self_hashed(
            {
                "schema_version": MINING_ADOPTION_SCHEMA,
                "status": "PASS",
                "proof": "EXACT_HPC_FULL_ROOT_UNIVERSE_STREAM_VERIFIED",
                "source_import_manifest": "import_manifest.json",
                "selected_patterns": {
                    "path": selected_path.name,
                    "bytes": selected_path.stat().st_size,
                    "sha256": _sha256(selected_path),
                    "selected_sha256": selected["selected_sha256"],
                },
                "selected_top20": {
                    "path": adopted_path.name,
                    "bytes": adopted_path.stat().st_size,
                    "sha256": _sha256(adopted_path),
                },
                "selected_semantic_identity_sha256": semantic_sha,
                "selected_count": len(graphs),
                "top_k": int(selected["top_k"]),
                "min_freq": 2,
                "seed": 7,
                "ordering": "SUPPORT_DESC_OFFICIAL_PREORDER_ASC",
                "scientific_input_sha256": verification[
                    "scientific_input_sha256"
                ],
                "partition_manifest_sha256": verification[
                    "partition_manifest_sha256"
                ],
                "merge_result_sha256": verification["merge_result_sha256"],
                "official_globalgce_commit": verification[
                    "official_globalgce_commit"
                ],
                "source_scope": {
                    "dataset": EXPECTED_DATASET,
                    "method": EXPECTED_METHOD,
                    "split": "train",
                    "source_label": 1,
                    "target_branches": [0, 2],
                    "full_root_universe": True,
                },
                "hpc_gine_inference_used": False,
                "hpc_calibration_or_test_used": False,
                "matrix_write_enabled": False,
            },
            "adoption_sha256",
        )
        _atomic_json(staging / "adoption_proof.json", adoption_proof)
        manifest = _self_hashed(
            {
                "schema_version": IMPORT_MANIFEST_SCHEMA,
                "status": "PASS",
                "state": "IMPORTED_TRAIN_SIDE_GSPAN_PENDING_AUTODL_T13",
                "imported_at": _utc_now(),
                "source_package_root": str(source),
                "source_archive_sha256": verification["archive_sha256"],
                "source_verification_sha256": verification["verification_sha256"],
                "execution_commit": verification["execution_commit"],
                "source_commit": verification["source_commit"],
                "official_globalgce_commit": verification[
                    "official_globalgce_commit"
                ],
                "scientific_input_sha256": verification[
                    "scientific_input_sha256"
                ],
                "partition_manifest_sha256": verification[
                    "partition_manifest_sha256"
                ],
                "merge_result_sha256": verification["merge_result_sha256"],
                "selected_patterns": {
                    "path": "selected_patterns.json",
                    "bytes": selected_path.stat().st_size,
                    "sha256": _sha256(selected_path),
                    "selected_count": selected["selected_count"],
                    "selected_sha256": selected["selected_sha256"],
                },
                "mining_adoption": {
                    "proof": "adoption_proof.json",
                    "proof_sha256": adoption_proof["adoption_sha256"],
                    "selected_top20": adopted_path.name,
                    "selected_top20_sha256": _sha256(adopted_path),
                    "selected_count": len(graphs),
                },
                "all_16_shards_verified": True,
                "partition_complete": True,
                "partition_disjoint": True,
                "train_only": True,
                "source_label": 1,
                "target_branches": [0, 2],
                "rhs_chemistry_pending_autodl": True,
                "gine_inference_pending_autodl": True,
                "strict_flip_pending_autodl": True,
                "calibration_test_export_pending_autodl": True,
                "zero_patterns_is_not_a_terminal_cell_pass": True,
                "hpc_gine_inference_used": False,
                "hpc_calibration_or_test_used": False,
                "matrix_write_enabled": False,
            },
            "import_manifest_sha256",
        )
        _atomic_json(staging / "import_manifest.json", manifest)
        _atomic_json(staging / "verification.json", verification)
        _atomic_bytes(staging / "HPC_IMPORT_PASS", IMPORT_PASS)
        directory = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if destination.exists():
            raise T8HPCAutoDLImportError("AutoDL import root appeared during publication")
        os.rename(staging, destination)
        parent = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_imported_hpc_result(destination)


def validate_imported_hpc_result(output_root: str | Path) -> dict[str, Any]:
    """Reopen the compact import without upgrading it to a T13 cell PASS."""

    root = _physical_dir(output_root, label="AutoDL import root")
    marker = _physical_file(root / "HPC_IMPORT_PASS", label="HPC import marker")
    if marker.read_bytes() != IMPORT_PASS:
        raise T8HPCAutoDLImportError("HPC import marker changed")
    manifest = _json(root / "import_manifest.json", label="import manifest")
    verification = _json(root / "verification.json", label="import verification")
    _require_self_hash(manifest, "import_manifest_sha256", "import manifest")
    _require_self_hash(verification, "verification_sha256", "import verification")
    selected_path = root / "selected_patterns.json"
    selected = _json(selected_path, label="selected patterns")
    selected_identity = manifest.get("selected_patterns")
    adoption_identity = manifest.get("mining_adoption")
    if (
        manifest.get("schema_version") != IMPORT_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("state")
        != "IMPORTED_TRAIN_SIDE_GSPAN_PENDING_AUTODL_T13"
        or manifest.get("source_verification_sha256")
        != verification.get("verification_sha256")
        or manifest.get("all_16_shards_verified") is not True
        or manifest.get("partition_complete") is not True
        or manifest.get("partition_disjoint") is not True
        or manifest.get("train_only") is not True
        or manifest.get("source_label") != 1
        or manifest.get("target_branches") != [0, 2]
        or manifest.get("rhs_chemistry_pending_autodl") is not True
        or manifest.get("gine_inference_pending_autodl") is not True
        or manifest.get("strict_flip_pending_autodl") is not True
        or manifest.get("calibration_test_export_pending_autodl") is not True
        or manifest.get("zero_patterns_is_not_a_terminal_cell_pass") is not True
        or manifest.get("hpc_gine_inference_used") is not False
        or manifest.get("hpc_calibration_or_test_used") is not False
        or manifest.get("matrix_write_enabled") is not False
        or type(selected_identity) is not dict
        or selected_identity.get("path") != selected_path.name
        or selected_identity.get("bytes") != selected_path.stat().st_size
        or selected_identity.get("sha256") != _sha256(selected_path)
        or selected_identity.get("selected_count") != selected.get("selected_count")
        or selected_identity.get("selected_sha256") != selected.get("selected_sha256")
        or selected.get("selected_sha256") != _canonical_sha(selected.get("selected"))
        or type(adoption_identity) is not dict
        or adoption_identity.get("proof") != "adoption_proof.json"
        or adoption_identity.get("selected_top20") != "selected_top20.pkl"
        or adoption_identity.get("selected_count") != selected.get("selected_count")
    ):
        raise T8HPCAutoDLImportError("compact AutoDL import closure failed")
    proof = validate_hpc_mining_adoption_proof(root / "adoption_proof.json")
    if (
        proof["adoption_sha256"] != adoption_identity.get("proof_sha256")
        or proof["selected_top20"]["sha256"]
        != adoption_identity.get("selected_top20_sha256")
    ):
        raise T8HPCAutoDLImportError("import/mining-adoption binding failed")
    return manifest


def validate_hpc_mining_adoption_proof(proof_path: str | Path) -> dict[str, Any]:
    """Reopen the compact top-K proof consumed by official GlobalGCE training."""

    proof_file = Path(proof_path).expanduser().resolve(strict=True)
    root = proof_file.parent
    proof = _json(proof_file, label="HPC mining adoption proof")
    _require_self_hash(proof, "adoption_sha256", "HPC mining adoption proof")
    selected_identity = proof.get("selected_patterns")
    pickle_identity = proof.get("selected_top20")
    source_scope = proof.get("source_scope")
    if (
        proof.get("schema_version") != MINING_ADOPTION_SCHEMA
        or proof.get("status") != "PASS"
        or proof.get("proof") != "EXACT_HPC_FULL_ROOT_UNIVERSE_STREAM_VERIFIED"
        or type(selected_identity) is not dict
        or type(pickle_identity) is not dict
        or type(source_scope) is not dict
        or proof.get("top_k") != 20
        or proof.get("min_freq") != 2
        or proof.get("seed") != 7
        or proof.get("ordering") != "SUPPORT_DESC_OFFICIAL_PREORDER_ASC"
        or proof.get("official_globalgce_commit") != exact.OFFICIAL_GLOBALGCE_COMMIT
        or source_scope
        != {
            "dataset": EXPECTED_DATASET,
            "method": EXPECTED_METHOD,
            "split": "train",
            "source_label": 1,
            "target_branches": [0, 2],
            "full_root_universe": True,
        }
        or proof.get("hpc_gine_inference_used") is not False
        or proof.get("hpc_calibration_or_test_used") is not False
        or proof.get("matrix_write_enabled") is not False
    ):
        raise T8HPCAutoDLImportError("HPC mining adoption contract changed")
    selected_path = root / str(selected_identity.get("path") or "")
    pickle_path = root / str(pickle_identity.get("path") or "")
    for path, identity, label in (
        (selected_path, selected_identity, "selected patterns"),
        (pickle_path, pickle_identity, "selected graph pickle"),
    ):
        _physical_file(path, label=label)
        if (
            identity.get("bytes") != path.stat().st_size
            or identity.get("sha256") != _sha256(path)
        ):
            raise T8HPCAutoDLImportError(f"{label} identity changed")
    selected = _json(selected_path, label="selected patterns")
    try:
        adopted = pickle.loads(pickle_path.read_bytes())
    except Exception as exc:
        raise T8HPCAutoDLImportError("selected graph pickle is unreadable") from exc
    if (
        type(adopted) is not dict
        or adopted.get("schema_version") != MINING_ADOPTION_SCHEMA
        or adopted.get("selected_patterns_sha256") != selected.get("selected_sha256")
        or type(adopted.get("graphs")) is not list
        or type(adopted.get("supports")) is not list
        or len(adopted["graphs"]) != proof.get("selected_count")
        or len(adopted["supports"]) != proof.get("selected_count")
        or not 0 <= int(proof.get("selected_count", -1)) <= int(proof["top_k"])
        or [int(value) for value in adopted["supports"]]
        != sorted((int(value) for value in adopted["supports"]), reverse=True)
        or _canonical_sha(
            [
                {
                    "rank": rank,
                    "support": int(adopted["supports"][rank - 1]),
                    "graph": _graph_semantics(graph),
                }
                for rank, graph in enumerate(adopted["graphs"], start=1)
            ]
        )
        != proof.get("selected_semantic_identity_sha256")
    ):
        raise T8HPCAutoDLImportError("selected graph payload changed")
    return proof


def load_hpc_adopted_top_k(
    proof_path: str | Path,
    *,
    validated_identity: Mapping[str, Any] | None = None,
) -> tuple[list[Any], list[int]]:
    proof = validate_hpc_mining_adoption_proof(proof_path)
    if validated_identity is not None and dict(validated_identity) != proof:
        raise T8HPCAutoDLImportError("prevalidated HPC adoption identity changed")
    payload = pickle.loads((Path(proof_path).resolve().parent / "selected_top20.pkl").read_bytes())
    return list(payload["graphs"]), [int(value) for value in payload["supports"]]


__all__ = [
    "EXPECTED_SHARD_COUNT",
    "IMPORT_MANIFEST_SCHEMA",
    "IMPORT_PASS",
    "IMPORT_VERIFICATION_SCHEMA",
    "MINING_ADOPTION_SCHEMA",
    "RELAY_READY_SCHEMA",
    "T8HPCAutoDLImportError",
    "import_relayed_hpc_package",
    "load_hpc_adopted_top_k",
    "validate_hpc_mining_adoption_proof",
    "validate_imported_hpc_result",
    "validate_relayed_hpc_package",
]
