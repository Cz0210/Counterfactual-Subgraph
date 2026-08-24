"""Build the fresh terminal-only AIDS ComRecGC exact-route v5 controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import (
    load_controller_manifest,
)
from scripts.autodl.verify_aids_comrecgc_v5_process_set import verify_process_set
from src.baselines.comrecgc.external_memory_recourse import (
    PAIR_STORE_SCHEMA,
    _file_stat_identity,
    _find_writable_process_references,
    _validate_pair_store_manifest,
)
from src.utils.aids_comrecgc_v5_snapshot import (
    EXPECTED_CANDIDATE_COUNT as SNAPSHOT_EXPECTED_CANDIDATE_COUNT,
    EXPECTED_PARENT_COUNT as SNAPSHOT_EXPECTED_PARENT_COUNT,
    EXPECTED_ROWS as SNAPSHOT_EXPECTED_ROWS,
    EXPECTED_VECTOR_DIM as SNAPSHOT_EXPECTED_VECTOR_DIM,
    MIN_FREE_AFTER_BYTES as SNAPSHOT_MIN_FREE_AFTER_BYTES,
)
from src.utils.aids_comrecgc_v5_snapshot_adoption import (
    SOURCE_CONTROLLER_ID as SNAPSHOT_OWNER_CONTROLLER_ID,
    SOURCE_SNAPSHOT_TASK_ID as SNAPSHOT_OWNER_TASK_ID,
)
from src.utils import autodl_aids_comrecgc_repair_v3 as v3
from src.utils.autodl_aids_comrecgc_repair_v4 import (
    CONTROLLER_ID as V4_CONTROLLER_ID,
    STANDARDIZATION_TASK_ID as V4_TASK_ID,
)
from src.utils.autodl_four_by_four_am_repair import STANDARDIZED_REQUIRED_FILES
from src.utils.autodl_four_by_four_repair import (
    RepairManifestError,
    sha256_file,
)


SPEC_SCHEMA = "aids_comrecgc_exact_route_v5_snapshot_adopt_v1_spec_v1"
CONTROLLER_ID = (
    "four_methods_four_datasets_aids_comrecgc_exact_route_v5_snapshot_adopt_v1"
)
TASK_ID = "aids_comrecgc_standardized_exact_route_v5_snapshot_adopt_v1"
SELECTOR_TASK_ID = "aids_comrecgc_exact_route_v5_snapshot_adopt_v1_selector_freeze"
SNAPSHOT_TASK_ID = "aids_comrecgc_pair_store_snapshot_adoption_v5_v1"
PAIR_SEMANTICS_BENCHMARK_TASK_ID = (
    "aids_comrecgc_pair_semantics_greed_benchmark_v1"
)
PAIR_SEMANTICS_TASK_ID = "aids_comrecgc_pair_semantics_greed_full_v1"
CLOSE_PAIR_VIEW_TASK_ID = "aids_comrecgc_theta_close_view_v1"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"
REVIEWED_SOURCE_CORE_COMMIT = "645c6e51b7abcdc5dd4a9e0a1226d71d020880da"
INTEGRATED_REVIEWED_CORE_COMMIT = "8c371b1c8ee1d8188555581c4f8e8b6060ae42eb"
# Kept as the public release-pin name for downstream audit consumers.  The
# independently reviewed source commit was cherry-picked into the integration
# lineage; therefore ancestry must use the integrated equivalent, not the
# non-ancestor source SHA.
REVIEWED_CORE_COMMIT = INTEGRATED_REVIEWED_CORE_COMMIT
REVIEWED_CORE_FILE_IDENTITIES = {
    "src/baselines/comrecgc/external_memory_dbscan.py": {
        "git_blob": "57e9f7d6c5463a29f30fc08a9792ea2e65a30754",
        "sha256": "383e391a1bb0bb9b4356f8b976b67e1098da2d3af6a90e15dc5a858b36ed0d5a",
    },
    "tests/baselines/comrecgc/test_external_memory_dbscan.py": {
        "git_blob": "d71a532b035f181805ef803a695d30cd31d7d6cc",
        "sha256": "d556a2eea865b180b0dff91bc4085f0a085e35e4786cedc0fcf3410b457d1f77",
    },
}
ROUTE_RELEASE_COMMIT = "a6cdfd51d19af7f390d1cbc9d00827c97baee150"
SNAPSHOT_RELEASE_COMMIT = "8b99498a0c1beab11a6844ddf5f6f9d7c2c4458f"
SNAPSHOT_RELEASE_FILE_IDENTITIES = {
    "src/utils/aids_comrecgc_v5_snapshot.py": {
        "git_blob": "a323ce28fe85afbb676c47135fa1e5685338dc0e",
        "sha256": "751ebc29837fdd5985f03fb54aec23f255485fc74ad90a6798bf2b6e73a7c952",
    },
}
# Two-commit release pin.  This exact implementation commit contains the
# snapshot-adoption authority closure, argv parser, supervisor preflight, tests,
# paired Slurm wrapper, and documentation.  The builder-only child commit may
# publish a controller only when this commit is a true execution ancestor.
SNAPSHOT_ADOPTION_RELEASE_COMMIT = "98c5125b8b68df8a8797c0228e85d9c8f45e1aed"
PAPER_PROTOCOL_RELEASE_COMMIT = "73923ce1cc822da3b592bebceb1ae90770d1f1cb"
PAPER_PROTOCOL_RELEASE_FILE_IDENTITIES = {
    "src/baselines/comrecgc/close_pair_scan.py": {
        "git_blob": "3aff0dc908d08a8da3f49d397f65084c80c7a37b",
        "sha256": "7fc151a16b9235fc698d448db5756e5c2455151955243443f1c28fe20582eb76",
    },
    "src/baselines/comrecgc/aids_pair_semantics.py": {
        "git_blob": "c3dfd7c5a963a20bd11a31d58177f522592ae8a0",
        "sha256": "478823410749e958f916490bf0f1aaa0633aeb2fbe62d4387b38ec70210a6636",
    },
    "scripts/autodl/run_aids_comrecgc_pair_semantics.py": {
        "git_blob": "0803199a1a3c99ed7c2cd8952ef23a54ec5fbee3",
        "sha256": "dd3fbce4b18cc1f95fdca42e5e3e2266208a2c1facab716b887409ef29f8ec2a",
    },
    "src/baselines/comrecgc/close_pair_view.py": {
        "git_blob": "3f9a7f070c1ae4576178e9435943f7bdb6e3a1cf",
        "sha256": "5d5a4619a055e30a94b720b836f16f5fafb0167463587e727780fef23d654b6f",
    },
    "scripts/baselines/comrecgc/build_close_pair_view.py": {
        "git_blob": "41bb56d95a8cfd5656f07a7f016e02ba66180273",
        "sha256": "b09cd3cb8e648018855559119a2d44549ab0161b812a2d47129a87d2179bd514",
    },
    "src/baselines/comrecgc/external_memory_dbscan.py": {
        "git_blob": "54fd31c1dfab46ff7b01b2871d48f52b664a7102",
        "sha256": "6e37c4289d950f1529f9bb1709b60040d24eeda528116b3baf70fce54d5b52dc",
    },
    "src/baselines/comrecgc/external_memory_recourse.py": {
        "git_blob": "71bc95019dce3b7fbed6c7ddc4a2a7ac5d080869",
        "sha256": "df449ba53bf8796a531551d865ce4e4082d16d73eb23d5bada169652618ff266",
    },
    "src/baselines/comrecgc/recourse.py": {
        "git_blob": "82d1f9e2e67e0d0cc87d4e9f4758db1afe8f8926",
        "sha256": "3c4d99dfa4bdb3f0cbeebb9fcc127e7e762f3eabbe7ddbb04c7326fe7536c836",
    },
    "scripts/baselines/comrecgc/run_common_recourse.py": {
        "git_blob": "a8b150f2d06a3d5f104114ba1f979613a3cccb95",
        "sha256": "da38387c8c6b921d051bd91a5850099ae194ee0010c72be84f16723121dc5834",
    },
    "scripts/autodl/run_comrecgc_standardized_continuation.py": {
        "git_blob": "a1efe5f2d0b1e3f837a69a1aaf35165767690a7c",
        "sha256": "95836505629a60a4c0fba2d9cc5fad516bfbb1600a77366f42d04ba0647991a6",
    },
    "scripts/autodl/run_comrecgc_standardized_continuation.sh": {
        "git_blob": "9dcc53c3eb7c3f6a10186ad6d6e2830d39d46b85",
        "sha256": "0197dab47a94d52a42d98b3a82fbc34bfbdbb244c3c1d221e8d8636180a19e94",
    },
    "scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh": {
        "git_blob": "926664bc03ef4ec5b3b47d47839dffd8930d4dfa",
        "sha256": "709f6a6b0ab3385d53b2af9715d743c68867fd140b276684a465233dfa2c45b5",
    },
    "scripts/autodl/run_four_gpu_recovery_controller.py": {
        "git_blob": "38a6a8688398e8681c6b6e31bcc1400ac547828f",
        "sha256": "21b0600e59417838085b7f5465eafb3a2337cbae026fadac2f91d76fed8ea957",
    },
    "src/utils/autodl_aids_greed_full_scan_supervisor.py": {
        "git_blob": "4a4c4a4072c3301174fdbfd975cc65ca8d171939",
        "sha256": "b082adeda87d4b31450fcf2f54765d3ab7d3a9e28d696d752c7a4d7497bbdfbe",
    },
    "scripts/autodl/run_aids_greed_full_scan_supervisor.py": {
        "git_blob": "6c067bce8da6d380d205e065099289ee97f9f888",
        "sha256": "b3286f8941a644311d466bd59777bc0216702b688c03815513da22ba4170998d",
    },
    "scripts/slurm/run_aids_greed_full_scan_supervisor.sh": {
        "git_blob": "9b421848240dafc098b0134f7230cdd58cdafbee",
        "sha256": "c936cf8a536330285b6afcfebb88647f6bcef2f577b50c4ff9dbd926a66b93d5",
    },
    "tests/autodl/test_aids_greed_full_scan_supervisor.py": {
        "git_blob": "d68f14a62289b489d4c185aadc7a7d0b8fecffbd",
        "sha256": "9182a53224cd86a374612340c5fcdd76eb1f7efa6c56452ed66b71e329b2a8b3",
    },
    "tests/autodl/test_four_gpu_recovery_controller.py": {
        "git_blob": "f00443fb3595e083de4ca7e7305136a14be0a411",
        "sha256": "0ef2ae0f77e495d5c4a221b7aa14fdec8a0a434bbfe1339c2bf8c0ef4df58365",
    },
    "scripts/autodl/manage_mut_traceoff_parity_v1.py": {
        "git_blob": "877e0419610e8261d13707272583a098cc2172a4",
        "sha256": "20606b28ca892500d04e4f1f27caff9b1878ee94f0eeb38754fd49a317720893",
    },
    "scripts/slurm/manage_mut_traceoff_parity_v1.sh": {
        "git_blob": "e8adbc1dbccc89e868b99e56d824eb68830fe9cc",
        "sha256": "e2339bcecfd17932daefa4b4d34183643a38f1f85be00bdd8743788f3bc8a6ca",
    },
    "src/utils/autodl_mut_traceoff_parity_v1.py": {
        "git_blob": "ba853462c39854005ee6f2feff9ae47eaaab4dc3",
        "sha256": "78f777fb52467afc5ae165cade720998e9f24e7dcf7fc5ce06674c4c0d28509e",
    },
    "tests/autodl/test_mut_traceoff_parity_v1.py": {
        "git_blob": "8c0e7dcc3144b6f221527640c5cca7fc42f9902a",
        "sha256": "1b68ac8a6b2032e51679a2b3f0b531116632ffad3fdce99ec6f82c81199b33a3",
    },
    "tests/autodl/test_aids_comrecgc_exact_route_v5.py": {
        "git_blob": "6e33246c2a6ebef8fdd92635d7cacb940247bb3b",
        "sha256": "f898b6d4974afcd957810d5e22e21febfb655763a5dd7eab3211440665729542",
    },
    "docs/AUTODL_AIDS_CLOSE_PAIR_MATERIALIZER.md": {
        "git_blob": "96f55dca50c8c5eb115c8252233aeb1c2304f84c",
        "sha256": "19fe25e6289f319f0764973c4e0623aa43fb119a7e6cfa91a3ed3fbff84f3b85",
    },
}
MINIMUM_CGROUP_FREE_BYTES = 128 * 1024**3
EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
EXPECTED_PAIR_COUNT = EXPECTED_PARENT_COUNT * EXPECTED_CANDIDATE_COUNT
EXPECTED_VECTOR_DIM = 64
EXPECTED_PARAMETERS = {
    "theta": 0.1,
    "delta": 0.02,
    "recourse_size": 100,
    "cf_size": 100000,
    "cluster_size": 3,
    "seed": 0,
}


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"{label} must be a physical nonempty file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RepairManifestError(f"{label} must be one JSON object")
    return payload


def _flag_value_options(
    command: tuple[str, ...], *, expected_prefix: tuple[str, ...]
) -> dict[str, str]:
    if command[: len(expected_prefix)] != expected_prefix:
        raise RepairManifestError("snapshot adoption command entrypoint changed")
    remainder = list(command[len(expected_prefix) :])
    if len(remainder) % 2:
        raise RepairManifestError("snapshot adoption command is not flag/value closed")
    options: dict[str, str] = {}
    for index in range(0, len(remainder), 2):
        flag = remainder[index]
        value = remainder[index + 1]
        if not flag.startswith("--") or flag in options:
            raise RepairManifestError("snapshot adoption command options changed")
        options[flag] = value
    return options


def _absolute(value: Any, *, label: str, kind: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RepairManifestError(f"{label} must be absolute")
    if kind == "fresh":
        return path.resolve(strict=False)
    if path.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink")
    resolved = path.resolve(strict=True)
    if kind == "file" and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise RepairManifestError(f"{label} must be a nonempty file")
    if kind == "dir" and not resolved.is_dir():
        raise RepairManifestError(f"{label} must be a directory")
    return resolved


def _git_head(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError("cannot resolve execution HEAD") from exc
    if len(value) != 40:
        raise RepairManifestError("execution HEAD is not a full SHA")
    return value


def _require_ancestor(project_root: Path, commit: str) -> dict[str, str]:
    head = _git_head(project_root)
    completed = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", commit, head],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RepairManifestError(
            f"required AIDS exact-route commit is not an ancestor: {commit}"
        )
    return {"required_commit": commit, "execution_head": head, "is_ancestor": "true"}


def _git_blob(project_root: Path, *, commit: str, relative_path: str) -> str:
    try:
        value = subprocess.check_output(
            [
                "git",
                "-C",
                str(project_root),
                "rev-parse",
                f"{commit}:{relative_path}",
            ],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError(
            f"cannot resolve reviewed core blob: {commit}:{relative_path}"
        ) from exc
    if len(value) != 40:
        raise RepairManifestError(
            f"reviewed core blob is not a full SHA: {relative_path}"
        )
    return value


def _require_reviewed_core_equivalence(project_root: Path) -> dict[str, Any]:
    ancestry = _require_ancestor(project_root, INTEGRATED_REVIEWED_CORE_COMMIT)
    files: dict[str, dict[str, str]] = {}
    for relative_path, expected in REVIEWED_CORE_FILE_IDENTITIES.items():
        integrated_blob = _git_blob(
            project_root,
            commit=INTEGRATED_REVIEWED_CORE_COMMIT,
            relative_path=relative_path,
        )
        if integrated_blob != expected["git_blob"]:
            raise RepairManifestError(
                f"integrated reviewed core blob changed: {relative_path}"
            )
        current_path = project_root / relative_path
        if current_path.is_symlink() or not current_path.is_file():
            raise RepairManifestError(
                f"reviewed core file is not physical: {relative_path}"
            )
        current_sha256 = sha256_file(current_path)
        files[relative_path] = {
            "reviewed_source_git_blob": expected["git_blob"],
            "integrated_git_blob": integrated_blob,
            "current_sha256": current_sha256,
        }
    return {
        "reviewed_source_commit": REVIEWED_SOURCE_CORE_COMMIT,
        "integrated_equivalent_commit": INTEGRATED_REVIEWED_CORE_COMMIT,
        "execution_head": ancestry["execution_head"],
        "integrated_commit_is_ancestor": True,
        "equivalence_basis": "ancestor-git-blob-with-protocol-extension",
        "current_extension_bound_by_protocol_release": True,
        "source_commit_object_required_at_build": False,
        "files": files,
    }


def _require_clean_worktree(project_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RepairManifestError("cannot verify execution worktree cleanliness")
    if completed.stdout.strip():
        raise RepairManifestError("execution worktree must be clean")
    return {"status": "PASS", "tracked_and_untracked_clean": True}


def _require_paper_protocol_release(project_root: Path) -> dict[str, Any]:
    ancestry = _require_ancestor(project_root, PAPER_PROTOCOL_RELEASE_COMMIT)
    files: dict[str, dict[str, str]] = {}
    for relative_path, expected in PAPER_PROTOCOL_RELEASE_FILE_IDENTITIES.items():
        release_blob = _git_blob(
            project_root,
            commit=PAPER_PROTOCOL_RELEASE_COMMIT,
            relative_path=relative_path,
        )
        current_path = project_root / relative_path
        if (
            release_blob != expected["git_blob"]
            or current_path.is_symlink()
            or not current_path.is_file()
            or sha256_file(current_path) != expected["sha256"]
        ):
            raise RepairManifestError(
                f"paper-protocol release content changed: {relative_path}"
            )
        files[relative_path] = {
            "release_git_blob": release_blob,
            "current_sha256": expected["sha256"],
        }
    return {
        "release_commit": PAPER_PROTOCOL_RELEASE_COMMIT,
        "execution_head": ancestry["execution_head"],
        "release_commit_is_ancestor": True,
        "files": files,
    }


def _require_snapshot_release(project_root: Path) -> dict[str, Any]:
    ancestry = _require_ancestor(project_root, SNAPSHOT_RELEASE_COMMIT)
    files: dict[str, dict[str, str]] = {}
    for relative_path, expected in SNAPSHOT_RELEASE_FILE_IDENTITIES.items():
        release_blob = _git_blob(
            project_root,
            commit=SNAPSHOT_RELEASE_COMMIT,
            relative_path=relative_path,
        )
        current_path = project_root / relative_path
        if release_blob != expected["git_blob"] or sha256_file(current_path) != expected[
            "sha256"
        ]:
            raise RepairManifestError(
                f"physical snapshot release content changed: {relative_path}"
            )
        files[relative_path] = {
            "release_git_blob": release_blob,
            "current_sha256": expected["sha256"],
        }
    return {
        "release_commit": SNAPSHOT_RELEASE_COMMIT,
        "execution_head": ancestry["execution_head"],
        "release_commit_is_ancestor": True,
        "files": files,
    }


def _terminal_source_evidence(
    *,
    root: Path,
    expected_sha256: str,
    proc_root: Path,
    allowed_snapshot_writer_pid: int,
) -> dict[str, Any]:
    if root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise RepairManifestError("terminal pair-store root must be a physical directory")
    root = root.resolve(strict=True)
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_symlink():
        raise RepairManifestError("terminal pair-store manifest may not be a symlink")
    manifest = _read_object(manifest_path, label="terminal pair-store manifest")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != expected_sha256:
        raise RepairManifestError("terminal pair-store manifest SHA256 mismatch")
    if (
        manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or manifest.get("run_complete") is not True
        or manifest.get("candidate_major_parent_minor_order") is not True
        or int(manifest.get("row_count", -1)) != EXPECTED_PAIR_COUNT
        or int(manifest.get("vector_dim", -1)) != EXPECTED_VECTOR_DIM
        or manifest.get("vectors_dtype") != "float32"
    ):
        raise RepairManifestError("terminal pair-store scientific shape is not AIDS v5")
    identity = manifest.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise RepairManifestError("terminal pair-store scientific identity is missing")
    parameters = identity.get("parameters")
    if (
        identity.get("dataset") != "aids"
        or identity.get("mode") != "full"
        or int(identity.get("parent_count", -1)) != EXPECTED_PARENT_COUNT
        or int(identity.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or identity.get("pair_order") != "candidate_major_parent_minor"
        or identity.get("device") != "cpu"
        or parameters != EXPECTED_PARAMETERS
    ):
        raise RepairManifestError("terminal pair-store scientific identity changed")
    for entry in root.iterdir():
        if entry.is_symlink():
            raise RepairManifestError(f"terminal pair store has a symlink: {entry}")
        if entry.name.endswith(".partial") or ".partial." in entry.name:
            raise RepairManifestError(f"PAIR_STORE_SOURCE_HAS_PARTIAL:{entry}")
    _validate_pair_store_manifest(manifest_path, manifest)
    pairs_path = Path(str(manifest["pairs_path"])).resolve(strict=True)
    vectors_path = Path(str(manifest["vectors_path"])).resolve(strict=True)
    writers = _find_writable_process_references(
        [manifest_path, pairs_path, vectors_path], proc_root=proc_root
    )
    if any(int(row.get("pid", -1)) != int(allowed_snapshot_writer_pid) for row in writers):
        raise RepairManifestError("terminal pair store has an unexpected live writer")
    return {
        "status": "PASS",
        "source_root": str(root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "pairs_path": str(pairs_path),
        "pairs_sha256": str(manifest["pairs_sha256"]),
        "pairs_stat": _file_stat_identity(pairs_path),
        "vectors_path": str(vectors_path),
        "vectors_sha256": str(manifest["vectors_sha256"]),
        "vectors_stat": _file_stat_identity(vectors_path),
        "row_count": EXPECTED_PAIR_COUNT,
        "parent_count": EXPECTED_PARENT_COUNT,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "vector_dim": EXPECTED_VECTOR_DIM,
        "source_guard": {
            "status": "PASS",
            "mode": "fresh_physical_snapshot_source_with_exact_old_writer",
            "source_owner_root": str(root),
            "allowed_writer_pid": int(allowed_snapshot_writer_pid),
            "writers": writers,
            "unexpected_writer_count": 0,
            "partial_count": 0,
            "symlink_count": 0,
            "source_direct_adoption_forbidden": True,
        },
    }


def _adopted_snapshot_evidence(
    *,
    spec: Mapping[str, Any],
    proc_root: Path,
    owner_namespace_root: Path,
    expected_source_root: Path,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    raw = spec.get("adopted_snapshot")
    if not isinstance(raw, Mapping):
        raise RepairManifestError("AIDS v5 adopted snapshot identity is missing")
    specified_namespace = _absolute(
        raw.get("owner_namespace_root"),
        label="snapshot owner namespace root",
        kind="dir",
    )
    if specified_namespace != owner_namespace_root:
        raise RepairManifestError("snapshot owner namespace is not control authority")
    owner_manifest = _absolute(
        raw.get("owner_manifest"), label="snapshot owner manifest", kind="file"
    )
    owner_manifest_sha = str(raw.get("owner_manifest_sha256") or "")
    if sha256_file(owner_manifest) != owner_manifest_sha:
        raise RepairManifestError("snapshot owner manifest SHA256 mismatch")
    owner = load_controller_manifest(owner_manifest)
    if owner.controller_id != SNAPSHOT_OWNER_CONTROLLER_ID:
        raise RepairManifestError("snapshot owner controller identity changed")
    if SNAPSHOT_OWNER_TASK_ID not in owner.by_id:
        raise RepairManifestError("snapshot owner task is missing")
    owner_task = owner.by_id[SNAPSHOT_OWNER_TASK_ID]
    gate_path = _absolute(
        raw.get("owner_task_gate"), label="snapshot owner task gate", kind="file"
    )
    gate_sha = str(raw.get("owner_task_gate_sha256") or "")
    expected_owner_manifest = (
        owner_namespace_root
        / "manifests"
        / f"{SNAPSHOT_OWNER_CONTROLLER_ID}.json"
    ).resolve(strict=False)
    expected_gate = (
        owner_namespace_root
        / SNAPSHOT_OWNER_CONTROLLER_ID
        / "tasks"
        / SNAPSHOT_OWNER_TASK_ID
        / "gate.json"
    ).resolve(strict=False)
    if owner_manifest != expected_owner_manifest or gate_path != expected_gate:
        raise RepairManifestError("snapshot owner authority path changed")
    if sha256_file(gate_path) != gate_sha:
        raise RepairManifestError("snapshot owner task gate SHA256 mismatch")
    gate = _read_object(gate_path, label="snapshot owner task gate")
    snapshot_root = _absolute(
        raw.get("snapshot_root"), label="adopted snapshot root", kind="dir"
    )
    expected_owner_output = owner_task.expected_output.replace("{attempt}", "0")
    runs = gate.get("runs")
    if (
        Path(expected_owner_output).resolve(strict=False) != snapshot_root
        or gate.get("status") != "PASS"
        or gate.get("task_id") != SNAPSHOT_OWNER_TASK_ID
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
        or runs[0].get("state") != "PASS"
        or runs[0].get("instance_id") != "main"
        or int(runs[0].get("attempt", -1)) != 0
        or Path(str(runs[0].get("expected_output") or "")).resolve(strict=False)
        != snapshot_root
    ):
        raise RepairManifestError("snapshot owner task is not exact PASS attempt-0")
    expected_hashes = {
        "snapshot_manifest.json": str(raw.get("snapshot_manifest_sha256") or ""),
        "dbscan_contract.json": str(raw.get("dbscan_contract_sha256") or ""),
        "pair_store/run_manifest.json": str(
            raw.get("pair_store_manifest_sha256") or ""
        ),
    }
    for relative, expected_sha in expected_hashes.items():
        artifact = _absolute(
            snapshot_root / relative,
            label=f"adopted snapshot {relative}",
            kind="file",
        )
        if sha256_file(artifact) != expected_sha:
            raise RepairManifestError(f"adopted snapshot hash changed: {relative}")
    snapshot_manifest = _read_object(
        snapshot_root / "snapshot_manifest.json", label="adopted snapshot manifest"
    )
    pair_manifest = _read_object(
        snapshot_root / "pair_store/run_manifest.json",
        label="adopted snapshot pair manifest",
    )
    snapshot_source = snapshot_manifest.get("source")
    pairs_sha = str(raw.get("pairs_sha256") or "")
    vectors_sha = str(raw.get("vectors_sha256") or "")
    if (
        snapshot_manifest.get("status") != "PASS"
        or snapshot_manifest.get("pair_store_manifest_sha256")
        != expected_hashes["pair_store/run_manifest.json"]
        or snapshot_manifest.get("dbscan_contract_sha256")
        != expected_hashes["dbscan_contract.json"]
        or pair_manifest.get("pairs_sha256") != pairs_sha
        or pair_manifest.get("vectors_sha256") != vectors_sha
        or pair_manifest.get("source_manifest_sha256")
        != expected_source_manifest_sha256
        or not isinstance(snapshot_source, Mapping)
        or snapshot_source.get("root") != str(expected_source_root)
        or int(pair_manifest.get("row_count", -1)) != EXPECTED_PAIR_COUNT
        or int(pair_manifest.get("vector_dim", -1)) != EXPECTED_VECTOR_DIM
    ):
        raise RepairManifestError("adopted snapshot scientific closure changed")
    pass_path = _absolute(
        snapshot_root / "PASS", label="adopted snapshot PASS", kind="file"
    )
    if pass_path.read_bytes() != b"PASS\n":
        raise RepairManifestError("adopted snapshot PASS marker changed")
    tree_files = [path for path in snapshot_root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in snapshot_root.rglob("*")):
        raise RepairManifestError("adopted snapshot tree contains a symlink")
    partials = [
        str(path)
        for path in snapshot_root.rglob("*")
        if ".partial" in path.name
    ]
    writers = _find_writable_process_references(tree_files, proc_root=proc_root)
    if partials or writers:
        raise RepairManifestError("adopted snapshot is not immutable/read-only")
    pairs_path = snapshot_root / "pair_store/pair_indices.npy"
    vectors_path = snapshot_root / "pair_store/recourse_vectors.npy"
    return {
        "status": "PASS",
        "owner_controller_id": owner.controller_id,
        "owner_namespace_root": str(owner_namespace_root),
        "owner_manifest": str(owner_manifest),
        "owner_manifest_sha256": owner_manifest_sha,
        "owner_task_id": SNAPSHOT_OWNER_TASK_ID,
        "owner_task_gate": str(gate_path),
        "owner_task_gate_sha256": gate_sha,
        "owner_task_status": "PASS",
        "owner_attempt": 0,
        "snapshot_root": str(snapshot_root),
        "source_root": str(expected_source_root),
        "source_manifest_sha256": expected_source_manifest_sha256,
        "snapshot_manifest_sha256": expected_hashes["snapshot_manifest.json"],
        "dbscan_contract_sha256": expected_hashes["dbscan_contract.json"],
        "pair_store_manifest_sha256": expected_hashes[
            "pair_store/run_manifest.json"
        ],
        "pairs_sha256": pairs_sha,
        "vectors_sha256": vectors_sha,
        "pairs_stat": _file_stat_identity(pairs_path),
        "vectors_stat": _file_stat_identity(vectors_path),
        "writable_reference_count": 0,
        "partial_count": 0,
        "copy_or_hardlink_performed": False,
    }


def _base_environment(
    *, manifest_path: Path, expected_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if sha256_file(manifest_path) != expected_sha256:
        raise RepairManifestError("base repair-v4 manifest SHA256 mismatch")
    manifest = load_controller_manifest(manifest_path)
    if manifest.controller_id != V4_CONTROLLER_ID or V4_TASK_ID not in manifest.by_id:
        raise RepairManifestError("base manifest is not the exact AIDS repair-v4 owner")
    task = manifest.by_id[V4_TASK_ID]
    environment = dict(task.environment)
    required = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
    }
    if any(environment.get(key) != value for key, value in required.items()):
        raise RepairManifestError("base repair-v4 scientific environment changed")
    for key in (
        "SOURCE_GENERATION_ROOT",
        "COMRECGC_UPSTREAM_ROOT",
        "DATASET_DIR",
        "SOURCE_CSV",
        "DISTANCE_CHECKPOINT",
        "DATASET_CSV",
        "TEACHER_PATH",
        "MOLCLR_ROOT",
        "MOLCLR_CHECKPOINT",
        "THRESHOLDS_PATH",
    ):
        value = Path(environment.get(key, "")).expanduser()
        if not value.is_absolute() or value.is_symlink() or not value.resolve(strict=True).exists():
            raise RepairManifestError(f"base scientific input is unavailable: {key}")
    environment.pop("AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES", None)
    environment.pop("AIDS_COMRECGC_V4_TEST_MODE", None)
    return environment, {
        "controller_id": manifest.controller_id,
        "task_id": task.task_id,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_sha256,
    }


def build_payload(*, spec_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _absolute(spec_path, label="v5 spec", kind="file")
    spec = _read_object(source, label="v5 spec")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("controller_id") != CONTROLLER_ID
        or spec.get("paper_frozen") is not True
        or spec.get("run_tastemolnet") != 0
        or spec.get("physical_snapshot_required") is not True
        or spec.get("snapshot_adoption_required") is not True
    ):
        raise RepairManifestError("invalid AIDS exact-route v5 spec identity")
    project_root = _absolute(spec.get("project_root"), label="project root", kind="dir")
    execution_commit = str(spec.get("execution_commit") or "")
    if _git_head(project_root) != execution_commit:
        raise RepairManifestError("v5 spec is not bound to execution HEAD")
    clean_worktree_gate = _require_clean_worktree(project_root)
    core_gate = _require_reviewed_core_equivalence(project_root)
    release_gate = _require_ancestor(project_root, ROUTE_RELEASE_COMMIT)
    snapshot_release_gate = _require_snapshot_release(project_root)
    snapshot_adoption_release_gate = _require_ancestor(
        project_root, SNAPSHOT_ADOPTION_RELEASE_COMMIT
    )
    paper_protocol_release_gate = _require_paper_protocol_release(project_root)
    if (
        SNAPSHOT_EXPECTED_ROWS != EXPECTED_PAIR_COUNT
        or SNAPSHOT_EXPECTED_VECTOR_DIM != EXPECTED_VECTOR_DIM
        or SNAPSHOT_EXPECTED_PARENT_COUNT != EXPECTED_PARENT_COUNT
        or SNAPSHOT_EXPECTED_CANDIDATE_COUNT != EXPECTED_CANDIDATE_COUNT
    ):
        raise RepairManifestError("snapshot and exact-route dimensions diverged")
    runtime_root = _absolute(spec.get("runtime_root"), label="runtime root", kind="dir")
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
    proc_root = _absolute(spec.get("proc_root"), label="proc root", kind="dir")
    cgroup_root = _absolute(
        spec.get("cgroup_memory_root"), label="cgroup memory root", kind="dir"
    )
    minimum_free = int(spec.get("min_cgroup_free_bytes", 0))
    if minimum_free < MINIMUM_CGROUP_FREE_BYTES:
        raise RepairManifestError("AIDS v5 cgroup headroom is below 128 GiB")
    limit = int((cgroup_root / "memory.limit_in_bytes").read_text().strip())
    usage = int((cgroup_root / "memory.usage_in_bytes").read_text().strip())
    if usage >= limit or limit - usage < minimum_free:
        raise RepairManifestError("AIDS v5 live cgroup headroom gate failed")
    python = _absolute(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError("configured Python is not executable")
    flock_bin = _absolute(spec.get("flock_bin"), label="flock", kind="file")
    if not os.access(flock_bin, os.X_OK):
        raise RepairManifestError("configured flock is not executable")
    scratch_root = _absolute(
        spec.get("local_scratch_root"), label="local scratch root", kind="dir"
    )
    route_lock = _absolute(spec.get("route_lock_path"), label="route lock", kind="fresh")
    try:
        route_lock.relative_to(scratch_root)
    except ValueError as exc:
        raise RepairManifestError("route lock must stay below local scratch root") from exc
    fresh_root = _absolute(spec.get("fresh_output_root"), label="fresh output", kind="fresh")
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError("AIDS v5 output root must be fresh")
    try:
        fresh_root.relative_to((runtime_root / "outputs/autodl").resolve(strict=False))
    except ValueError as exc:
        raise RepairManifestError("AIDS v5 output must stay below runtime outputs") from exc
    controller_root = (
        control_root / SOURCE_NAMESPACE / CONTROLLER_ID
    ).resolve(strict=False)
    if controller_root.exists() or controller_root.is_symlink():
        raise RepairManifestError("AIDS v5 controller root already exists")
    base_manifest = _absolute(
        spec.get("base_v4_manifest"), label="base repair-v4 manifest", kind="file"
    )
    base_sha = str(spec.get("base_v4_manifest_sha256") or "")
    environment, base_evidence = _base_environment(
        manifest_path=base_manifest, expected_sha256=base_sha
    )
    highmem_lock_logical = Path(
        str(environment.get("COMRECGC_HIGHMEM_LOCK_PATH") or "")
    ).expanduser()
    if not highmem_lock_logical.is_absolute() or highmem_lock_logical.is_symlink():
        raise RepairManifestError("AIDS v5 global high-memory lock must be physical")
    highmem_lock = highmem_lock_logical.resolve(strict=False)
    expected_highmem_lock = (
        runtime_root / "locks/comrecgc_common_recourse_highmem.lock"
    ).resolve(strict=False)
    if highmem_lock != expected_highmem_lock or highmem_lock.is_symlink():
        raise RepairManifestError("AIDS v5 global high-memory lock identity changed")
    allowed_old_raw = spec.get("allowed_old_read_only_process")
    if not isinstance(allowed_old_raw, Mapping):
        raise RepairManifestError("allowed old read-only process identity is missing")
    old_pid = int(allowed_old_raw.get("pid", 0))
    old_start_ticks = int(allowed_old_raw.get("start_ticks", 0))
    old_cmdline_sha256 = str(allowed_old_raw.get("cmdline_sha256") or "")
    old_output_root = _absolute(
        allowed_old_raw.get("output_root"), label="old v4 output root", kind="dir"
    )
    old_project_root = _absolute(
        allowed_old_raw.get("project_root"), label="old v4 project root", kind="dir"
    )
    terminal_root = _absolute(
        spec.get("terminal_pair_store_root"), label="terminal pair-store root", kind="dir"
    )
    try:
        terminal_root.relative_to(old_output_root)
    except ValueError as exc:
        raise RepairManifestError("terminal pair store is outside the bound old output") from exc
    process_gate = verify_process_set(
        proc_root=proc_root,
        allowed_pid=old_pid,
        allowed_start_ticks=old_start_ticks,
        allowed_cmdline_sha256=old_cmdline_sha256,
        allowed_output_root=old_output_root,
        allowed_project_root=old_project_root,
    )
    if (
        process_gate.get("process_set_status")
        != "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
        or int(process_gate.get("active_common_recourse_count", -1)) != 1
    ):
        raise RepairManifestError("old read-only process must be present at v5 build")
    terminal = _terminal_source_evidence(
        root=terminal_root,
        expected_sha256=str(spec.get("terminal_pair_store_manifest_sha256") or ""),
        proc_root=proc_root,
        allowed_snapshot_writer_pid=old_pid,
    )
    adopted_snapshot = _adopted_snapshot_evidence(
        spec=spec,
        proc_root=proc_root,
        owner_namespace_root=(control_root / SOURCE_NAMESPACE).resolve(strict=True),
        expected_source_root=terminal_root,
        expected_source_manifest_sha256=str(terminal["source_manifest_sha256"]),
    )
    generation_manifest = (
        Path(environment["SOURCE_GENERATION_ROOT"]) / "run_manifest.json"
    ).resolve(strict=True)
    pair_manifest = _read_object(
        terminal["source_manifest"], label="terminal pair-store manifest"
    )
    pair_identity = pair_manifest["scientific_identity"]
    if (
        pair_identity.get("generation_manifest_sha256")
        != sha256_file(generation_manifest)
        or pair_identity.get("distance_checkpoint_sha256")
        != sha256_file(environment["DISTANCE_CHECKPOINT"])
    ):
        raise RepairManifestError("terminal pair store does not match frozen v4 inputs")
    snapshot_output = fresh_root / "snapshot_adoption/attempt-{attempt}"
    snapshot_dependency_root = "{dep_" + SNAPSHOT_TASK_ID + "_output}"
    adopted_snapshot_root = str(adopted_snapshot["snapshot_root"])
    snapshot_pair_root = adopted_snapshot_root + "/pair_store"
    pair_semantics_benchmark_output = (
        fresh_root / "pair_semantics_benchmark/attempt-{attempt}"
    )
    pair_semantics_science_root = fresh_root / "pair_semantics_science"
    pair_semantics_output = (
        fresh_root / "pair_semantics_receipt/attempt-{attempt}"
    )
    pair_semantics_dependency_root = (
        "{dep_" + PAIR_SEMANTICS_TASK_ID + "_output}"
    )
    pair_semantics_receipt = (
        pair_semantics_dependency_root
        + "/pair_semantics_supervisor_receipt.json"
    )
    close_pair_view_output = fresh_root / "close_pair_view/attempt-{attempt}"
    close_pair_view_dependency_root = (
        "{dep_" + CLOSE_PAIR_VIEW_TASK_ID + "_output}"
    )
    environment.update(
        {
            "AUTODL_PYTHON": "{python}",
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cpu",
            "GPU_REQUIRED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
            "COMRECGC_COMMON_RECOURSE_RESUME": "1",
            "COMRECGC_EXTERNAL_MAX_RSS_GB": "96",
            "COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE": "8",
            "COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS": "1",
            "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
                "all_core_one_component_adaptive_anchor_v1"
            ),
            "COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT": "3",
            "COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP": "4096",
            "COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE": "65536",
            "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
            "COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE": "65536",
            "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
            "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
            "COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT": snapshot_pair_root,
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT": snapshot_pair_root,
            "COMRECGC_EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST": (
                close_pair_view_dependency_root + "/close_pair_contract.json"
            ),
            "COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB": "3",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT": str(proc_root),
            "COMRECGC_EXTERNAL_ROUTE_LOCK": str(route_lock),
            "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup_root),
            "COMRECGC_MIN_CGROUP_FREE_BYTES": str(minimum_free),
            "AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES": str(minimum_free),
            "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES": "1",
            "AIDS_COMRECGC_V5_ALLOWED_OLD_PID": str(old_pid),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS": str(old_start_ticks),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256": old_cmdline_sha256,
            "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT": str(old_output_root),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT": str(old_project_root),
            "AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT": snapshot_dependency_root,
            "AIDS_COMRECGC_V5_SNAPSHOT_ROOT": adopted_snapshot_root,
            "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST": str(
                adopted_snapshot["owner_manifest"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_NAMESPACE_ROOT": str(
                adopted_snapshot["owner_namespace_root"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256": str(
                adopted_snapshot["owner_manifest_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE": str(
                adopted_snapshot["owner_task_gate"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256": str(
                adopted_snapshot["owner_task_gate_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256": str(
                adopted_snapshot["snapshot_manifest_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256": str(
                adopted_snapshot["dbscan_contract_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256": str(
                adopted_snapshot["pair_store_manifest_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256": str(
                adopted_snapshot["pairs_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256": str(
                adopted_snapshot["vectors_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_ROOT": str(terminal_root),
            "AIDS_COMRECGC_V5_SNAPSHOT_SOURCE_MANIFEST_SHA256": str(
                terminal["source_manifest_sha256"]
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_PROC_ROOT": str(proc_root),
            "AIDS_COMRECGC_V5_SNAPSHOT_MIN_FREE_AFTER_BYTES": str(
                SNAPSHOT_MIN_FREE_AFTER_BYTES
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_ROWS": str(EXPECTED_PAIR_COUNT),
            "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_VECTOR_DIM": str(
                EXPECTED_VECTOR_DIM
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_PARENT_COUNT": str(
                EXPECTED_PARENT_COUNT
            ),
            "AIDS_COMRECGC_V5_SNAPSHOT_EXPECTED_CANDIDATE_COUNT": str(
                EXPECTED_CANDIDATE_COUNT
            ),
            "COMRECGC_FLOCK_BIN": str(flock_bin),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        }
    )
    for forbidden in (
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST",
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK",
    ):
        environment.pop(forbidden, None)
    expected_output = fresh_root / "cells/aids/comrecgc/standardized/attempt-{attempt}"
    threshold_path = Path(environment["THRESHOLDS_PATH"]).resolve(strict=True)
    threshold_sha256 = sha256_file(threshold_path)
    selector_task = {
        "id": SELECTOR_TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "runner_dataset": "paper-cell-aids-comrecgc-exact-route-v5",
        "runner_stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "depends_on": [],
        "resource": "cpu",
        "priority": 1,
        "manifest_only": True,
        "freezes_selector": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/write_aids_comrecgc_v5_selector_gate.py",
            "--config",
            "configs/hpc.yaml",
            "--thresholds",
            str(threshold_path),
            "--expected-sha256",
            threshold_sha256,
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(base_manifest),
        "config_files": [str(threshold_path)],
        "expected_output": str(fresh_root / "gates/selector/attempt-{attempt}"),
        "required_output_files": ["selector_gate.json", "PASS"],
        "required_log_marker": "[AIDS_COMRECGC_EXACT_ROUTE_V5_SELECTOR_PASS]",
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
    }
    snapshot_task = {
        "id": SNAPSHOT_TASK_ID,
        "dataset": "aids",
        # Adoption reopens held-out bytes only after selector freeze.  It does
        # not copy, link, mutate, reorder, or recompute those bytes.
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-exact-route-v5",
        "runner_stage": "AM_COMRECGC_PAIR_STORE_SNAPSHOT_ADOPTION",
        "depends_on": [SELECTOR_TASK_ID],
        "resource": "cpu",
        "priority": 0,
        "manifest_only": False,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py",
            "--config",
            "configs/hpc.yaml",
            "--output-dir",
            "{task_output}",
            "--proc-root",
            str(proc_root),
            "--owner-manifest",
            str(adopted_snapshot["owner_manifest"]),
            "--owner-manifest-sha256",
            str(adopted_snapshot["owner_manifest_sha256"]),
            "--owner-namespace-root",
            str(adopted_snapshot["owner_namespace_root"]),
            "--owner-task-gate",
            str(adopted_snapshot["owner_task_gate"]),
            "--owner-task-gate-sha256",
            str(adopted_snapshot["owner_task_gate_sha256"]),
            "--snapshot-root",
            adopted_snapshot_root,
            "--snapshot-manifest-sha256",
            str(adopted_snapshot["snapshot_manifest_sha256"]),
            "--dbscan-contract-sha256",
            str(adopted_snapshot["dbscan_contract_sha256"]),
            "--pair-store-manifest-sha256",
            str(adopted_snapshot["pair_store_manifest_sha256"]),
            "--pairs-sha256",
            str(adopted_snapshot["pairs_sha256"]),
            "--vectors-sha256",
            str(adopted_snapshot["vectors_sha256"]),
            "--source-root",
            str(terminal_root),
            "--source-manifest-sha256",
            str(terminal["source_manifest_sha256"]),
            "--allowed-pid",
            str(old_pid),
            "--allowed-start-ticks",
            str(old_start_ticks),
            "--allowed-cmdline-sha256",
            old_cmdline_sha256,
            "--allowed-output-root",
            str(old_output_root),
            "--allowed-project-root",
            str(old_project_root),
            "--expected-row-count",
            str(EXPECTED_PAIR_COUNT),
            "--expected-vector-dim",
            str(EXPECTED_VECTOR_DIM),
            "--expected-parent-count",
            str(EXPECTED_PARENT_COUNT),
            "--expected-candidate-count",
            str(EXPECTED_CANDIDATE_COUNT),
        ],
        "input_manifest": str(adopted_snapshot["owner_manifest"]),
        "config_files": [
            str(base_manifest),
            str(adopted_snapshot["owner_task_gate"]),
        ],
        "expected_output": str(snapshot_output),
        "required_output_files": [
            "snapshot_adoption_manifest.json",
            "PASS",
        ],
        "required_log_marker": "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_PASS]",
        "environment": {
            "GPU_REQUIRED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
        "semantic_failure_markers": [
            "snapshot owner",
            "snapshot full closure validation failed",
            "snapshot adoption",
            "live writer",
        ],
    }
    pair_semantics_common_command = [
        "{python}",
        "{project_root}/scripts/autodl/run_aids_comrecgc_pair_semantics.py",
        "--config",
        "configs/hpc.yaml",
        "--project-root",
        "{project_root}",
        "--upstream-root",
        environment["COMRECGC_UPSTREAM_ROOT"],
        "--dataset-dir",
        environment["DATASET_DIR"],
        "--source-csv",
        environment["SOURCE_CSV"],
        "--generation-dir",
        environment["SOURCE_GENERATION_ROOT"],
        "--distance-checkpoint",
        environment["DISTANCE_CHECKPOINT"],
        "--pair-store-manifest",
        snapshot_pair_root + "/run_manifest.json",
        "--expected-pair-store-manifest-sha256",
        str(adopted_snapshot["pair_store_manifest_sha256"]),
        "--parent-limit",
        str(EXPECTED_PARENT_COUNT),
        "--theta",
        str(EXPECTED_PARAMETERS["theta"]),
        "--device",
        "cpu",
        "--distance-batch-size",
        "128",
    ]
    pair_semantics_environment = {
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUN_TASTEMOLNET": "0",
    }
    pair_semantics_benchmark_task = {
        "id": PAIR_SEMANTICS_BENCHMARK_TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-original-protocol-v1",
        "runner_stage": "AM_COMRECGC_GREED_CLOSE_PAIR_BENCHMARK",
        "depends_on": [SELECTOR_TASK_ID, SNAPSHOT_TASK_ID],
        "resource": "cpu",
        "priority": 2,
        "manifest_only": False,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            *pair_semantics_common_command,
            "--output-dir",
            "{task_output}",
            "--max-chunks",
            "2",
            "--skip-source-array-hash-verification",
        ],
        "input_manifest": snapshot_pair_root + "/run_manifest.json",
        "config_files": [str(base_manifest)],
        # The benchmark and the full scan intentionally use distinct fresh
        # roots.  The generic exp_run launcher rejects any nonempty output
        # root, so cross-task resume would fail before the full scan starts.
        # Recomputing 2/560 chunks is bounded and preserves no-clobber launch
        # semantics for both tasks.
        "expected_output": str(pair_semantics_benchmark_output),
        "required_output_files": ["benchmark_result.json", "progress.json"],
        "required_log_marker": "BENCHMARK_COMPLETE_NOT_SCIENTIFIC_PASS",
        "environment": dict(pair_semantics_environment),
        "semantic_failure_markers": [
            "pair-store provenance contract failed",
            "physical pair axes differ",
            "dataset fingerprint differs",
            "source binding failed",
        ],
    }
    pair_semantics_task = {
        "id": PAIR_SEMANTICS_TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-original-protocol-v1",
        "runner_stage": "AM_COMRECGC_GREED_CLOSE_PAIR_FULL",
        "depends_on": [PAIR_SEMANTICS_BENCHMARK_TASK_ID],
        "resource": "cpu",
        "priority": 2,
        "manifest_only": False,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/run_aids_greed_full_scan_supervisor.py",
            "--config",
            "configs/hpc.yaml",
            "--project-root",
            "{project_root}",
            "--execution-commit",
            execution_commit,
            "--campaign-root",
            str(fresh_root),
            "--science-root",
            str(pair_semantics_science_root),
            "--receipt-output",
            "{task_output}",
            "--proc-root",
            str(proc_root),
            "--max-same-root-resumes",
            "1",
            "--",
            *pair_semantics_common_command,
            "--output-dir",
            str(pair_semantics_science_root),
        ],
        "input_manifest": snapshot_pair_root + "/run_manifest.json",
        "config_files": [str(base_manifest)],
        "expected_output": str(pair_semantics_output),
        "required_output_files": [
            "pair_semantics_supervisor_receipt.json",
            "PASS",
        ],
        "required_log_marker": "[AIDS_GREED_FULL_SCAN_SUPERVISOR_PASS]",
        "retry_on_process_loss": True,
        "environment": dict(pair_semantics_environment),
        "semantic_failure_markers": [
            "pair-store provenance contract failed",
            "physical pair axes differ",
            "read-only source stat identity changed",
            "direct pair-store SHA256 differs",
            "non-finite values",
            "[AIDS_GREED_FULL_SCAN_SEMANTIC_FAILURE]",
            "semantic/provenance failure forbids resume",
        ],
    }
    close_pair_view_task = {
        "id": CLOSE_PAIR_VIEW_TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-original-protocol-v1",
        "runner_stage": "AM_COMRECGC_THETA_CLOSE_VIEW",
        "depends_on": [PAIR_SEMANTICS_TASK_ID],
        "resource": "cpu",
        "priority": 2,
        "manifest_only": False,
        "data_splits": ["test"],
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "{python}",
            "{project_root}/scripts/baselines/comrecgc/build_close_pair_view.py",
            "--config",
            "configs/hpc.yaml",
            "--pair-semantics-contract",
            str(pair_semantics_science_root / "close_pair_contract.json"),
            "--pair-semantics-receipt",
            pair_semantics_receipt,
            "--expected-pair-semantics-science-root",
            str(pair_semantics_science_root),
            "--expected-execution-commit",
            execution_commit,
            "--physical-vectors",
            snapshot_pair_root + "/recourse_vectors.npy",
            "--normalized-distances",
            (
                str(
                    pair_semantics_science_root
                    / "distance_scan/normalized_distances.greed.float32.npy"
                )
            ),
            "--all-pairs-close-certificate",
            str(pair_semantics_science_root / "all_pairs_close_certificate.json"),
            "--output-dir",
            "{task_output}",
            "--max-compact-gb",
            "8",
        ],
        "input_manifest": pair_semantics_receipt,
        "config_files": [str(base_manifest)],
        "expected_output": str(close_pair_view_output),
        "required_output_files": ["close_pair_contract.json", "PASS"],
        "required_log_marker": "[COMRECGC_CLOSE_PAIR_VIEW_PASS]",
        "environment": dict(pair_semantics_environment),
        "semantic_failure_markers": [
            "CLOSE_PAIR_VIEW_BLOCKED_STORAGE",
            "all-pairs-close certificate",
            "physical vector source",
            "logical theta-close manifest",
        ],
    }
    task = {
        "id": TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-exact-route-v5",
        "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
        "depends_on": [
            SELECTOR_TASK_ID,
            SNAPSHOT_TASK_ID,
            PAIR_SEMANTICS_TASK_ID,
            CLOSE_PAIR_VIEW_TASK_ID,
        ],
        "resource": "cpu",
        "priority": 1,
        "data_splits": ["test"],
        "manifest_only": False,
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh",
        ],
        # The logical theta-close view is the scientific DBSCAN input.  Keep
        # the physical snapshot as a separately hashed config/source and let
        # exp_run bind its primary input hash to the close-view manifest.
        "input_manifest": close_pair_view_dependency_root + "/close_pair_contract.json",
        "config_files": [
            str(base_manifest),
            environment["THRESHOLDS_PATH"],
            snapshot_pair_root + "/run_manifest.json",
        ],
        "expected_output": str(expected_output),
        "required_output_files": list(STANDARDIZED_REQUIRED_FILES),
        "required_log_marker": "[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset=aids",
        "environment": environment,
        "semantic_failure_markers": [
            "source closure changed",
            "live writer",
            "sklearn version mismatch",
            "rss budget exceeded",
            "exact_dbscan_complexity_blocked",
            "UNPROVEN_CARTESIAN_DBSCAN_INPUT",
            "logical theta-close manifest",
            "test leakage",
        ],
    }
    contract = {
        "schema_version": SPEC_SCHEMA,
        "spec_path": str(source),
        "spec_sha256": sha256_file(source),
        "execution_project_root": str(project_root),
        "execution_commit": execution_commit,
        "reviewed_core_gate": core_gate,
        "clean_worktree_gate": clean_worktree_gate,
        "route_release_gate": release_gate,
        "snapshot_release_gate": snapshot_release_gate,
        "snapshot_adoption_release_gate": snapshot_adoption_release_gate,
        "paper_protocol_release_gate": paper_protocol_release_gate,
        "base_v4": base_evidence,
        "terminal_pair_store": terminal,
        "snapshot_adoption": {
            "task_id": SNAPSHOT_TASK_ID,
            "expected_output": str(snapshot_output),
            "adopted_snapshot": adopted_snapshot,
            "source_root": str(terminal_root),
            "source_manifest_sha256": str(terminal["source_manifest_sha256"]),
            "proc_root": str(proc_root),
            "expected_rows": EXPECTED_PAIR_COUNT,
            "expected_vector_dim": EXPECTED_VECTOR_DIM,
            "expected_parent_count": EXPECTED_PARENT_COUNT,
            "expected_candidate_count": EXPECTED_CANDIDATE_COUNT,
            "copy_mode": "read_only_existing_snapshot_adoption_no_copy",
            "atomic_no_clobber_promotion": False,
            "source_hardlinks_forbidden": True,
            "copy_or_hardlink_performed": False,
            "source_writer_policy": "only_exact_frozen_old_generation_or_natural_exit",
            "old_v4_signal_authorized": False,
            "dbscan_contract_required": True,
            "full_closure_reopened_before_adoption_pass": True,
            "full_closure_reopened_before_science": True,
        },
        "original_protocol": {
            "official_comrecgc_commit": (
                "122f9341a360e9f06bb58a2f5823bb596021f6bf"
            ),
            "physical_pair_count": EXPECTED_PAIR_COUNT,
            "dbscan_input": "theta_close_recourse_vectors_only",
            "theta": EXPECTED_PARAMETERS["theta"],
            "filter_operator": "<=",
            "delta": EXPECTED_PARAMETERS["delta"],
            "min_samples": EXPECTED_PARAMETERS["cluster_size"],
            "metric": "euclidean",
            "self_neighbor_counted": True,
            "pair_semantics_benchmark_task_id": (
                PAIR_SEMANTICS_BENCHMARK_TASK_ID
            ),
            "pair_semantics_benchmark_expected_output": str(
                pair_semantics_benchmark_output
            ),
            "pair_semantics_task_id": PAIR_SEMANTICS_TASK_ID,
            "pair_semantics_expected_output": str(pair_semantics_output),
            "pair_semantics_receipt_name": (
                "pair_semantics_supervisor_receipt.json"
            ),
            "pair_semantics_science_root": str(pair_semantics_science_root),
            "close_pair_view_task_id": CLOSE_PAIR_VIEW_TASK_ID,
            "close_pair_view_expected_output": str(close_pair_view_output),
            "benchmark_chunks": 2,
            "benchmark_is_scientific_pass": False,
            "full_scan_resumes_benchmark_root": False,
            "full_scan_recomputes_benchmark_chunks": True,
            "full_scan_fixed_science_root_across_controller_attempts": True,
            "full_scan_receipt_root_is_attempt_qualified": True,
            "full_scan_same_root_resume_limit": 1,
            "partial_close_compact_copy_limit_gb": 8,
            "full_25gb_copy_forbidden": True,
            "physical_pair_store_regenerated": False,
            "physical_pair_store_copied": False,
        },
        "fresh_output_root": str(fresh_root),
        "gpu_required": False,
        "parameters": EXPECTED_PARAMETERS,
        "strict_flip": True,
        "test_loaded_after_selector_freeze": True,
        "old_v4_mutated": False,
        "old_v4_signal_authorized": False,
        "allowed_old_read_only_process": process_gate,
        "highmem_exclusion": {
            "global_highmem_lock_held_at_build": False,
            "reason": (
                "a monitored waiter is queued before science and takes the "
                "same physical lock when old v4 exits naturally"
            ),
            "global_highmem_lock_path": str(highmem_lock),
            "lock_handover_queued_before_science": True,
            "lock_handover_helper_generation_monitored": True,
            "lock_retained_until_supervisor_exit": True,
            "old_read_only_consumer_is_the_only_colocation_exception": True,
            "v5_route_lock_held": True,
            "per_attempt_cgroup_headroom_gate_bytes": minimum_free,
            "cgroup_headroom_gate": {
                "root": str(cgroup_root),
                "limit_path": str(cgroup_root / "memory.limit_in_bytes"),
                "usage_path": str(cgroup_root / "memory.usage_in_bytes"),
                "limit_bytes_at_build": limit,
                "usage_bytes_at_build": usage,
                "free_bytes_at_build": limit - usage,
                "semantics": "memory.limit_in_bytes-minus-memory.usage_in_bytes",
                "host_memfree_used": False,
                "revalidated_before_and_during_each_attempt": True,
            },
            "v5_rss_budget_gib": 96,
            "process_set_revalidated_before_every_attempt": True,
            "mut_dependency_blocks_until_v5_pass": True,
        },
        "mut_dependency": {
            "controller_id": CONTROLLER_ID,
            "controller_manifest": str(
                control_root
                / SOURCE_NAMESPACE
                / "manifests"
                / f"{CONTROLLER_ID}.json"
            ),
            "controller_root": str(
                control_root / SOURCE_NAMESPACE / CONTROLLER_ID
            ),
            "task_id": TASK_ID,
            "task_gate": str(
                control_root
                / SOURCE_NAMESPACE
                / CONTROLLER_ID
                / "tasks"
                / TASK_ID
                / "gate.json"
            ),
            "passing_output_resolution": "unique_PASS_run.expected_output",
            "attempt_is_controller_authoritative": True,
            "terminal_marker": "PASS",
        },
    }
    payload = {
        "schema_version": 1,
        "controller_id": CONTROLLER_ID,
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "auto",
            "max_cpu_tasks": 1,
            "launch_grace_seconds": 180,
            "max_transient_retries": 1,
            "keep_alive_when_blocked": True,
        },
        "resource_gates": {
            "min_available_ram_gb": 128,
            "min_free_disk_gb": 20,
            "max_cpu_load_fraction": 0.95,
        },
        "aids_comrecgc_exact_route_v5_contract": contract,
        "tasks": [
            selector_task,
            snapshot_task,
            pair_semantics_benchmark_task,
            pair_semantics_task,
            close_pair_view_task,
            task,
        ],
    }
    validation = validate_payload(payload)
    return payload, {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_id": TASK_ID,
        "fresh_output_root": str(fresh_root),
        "mut_dependency_task_gate": contract["mut_dependency"]["task_gate"],
        "gpu_required": False,
        "task_count": validation["task_count"],
    }


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aids-comrecgc-exact-v5-") as directory:
        path = Path(directory) / "manifest.json"
        v3._atomic_json(path, payload)
        manifest = load_controller_manifest(path)
    if manifest.controller_id != CONTROLLER_ID or set(manifest.by_id) != {
        SNAPSHOT_TASK_ID,
        SELECTOR_TASK_ID,
        PAIR_SEMANTICS_BENCHMARK_TASK_ID,
        PAIR_SEMANTICS_TASK_ID,
        CLOSE_PAIR_VIEW_TASK_ID,
        TASK_ID,
    }:
        raise RepairManifestError(
            "AIDS v5 paper protocol must contain exactly six serialized tasks"
        )
    if int(manifest.runtime.get("max_transient_retries", -1)) != 1:
        raise RepairManifestError(
            "AIDS v5 controller must permit one fresh receipt transient retry"
        )
    task = manifest.by_id[TASK_ID]
    selector = manifest.by_id[SELECTOR_TASK_ID]
    snapshot_task = manifest.by_id[SNAPSHOT_TASK_ID]
    benchmark_task = manifest.by_id[PAIR_SEMANTICS_BENCHMARK_TASK_ID]
    pair_semantics_task = manifest.by_id[PAIR_SEMANTICS_TASK_ID]
    close_pair_view_task = manifest.by_id[CLOSE_PAIR_VIEW_TASK_ID]
    if (
        selector.stage != "AM_COMRECGC_THRESHOLD_FREEZE"
        or not selector.freezes_selector
        or task.depends_on
        != (
            SELECTOR_TASK_ID,
            SNAPSHOT_TASK_ID,
            PAIR_SEMANTICS_TASK_ID,
            CLOSE_PAIR_VIEW_TASK_ID,
        )
    ):
        raise RepairManifestError("AIDS v5 selector-freeze dependency changed")
    if snapshot_task.resource != "cpu":
        raise RepairManifestError("AIDS v5 snapshot adoption must be CPU-only")
    if snapshot_task.depends_on != (SELECTOR_TASK_ID,):
        raise RepairManifestError("AIDS v5 snapshot must follow selector freeze")
    if (
        snapshot_task.data_splits != ("test",)
        or not snapshot_task.read_only_test
        or not snapshot_task.selector_parameters_frozen
        or snapshot_task.input_manifest
        != str(
            payload["aids_comrecgc_exact_route_v5_contract"]["snapshot_adoption"][
                "adopted_snapshot"
            ]["owner_manifest"]
        )
        or snapshot_task.expected_output
        != str(
            payload["aids_comrecgc_exact_route_v5_contract"]["snapshot_adoption"][
                "expected_output"
            ]
        )
        or snapshot_task.required_output_files
        != ("snapshot_adoption_manifest.json", "PASS")
        or snapshot_task.required_log_marker
        != "[AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_PASS]"
    ):
        raise RepairManifestError("AIDS v5 snapshot adoption artifact contract changed")
    expected_benchmark_output = str(
        payload["aids_comrecgc_exact_route_v5_contract"]["original_protocol"][
            "pair_semantics_benchmark_expected_output"
        ]
    )
    expected_pair_output = str(
        payload["aids_comrecgc_exact_route_v5_contract"]["original_protocol"][
            "pair_semantics_expected_output"
        ]
    )
    protocol_contract = payload["aids_comrecgc_exact_route_v5_contract"]
    expected_science_root = str(
        protocol_contract["original_protocol"]["pair_semantics_science_root"]
    )
    expected_campaign_root = str(protocol_contract["fresh_output_root"])
    expected_execution_commit = str(protocol_contract["execution_commit"])
    expected_proc_root = str(protocol_contract["snapshot_adoption"]["proc_root"])
    expected_pair_environment = {
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUN_TASTEMOLNET": "0",
    }
    if (
        benchmark_task.resource != "cpu"
        or benchmark_task.depends_on != (SELECTOR_TASK_ID, SNAPSHOT_TASK_ID)
        or benchmark_task.expected_output != expected_benchmark_output
        or benchmark_task.expected_output == pair_semantics_task.expected_output
        or benchmark_task.required_output_files
        != ("benchmark_result.json", "progress.json")
        or benchmark_task.required_log_marker
        != "BENCHMARK_COMPLETE_NOT_SCIENTIFIC_PASS"
        or benchmark_task.command[-5:]
        != (
            "--output-dir",
            "{task_output}",
            "--max-chunks",
            "2",
            "--skip-source-array-hash-verification",
        )
        or benchmark_task.environment != expected_pair_environment
    ):
        raise RepairManifestError("AIDS close-pair benchmark contract changed")
    if (
        pair_semantics_task.resource != "cpu"
        or pair_semantics_task.depends_on
        != (PAIR_SEMANTICS_BENCHMARK_TASK_ID,)
        or pair_semantics_task.expected_output != expected_pair_output
        or pair_semantics_task.required_output_files
        != (
            "pair_semantics_supervisor_receipt.json",
            "PASS",
        )
        or pair_semantics_task.required_log_marker
        != "[AIDS_GREED_FULL_SCAN_SUPERVISOR_PASS]"
        or not pair_semantics_task.retry_on_process_loss
        or pair_semantics_task.input_manifest != benchmark_task.input_manifest
        or not pair_semantics_task.input_manifest.endswith(
            "/pair_store/run_manifest.json"
        )
        or pair_semantics_task.environment != expected_pair_environment
    ):
        raise RepairManifestError("AIDS full close-pair scan contract changed")
    expected_supervisor_command = (
        "{python}",
        "{project_root}/scripts/autodl/run_aids_greed_full_scan_supervisor.py",
        "--config",
        "configs/hpc.yaml",
        "--project-root",
        "{project_root}",
        "--execution-commit",
        expected_execution_commit,
        "--campaign-root",
        expected_campaign_root,
        "--science-root",
        expected_science_root,
        "--receipt-output",
        "{task_output}",
        "--proc-root",
        expected_proc_root,
        "--max-same-root-resumes",
        "1",
        "--",
        *benchmark_task.command[:-5],
        "--output-dir",
        expected_science_root,
    )
    if pair_semantics_task.command != expected_supervisor_command:
        raise RepairManifestError("AIDS GREED supervisor command contract changed")
    expected_close_output = str(
        payload["aids_comrecgc_exact_route_v5_contract"]["original_protocol"][
            "close_pair_view_expected_output"
        ]
    )
    if (
        close_pair_view_task.resource != "cpu"
        or close_pair_view_task.depends_on != (PAIR_SEMANTICS_TASK_ID,)
        or close_pair_view_task.expected_output != expected_close_output
        or close_pair_view_task.required_output_files
        != ("close_pair_contract.json", "PASS")
        or close_pair_view_task.required_log_marker
        != "[COMRECGC_CLOSE_PAIR_VIEW_PASS]"
        or "--all-pairs-close-certificate" not in close_pair_view_task.command
        or "--max-compact-gb" not in close_pair_view_task.command
    ):
        raise RepairManifestError("AIDS theta-close view contract changed")
    compact_index = close_pair_view_task.command.index("--max-compact-gb")
    pair_dependency = "{dep_" + PAIR_SEMANTICS_TASK_ID + "_output}"
    snapshot_pair_root = str(Path(pair_semantics_task.input_manifest).parent)
    receipt_manifest = pair_dependency + "/pair_semantics_supervisor_receipt.json"
    expected_close_command = (
        "{python}",
        "{project_root}/scripts/baselines/comrecgc/build_close_pair_view.py",
        "--config",
        "configs/hpc.yaml",
        "--pair-semantics-contract",
        expected_science_root + "/close_pair_contract.json",
        "--pair-semantics-receipt",
        receipt_manifest,
        "--expected-pair-semantics-science-root",
        expected_science_root,
        "--expected-execution-commit",
        expected_execution_commit,
        "--physical-vectors",
        snapshot_pair_root + "/recourse_vectors.npy",
        "--normalized-distances",
        expected_science_root + "/distance_scan/normalized_distances.greed.float32.npy",
        "--all-pairs-close-certificate",
        expected_science_root + "/all_pairs_close_certificate.json",
        "--output-dir",
        "{task_output}",
        "--max-compact-gb",
        "8",
    )
    if (
        close_pair_view_task.command != expected_close_command
        or close_pair_view_task.command[compact_index + 1] != "8"
        or close_pair_view_task.input_manifest
        != receipt_manifest
        or close_pair_view_task.environment != expected_pair_environment
    ):
        raise RepairManifestError("AIDS theta-close compact-copy bound changed")
    if task.resource != "cpu" or task.command != (
        "bash",
        "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh",
    ):
        raise RepairManifestError("AIDS v5 task launch contract changed")
    contract = payload.get("aids_comrecgc_exact_route_v5_contract")
    if not isinstance(contract, Mapping):
        raise RepairManifestError("AIDS v5 contract is missing")
    clean_gate = contract.get("clean_worktree_gate")
    protocol_release_gate = contract.get("paper_protocol_release_gate")
    if (
        not isinstance(clean_gate, Mapping)
        or clean_gate.get("status") != "PASS"
        or clean_gate.get("tracked_and_untracked_clean") is not True
        or not isinstance(protocol_release_gate, Mapping)
        or protocol_release_gate.get("release_commit")
        != PAPER_PROTOCOL_RELEASE_COMMIT
        or protocol_release_gate.get("release_commit_is_ancestor") is not True
        or set(protocol_release_gate.get("files") or {})
        != set(PAPER_PROTOCOL_RELEASE_FILE_IDENTITIES)
    ):
        raise RepairManifestError("AIDS paper-protocol release gate changed")
    highmem_contract = contract.get("highmem_exclusion")
    if not isinstance(highmem_contract, Mapping):
        raise RepairManifestError("AIDS v5 high-memory contract is missing")
    cgroup_contract = highmem_contract.get("cgroup_headroom_gate")
    if not isinstance(cgroup_contract, Mapping):
        raise RepairManifestError("AIDS v5 cgroup contract is missing")
    cgroup_contract_root = Path(str(cgroup_contract.get("root") or ""))
    if (
        str(cgroup_contract.get("limit_path"))
        != str(cgroup_contract_root / "memory.limit_in_bytes")
        or str(cgroup_contract.get("usage_path"))
        != str(cgroup_contract_root / "memory.usage_in_bytes")
        or int(cgroup_contract.get("free_bytes_at_build", -1))
        != int(cgroup_contract.get("limit_bytes_at_build", -1))
        - int(cgroup_contract.get("usage_bytes_at_build", -1))
        or int(cgroup_contract.get("free_bytes_at_build", -1))
        < MINIMUM_CGROUP_FREE_BYTES
    ):
        raise RepairManifestError("AIDS v5 cgroup evidence changed")
    required = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
        "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
        "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
            "all_core_one_component_adaptive_anchor_v1"
        ),
        "COMRECGC_EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST": (
            "{dep_"
            + CLOSE_PAIR_VIEW_TASK_ID
            + "_output}/close_pair_contract.json"
        ),
        "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES": "1",
        "COMRECGC_HIGHMEM_LOCK_PATH": str(
            highmem_contract.get("global_highmem_lock_path")
        ),
        "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup_contract.get("root")),
        "COMRECGC_MIN_CGROUP_FREE_BYTES": str(
            highmem_contract.get("per_attempt_cgroup_headroom_gate_bytes")
        ),
        "AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES": str(
            highmem_contract.get("per_attempt_cgroup_headroom_gate_bytes")
        ),
        "RUN_TASTEMOLNET": "0",
    }
    if any(task.environment.get(key) != value for key, value in required.items()):
        raise RepairManifestError("AIDS v5 production environment is incomplete")
    snapshot_contract = contract.get("snapshot_adoption")
    if not isinstance(snapshot_contract, Mapping):
        raise RepairManifestError("AIDS v5 snapshot adoption contract is missing")
    adopted = snapshot_contract.get("adopted_snapshot")
    if not isinstance(adopted, Mapping) or adopted.get("status") != "PASS":
        raise RepairManifestError("AIDS v5 adopted snapshot evidence is missing")
    science_adoption_environment = {
        "AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT": (
            "{dep_" + SNAPSHOT_TASK_ID + "_output}"
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_ROOT": str(adopted.get("snapshot_root")),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST": str(
            adopted.get("owner_manifest")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_NAMESPACE_ROOT": str(
            adopted.get("owner_namespace_root")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_MANIFEST_SHA256": str(
            adopted.get("owner_manifest_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE": str(
            adopted.get("owner_task_gate")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_OWNER_TASK_GATE_SHA256": str(
            adopted.get("owner_task_gate_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_MANIFEST_SHA256": str(
            adopted.get("snapshot_manifest_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_DBSCAN_SHA256": str(
            adopted.get("dbscan_contract_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_PAIR_MANIFEST_SHA256": str(
            adopted.get("pair_store_manifest_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_PAIRS_SHA256": str(
            adopted.get("pairs_sha256")
        ),
        "AIDS_COMRECGC_V5_SNAPSHOT_VECTORS_SHA256": str(
            adopted.get("vectors_sha256")
        ),
    }
    if any(
        task.environment.get(key) != value
        for key, value in science_adoption_environment.items()
    ):
        raise RepairManifestError("AIDS v5 science adoption identity drifted")
    adoption_options = _flag_value_options(
        snapshot_task.command,
        expected_prefix=(
            "{python}",
            "{project_root}/scripts/autodl/adopt_aids_comrecgc_v5_snapshot.py",
        ),
    )
    expected_adoption_options = {
        "--config": "configs/hpc.yaml",
        "--output-dir": "{task_output}",
        "--proc-root": str(snapshot_contract.get("proc_root")),
        "--owner-manifest": str(adopted.get("owner_manifest")),
        "--owner-manifest-sha256": str(adopted.get("owner_manifest_sha256")),
        "--owner-namespace-root": str(adopted.get("owner_namespace_root")),
        "--owner-task-gate": str(adopted.get("owner_task_gate")),
        "--owner-task-gate-sha256": str(adopted.get("owner_task_gate_sha256")),
        "--snapshot-root": str(adopted.get("snapshot_root")),
        "--snapshot-manifest-sha256": str(
            adopted.get("snapshot_manifest_sha256")
        ),
        "--dbscan-contract-sha256": str(adopted.get("dbscan_contract_sha256")),
        "--pair-store-manifest-sha256": str(
            adopted.get("pair_store_manifest_sha256")
        ),
        "--pairs-sha256": str(adopted.get("pairs_sha256")),
        "--vectors-sha256": str(adopted.get("vectors_sha256")),
        "--source-root": str(snapshot_contract.get("source_root")),
        "--source-manifest-sha256": str(
            snapshot_contract.get("source_manifest_sha256")
        ),
        "--allowed-pid": str(
            contract["allowed_old_read_only_process"].get("allowed_pid")
        ),
        "--allowed-start-ticks": str(
            contract["allowed_old_read_only_process"].get("allowed_start_ticks")
        ),
        "--allowed-cmdline-sha256": str(
            contract["allowed_old_read_only_process"].get(
                "allowed_cmdline_sha256"
            )
        ),
        "--allowed-output-root": str(
            contract["allowed_old_read_only_process"].get("allowed_output_root")
        ),
        "--allowed-project-root": str(
            contract["allowed_old_read_only_process"].get("allowed_project_root")
        ),
        "--expected-row-count": str(EXPECTED_PAIR_COUNT),
        "--expected-vector-dim": str(EXPECTED_VECTOR_DIM),
        "--expected-parent-count": str(EXPECTED_PARENT_COUNT),
        "--expected-candidate-count": str(EXPECTED_CANDIDATE_COUNT),
    }
    if adoption_options != expected_adoption_options or any(
        snapshot_task.environment.get(key) != value
        for key, value in {
            "GPU_REQUIRED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        }.items()
    ):
        raise RepairManifestError("AIDS v5 snapshot adoption command changed")
    dependency_root = "{dep_" + SNAPSHOT_TASK_ID + "_output}"
    adopted_root = str(adopted.get("snapshot_root"))
    if (
        task.environment.get("COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT")
        != adopted_root + "/pair_store"
        or task.environment.get("COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT")
        != adopted_root + "/pair_store"
        or task.environment.get("AIDS_COMRECGC_V5_SNAPSHOT_ROOT") != adopted_root
        or task.environment.get("AIDS_COMRECGC_V5_SNAPSHOT_ADOPTION_ROOT")
        != dependency_root
        or task.input_manifest
        != "{dep_"
        + CLOSE_PAIR_VIEW_TASK_ID
        + "_output}/close_pair_contract.json"
        or adopted_root + "/pair_store/run_manifest.json" not in task.config_files
        or task.environment.get("COMRECGC_EXTERNAL_CLOSE_PAIR_VIEW_MANIFEST")
        != "{dep_"
        + CLOSE_PAIR_VIEW_TASK_ID
        + "_output}/close_pair_contract.json"
    ):
        raise RepairManifestError("AIDS v5 science does not consume the exact snapshot")
    if any(
        key in task.environment
        for key in (
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST",
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK",
        )
    ):
        raise RepairManifestError("AIDS v5 production task contains a fallback bypass")
    process_contract = contract.get("allowed_old_read_only_process")
    if not isinstance(process_contract, Mapping):
        raise RepairManifestError("AIDS v5 old-process contract is missing")
    process_environment = {
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PID": str(process_contract.get("allowed_pid")),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS": str(
            process_contract.get("allowed_start_ticks")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256": str(
            process_contract.get("allowed_cmdline_sha256")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT": str(
            process_contract.get("allowed_output_root")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT": str(
            process_contract.get("allowed_project_root")
        ),
    }
    if any(
        task.environment.get(key) != value
        for key, value in process_environment.items()
    ):
        raise RepairManifestError("AIDS v5 old-process environment drifted")
    if (
        contract.get("parameters") != EXPECTED_PARAMETERS
        or contract.get("gpu_required") is not False
        or contract.get("old_v4_mutated") is not False
        or contract.get("old_v4_signal_authorized") is not False
        or snapshot_contract.get("source_hardlinks_forbidden") is not True
        or snapshot_contract.get("atomic_no_clobber_promotion") is not False
        or snapshot_contract.get("copy_mode")
        != "read_only_existing_snapshot_adoption_no_copy"
        or snapshot_contract.get("copy_or_hardlink_performed") is not False
        or snapshot_contract.get("source_writer_policy")
        != "only_exact_frozen_old_generation_or_natural_exit"
        or snapshot_contract.get("old_v4_signal_authorized") is not False
        or snapshot_contract.get("dbscan_contract_required") is not True
        or snapshot_contract.get("full_closure_reopened_before_adoption_pass")
        is not True
        or snapshot_contract.get("full_closure_reopened_before_science") is not True
        or int(snapshot_contract.get("expected_rows", -1)) != EXPECTED_PAIR_COUNT
        or int(snapshot_contract.get("expected_vector_dim", -1))
        != EXPECTED_VECTOR_DIM
        or process_contract.get("status") != "PASS"
        or highmem_contract.get(
            "process_set_revalidated_before_every_attempt"
        )
        is not True
        or highmem_contract.get(
            "lock_handover_queued_before_science"
        )
        is not True
        or highmem_contract.get(
            "lock_handover_helper_generation_monitored"
        )
        is not True
        or highmem_contract.get(
            "lock_retained_until_supervisor_exit"
        )
        is not True
        or cgroup_contract.get("semantics")
        != "memory.limit_in_bytes-minus-memory.usage_in_bytes"
        or cgroup_contract.get("host_memfree_used")
        is not False
        or cgroup_contract.get("revalidated_before_and_during_each_attempt")
        is not True
    ):
        raise RepairManifestError("AIDS v5 scientific/safety contract changed")
    terminal = contract.get("terminal_pair_store")
    if not isinstance(terminal, Mapping) or terminal.get("status") != "PASS":
        raise RepairManifestError("AIDS v5 terminal-source evidence is missing")
    return {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_id": TASK_ID,
        "task_count": 6,
        "gpu_required": False,
        "manifest_sha256": manifest.sha256,
    }


def build_manifest(*, spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    destination = _absolute(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"manifest output must be fresh: {destination}")
    payload, summary = build_payload(spec_path=spec_path)
    spec = _read_object(spec_path, label="v5 spec")
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
    expected = (
        control_root / SOURCE_NAMESPACE / "manifests" / f"{CONTROLLER_ID}.json"
    ).resolve(strict=False)
    if destination != expected:
        raise RepairManifestError(f"manifest output must be exact: {expected}")
    validation = validate_payload(payload)
    v3._atomic_json(destination, payload)
    frozen = load_controller_manifest(destination)
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("published AIDS v5 manifest changed")
    return {
        **summary,
        "manifest": str(destination),
        "manifest_sha256": frozen.sha256,
    }


__all__ = [
    "CONTROLLER_ID",
    "CLOSE_PAIR_VIEW_TASK_ID",
    "EXPECTED_PAIR_COUNT",
    "INTEGRATED_REVIEWED_CORE_COMMIT",
    "PAIR_SEMANTICS_BENCHMARK_TASK_ID",
    "PAIR_SEMANTICS_TASK_ID",
    "PAPER_PROTOCOL_RELEASE_COMMIT",
    "PAPER_PROTOCOL_RELEASE_FILE_IDENTITIES",
    "REVIEWED_CORE_COMMIT",
    "REVIEWED_CORE_FILE_IDENTITIES",
    "REVIEWED_SOURCE_CORE_COMMIT",
    "ROUTE_RELEASE_COMMIT",
    "SELECTOR_TASK_ID",
    "SNAPSHOT_RELEASE_COMMIT",
    "SNAPSHOT_RELEASE_FILE_IDENTITIES",
    "SNAPSHOT_TASK_ID",
    "SPEC_SCHEMA",
    "TASK_ID",
    "build_manifest",
    "build_payload",
    "validate_payload",
]
