from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest


pytest.importorskip("networkx")
pytest.importorskip("pandas")

from src.baselines.globalgce_hpc_exact import (  # noqa: E402
    build_partition_manifest,
    canonical_sha256,
    merge_exact_shards,
    run_exact_reference,
    run_mining_shard,
    verify_exact_parity,
)
from src.baselines.globalgce_hpc_storage_safe import (  # noqa: E402
    StorageSafeT8Error,
    build_storage_safe_archive,
    merge_package_storage_safe,
    publish_storage_safe_archive,
    storage_admission,
    stream_verify_storage_safe_bundle,
    validate_storage_path_policy,
)
import src.baselines.globalgce_hpc_storage_safe as storage_safe  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SRC = Path(
    os.environ.get(
        "GLOBALGCE_OFFICIAL_SRC",
        str(ROOT / "baselines/globalgce_official/src"),
    )
)
EXECUTION_COMMIT = "b" * 40
OFFICIAL_COMMIT = "157e65c2850bc787f229a1ee8c60564906b933f2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_sha(payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_graphs(path: Path) -> None:
    rows = [
        {
            "graph_id": 0,
            "nodes": [{"id": index, "label": 0} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
                {"source": 0, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 1,
            "nodes": [{"id": index, "label": 0} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
        {
            "graph_id": 2,
            "nodes": [{"id": index, "label": 2} for index in range(3)],
            "edges": [
                {"source": 0, "target": 1, "label": 1},
                {"source": 1, "target": 2, "label": 1},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_input_manifest(path: Path, graph_path: Path) -> None:
    graph_sha = _sha(graph_path)
    mining: dict[str, object] = {
        "source_label": 1,
        "seed": 7,
        "epochs": 100,
        "min_support": 1,
        "min_vertices": 2,
        "max_vertices": 3,
        "top_k": 20,
        "root_count": 2,
        "exact": True,
        "approximate_pruning": False,
    }
    payload: dict[str, object] = {
        "state": "PASS",
        "dataset": "tastemolnet",
        "method": "globalgce",
        "stage": "EXACT_GSPAN_CPU_INPUT",
        "route_kind": "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD",
        "split_scope": "train_only",
        "calibration_payload_included": False,
        "test_payload_included": False,
        "matrix_publication_allowed_from_hpc": False,
        "official_globalgce_commit": OFFICIAL_COMMIT,
        "source_commit": "a" * 40,
        "mining_config": mining,
        "mining_config_sha256": _line_sha(mining),
        "hpc_runtime_config": {"name": "hpc.yaml", "sha256": "c" * 64},
        "files": [{"role": "graph_jsonl", "sha256": graph_sha}],
        "transaction_binding": {
            "shared_transaction_database": True,
            "target_labels": [0, 2],
            "target_semantics_do_not_modify_transaction_database": True,
            "graph_jsonl_sha256": graph_sha,
            "graph_count": 3,
        },
        "transfer_policy": {
            "source_data_is_train_only_derived": True,
            "hpc_may_modify_autodl_matrix": False,
        },
    }
    payload["manifest_sha256"] = _line_sha(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _manifest(
    tmp_path: Path,
    *,
    name: str,
    included_roots: tuple[int, ...] | None,
) -> tuple[Path, dict]:
    graphs = tmp_path / "graphs.jsonl"
    if not graphs.exists():
        _write_graphs(graphs)
    input_manifest = tmp_path / "input-manifest.json"
    if not input_manifest.exists():
        _write_input_manifest(input_manifest, graphs)
    output = tmp_path / name
    report = build_partition_manifest(
        graph_jsonl=graphs,
        input_manifest=input_manifest,
        expected_commit=EXECUTION_COMMIT,
        official_src=OFFICIAL_SRC,
        output=output,
        shard_count=2,
        min_support=1,
        min_vertices=2,
        max_vertices=3,
        top_k=20,
        split_root_indices=(0,),
        split_depth=2,
        canary_root_indices=(0,),
        included_root_indices=included_roots,
    )
    return output, report


def _run_shards(tmp_path: Path, manifest_path: Path, manifest: dict, name: str) -> Path:
    root = tmp_path / f"{name}-shards"
    for index in range(manifest["shard_count"]):
        run_mining_shard(
            partition_manifest=manifest_path,
            shard_index=index,
            output_root=root / f"shard-{index:03d}",
            flush_every=1,
        )
    return root


def _parity_and_full_shards(tmp_path: Path) -> tuple[Path, Path, Path]:
    canary_path, canary = _manifest(
        tmp_path,
        name="canary-manifest.json",
        included_roots=(0,),
    )
    canary_shards = _run_shards(tmp_path, canary_path, canary, "canary")
    canary_merge = tmp_path / "canary-merge"
    merge_exact_shards(
        partition_manifest=canary_path,
        shards_root=canary_shards,
        output_root=canary_merge,
    )
    reference = tmp_path / "reference"
    run_exact_reference(
        partition_manifest=canary_path,
        output_root=reference,
        flush_every=1,
    )
    parity = tmp_path / "parity.json"
    verify_exact_parity(
        partition_manifest=canary_path,
        reference_root=reference,
        merged_root=canary_merge,
        output=parity,
    )
    full_path, full = _manifest(
        tmp_path,
        name="full-manifest.json",
        included_roots=None,
    )
    full_shards = _run_shards(tmp_path, full_path, full, "full")
    return full_path, full_shards, parity


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths: list[Path] = []
    for name in ("environment", "slurm", "resources"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"schema_version": f"tiny_{name}_v1", "status": "PASS"}),
            encoding="utf-8",
        )
        paths.append(path)
    return paths[0], paths[1], paths[2]


def test_storage_safe_merge_roundtrip_preserves_all_exact_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, shards, parity = _parity_and_full_shards(tmp_path)
    environment, slurm, resources = _evidence(tmp_path)
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    persistent = tmp_path / "persistent" / "result"
    monkeypatch.setattr(
        storage_safe,
        "filesystem_free_bytes",
        lambda _path: 20 * 1024**3,
    )
    report = merge_package_storage_safe(
        partition_manifest=manifest,
        shards_root=shards,
        parity_receipt=parity,
        environment_manifest=environment,
        slurm_inventory=slurm,
        resource_metrics=resources,
        packaging_commit=EXECUTION_COMMIT,
        scratch_root=scratch,
        output_root=persistent,
        require_distinct_filesystems=False,
        minimum_reserve_bytes=0,
        reserve_fraction=0.0,
    )
    assert report["state"] == "PASS"
    assert report["matrix_write_enabled"] is False
    assert {path.name for path in persistent.iterdir()} == {
        "t8_exact_result_bundle.tar.gz",
        "result_manifest.json",
    }
    verified = stream_verify_storage_safe_bundle(
        persistent / "t8_exact_result_bundle.tar.gz",
        receipt_path=persistent / "result_manifest.json",
    )
    assert verified["status"] == "PASS"
    assert verified["streaming_verification"] is True
    assert verified["extracted_to_disk"] is False
    with tarfile.open(persistent / "t8_exact_result_bundle.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "merge/events.jsonl",
        "merge/patterns.jsonl",
        "merge/rejection_events.jsonl",
    } <= names


def test_compressed_archive_is_byte_deterministic(tmp_path: Path) -> None:
    manifest, shards, parity = _parity_and_full_shards(tmp_path)
    environment, slurm, resources = _evidence(tmp_path)
    merge = tmp_path / "merge"
    merge_exact_shards(
        partition_manifest=manifest,
        shards_root=shards,
        output_root=merge,
    )
    archives = [tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"]
    for archive in archives:
        build_storage_safe_archive(
            partition_manifest=manifest,
            merge_root=merge,
            parity_receipt=parity,
            environment_manifest=environment,
            slurm_inventory=slurm,
            resource_metrics=resources,
            packaging_commit=EXECUTION_COMMIT,
            output_archive=archive,
        )
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_enospc_admission_fails_before_publication(tmp_path: Path) -> None:
    free_bytes = 6_450_000_000
    decision = storage_admission(
        required_bytes=8 * 1024**3,
        free_bytes=free_bytes,
        minimum_reserve_bytes=2 * 1024**3,
        reserve_fraction=0.20,
    )
    assert decision.state == "BLOCKED_INSUFFICIENT_PERSISTENT_SPACE"
    assert decision.free_bytes == free_bytes
    assert decision.reserve_bytes == 2 * 1024**3
    assert decision.shortfall_bytes == 8 * 1024**3 - decision.usable_bytes
    assert not (tmp_path / "result").exists()


def test_ssdfs_is_forbidden_even_when_cli_wrapper_is_bypassed() -> None:
    with pytest.raises(StorageSafeT8Error, match="may not use /ssdfs"):
        validate_storage_path_policy(
            "/ssdfs/datahome/u20526/t8-result",
            label="persistent result",
        )


def test_persistent_enospc_leaves_no_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_archive = tmp_path / "scratch.tar.gz"
    scratch_archive.write_bytes(b"closed compressed artifact")
    destination = tmp_path / "persistent" / "result"
    monkeypatch.setattr(storage_safe, "filesystem_free_bytes", lambda _path: 0)
    inner_manifest = {
        "bundle_content_sha256": "a" * 64,
        "merge_result_sha256": "b" * 64,
        "packaging_commit": "c" * 40,
        "manifest_sha256": "d" * 64,
        "scientific_input_sha256": "e" * 64,
        "event_count": 1,
        "pattern_count": 1,
        "rejection_count": 0,
        "compression": {"algorithm": "gzip", "lossless": True},
    }
    verification = {
        "schema_version": "globalgce_hpc_storage_safe_verification_v1",
        "status": "PASS",
        "archive_bytes": scratch_archive.stat().st_size,
        "archive_sha256": _sha(scratch_archive),
        "bundle_content_sha256": inner_manifest["bundle_content_sha256"],
        "streaming_verification": True,
        "extracted_to_disk": False,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    with pytest.raises(StorageSafeT8Error, match="BLOCKED_HPC_USER_QUOTA_SHORTFALL"):
        publish_storage_safe_archive(
            scratch_archive=scratch_archive,
            inner_manifest=inner_manifest,
            prepublication_verification=verification,
            output_root=destination,
        )
    assert not destination.exists()
    assert list(destination.parent.glob(".*.incomplete")) == []


def test_storage_safe_slurm_job_is_afterok_cpu_only_and_matrix_inert() -> None:
    script = ROOT / "scripts/hpc/t8/slurm_storage_safe_merge_package.sh"
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert '--dependency="afterok:${T8_FULL_ARRAY_JOB_ID}"' in text
    assert "SLURM_TMPDIR" in text
    assert "run_storage_safe_merge_package.py" in text
    assert "build_result_bundle.py" not in text
    assert "T8_FULL_MERGE_ROOT" not in text
    assert "matrix-authority" not in text.lower()
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize(
    "name",
    [
        "run_storage_safe_merge_package.sh",
        "stream_verify_storage_safe_bundle.sh",
        "slurm_storage_safe_merge_package.sh",
    ],
)
def test_storage_safe_entrypoints_have_cpu_only_slurm_wrappers(name: str) -> None:
    script = ROOT / "scripts/slurm" / name
    text = script.read_text(encoding="utf-8")
    assert "CPU-only" in text
    assert "#SBATCH --partition=intel" in text
    assert "#SBATCH --gres" not in text
    assert "source ~/.bashrc" in text
    assert "conda activate smiles_pip118" in text
    assert "export PYTHONPATH=\"$PWD\"" in text
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_stream_verifier_fails_closed_on_archive_corruption(tmp_path: Path) -> None:
    archive = tmp_path / "broken.tar.gz"
    archive.write_bytes(b"not a gzip stream")
    with pytest.raises((StorageSafeT8Error, tarfile.TarError, OSError)):
        stream_verify_storage_safe_bundle(archive)
